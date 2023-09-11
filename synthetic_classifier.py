import os

import lightning.pytorch as pl
import torch
import torchmetrics
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision import models, utils
from typing_extensions import Literal

import wandb
from load_timi_dataset import LoadTimiDataset
from models.attention_vgg16 import AttentionVgg16
from models.blocks.attention_block import visualize_attention
from utils.timi_tts_constants import AUDIO_KEY, CLASS_KEY, LABELS_MAP


class SyntheticClassifier(pl.LightningModule):
    global_step = 0

    def __init__(self, pretrained: bool, metadata_file: Literal[
        "clean.csv", "dtw.csv", "aug.csv", "aug_dtw.csv", "clean_reduced.csv", "dtw_reduced.csv", "aug_reduced.csv", "aug_dtw_reduced.csv"],
                 model_name: Literal["resnet18", "resnet34", "resnet50", "attVgg16"],
                 is_validation_enabled: bool = True,
                 lr: float = 0.0005,
                 decay: float = 0.00002, momentum: float = 0.99, batch_size: int = 128,
                 optimizer: str = "SGD",
                 is_gpu_enabled: bool = False, mode: Literal["normal", "reduced"] = "normal"
                 ):
        super().__init__()
        self.mode = mode
        self.metadata_file = metadata_file
        self.n_classes = 2 if self.mode == "reduced" else 12  # TODO add dyanamic num classes
        self.model_name = model_name
        self.is_validation_enabled = is_validation_enabled
        self.test_set = None
        self.validation_set = None
        self.training_set = None
        self.pretrained = pretrained
        self.lr = lr
        self.optimizer = optimizer
        self.decay = decay
        self.momentum = momentum
        self.batch_size = batch_size
        self.model = self.load_model()
        self.is_gpu_enabled = is_gpu_enabled
        self.loss_module = torch.nn.CrossEntropyLoss()
        self.outputs = []

        load_dotenv()

    def load_model(self):
        if self.model_name == "resnet18" or self.model_name == "resnet34":
            backbone = models.resnet18(
                pretrained=self.pretrained) if self.model_name == "resnet18" else models.resnet34(
                pretrained=self.pretrained)
            # change output from 1000 categories to 12
            backbone.fc = nn.Linear(512, self.n_classes)
            # changed to accept 1 channel img
            backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        elif self.model_name == "resnet50":
            backbone = models.resnet50(pretrained=self.pretrained)
            backbone.fc = nn.Linear(2048, self.n_classes)
            backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif self.model_name == "attVgg16":
            backbone = AttentionVgg16(num_classes=self.n_classes, normalize_attn=True, pretrained=self.pretrained)
        else:
            raise Exception("Unsupported model ", self.mode)
        return backbone

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            optimizer = Adam(self.model.parameters(),
                             lr=self.lr, weight_decay=self.decay)
        elif self.optimizer == "SGD":
            optimizer = SGD(self.model.parameters(), self.lr,
                            self.momentum, weight_decay=self.decay)
            # learning rate scheduler
        return optimizer

    def forward(self, audio):
        return self.model(audio)

    def setup(self, stage: str = None):

        if stage == "fit" or stage is None:
            self.training_set = LoadTimiDataset(base_path=os.getenv("TIMI-TTS-ROOT-DIR"),
                                                metadata_file_path=self.metadata_file, partition="training",
                                                model_name="attVgg16",
                                                transform=True, is_validation_enabled=self.is_validation_enabled)
            if self.is_validation_enabled:
                self.validation_set = LoadTimiDataset(base_path=os.getenv("TIMI-TTS-ROOT-DIR"),
                                                      metadata_file_path=self.metadata_file, partition="validation",
                                                      model_name="attVgg16",
                                                      transform=True)
        if stage == "test" or stage is None:
            self.test_set = LoadTimiDataset(base_path=os.getenv("TIMI-TTS-ROOT-DIR"),
                                            metadata_file_path=self.metadata_file, partition="test",
                                            model_name="attVgg16",
                                            transform=True)

    def _get_preds_loss_accuracy(self, batch, batch_idx):
        audios = batch[AUDIO_KEY]
        labels = batch[CLASS_KEY]
        if self.model_name == "attVgg16":
            logits, _, _ = self.model(audios)
        else:
            logits = self.model(audios)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss_module(logits, labels)
        acc = torchmetrics.functional.accuracy(preds, labels, "multiclass",
                                               num_classes=self.n_classes)
        f1 = torchmetrics.functional.f1_score(preds, labels, "multiclass",
                                              num_classes=self.n_classes)

        if self.current_epoch == 1 and batch_idx == 10 and self.model_name == "attVgg16":
            _, attFilter1, attFilter2 = self.model(audios[0:8])

            I_train = utils.make_grid(audios[0:8], nrow=8, normalize=True,
                                      scale_each=True)
            orig = visualize_attention(I_train, attFilter1, up_factor=2, no_attention=True)
            first = visualize_attention(I_train, attFilter1, up_factor=2, no_attention=False)
            second = visualize_attention(I_train, attFilter2, up_factor=4, no_attention=False)

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
            ax1.imshow(orig)
            ax2.imshow(first)
            ax3.imshow(second)
            ax1.title.set_text('Input Images')
            ax2.title.set_text('pool-3 attention')
            ax3.title.set_text('pool-4 attention')
            plt.pause(0.001)
            plt.savefig('vgg16_attentions.png')
        return preds, loss, acc, f1

    def _get_probs_preds_loss_accuracy(self, batch):
        audios = batch[AUDIO_KEY]
        labels = batch[CLASS_KEY]

        if self.model_name == "attVgg16":
            logits, _, _ = self.model(audios)
        else:
            logits = self.model(audios)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss_module(logits, labels)
        acc = torchmetrics.functional.accuracy(preds, labels, "multiclass",
                                               num_classes=self.n_classes)
        f1 = torchmetrics.functional.f1_score(preds, labels, "multiclass",
                                              num_classes=self.n_classes)

        return probs, logits, loss, acc, f1

    def training_step(self, train_batch, batch_idx):
        x = train_batch[AUDIO_KEY]
        n = x.shape[0]
        self.global_step += n
        _, loss, acc, f1 = self._get_preds_loss_accuracy(train_batch, batch_idx)

        self.log('train_acc', acc)
        self.log('train_loss', loss)
        self.log('train_f1', f1)

        return loss

    def validation_step(self, val_batch, batch_idx):
        preds, loss, acc, f1 = self._get_preds_loss_accuracy(val_batch, batch_idx)

        self.log('val_acc', acc)
        self.log('val_loss', loss)
        self.log('val_f1', f1)
        return preds

    def test_step(self, test_batch, batch_idx):

        probs, preds, loss, acc, f1 = self._get_probs_preds_loss_accuracy(
            test_batch)

        self.log('test_acc', acc)
        self.log('test_loss', loss)
        self.log('test_f1', f1)

        return {"preds": preds, "probs": probs, "y_true": test_batch[CLASS_KEY]}

    def on_test_batch_end(
            self, outputs, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:

        self.outputs.append(outputs)

    def on_test_epoch_end(self):
        preds = torch.cat([tmp['preds'] for tmp in self.outputs]).cpu().numpy()
        probs = torch.cat([tmp['probs'] for tmp in self.outputs]).cpu().numpy()
        y_true = torch.cat([tmp['y_true'] for tmp in self.outputs]).tolist()
        all_class_names = LABELS_MAP.keys()
        cm = wandb.plot.confusion_matrix(
            y_true=y_true,
            probs=probs,
            class_names=["glowtts", "fastpitch"] if self.mode == "reduced" else all_class_names
        )
        wandb.log({"confusion_matrix": cm})

        # wandb.sklearn.plot_confusion_matrix(y_true, preds, class_names)

        wandb.log({"roc": wandb.plot.roc_curve(y_true, probs)})
        self.outputs.clear()

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        audios = batch[AUDIO_KEY]
        logits = self.model(audios)
        preds = torch.softmax(logits, dim=1)
        return preds

    def train_dataloader(self):
        return DataLoader(self.training_set, batch_size=self.batch_size, shuffle=False, num_workers=4,
                          pin_memory=self.is_gpu_enabled)

    def val_dataloader(self):
        return DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=False, num_workers=4,
                          pin_memory=self.is_gpu_enabled)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=4,
                          pin_memory=self.is_gpu_enabled)


class _EarlyStopping(EarlyStopping, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    isPretrained = True
    optimizer = "Adam"
    lr = 0.001
    decay = 0.00008
    batch_size = 32
    mode = "reduced"
    metadata_file = "aug_reduced.csv"
    # model_name = "resnet50"
    model_name = "attVgg16"
    isValidationEnabled = True

    epochs = 25
    infos = ""

    isGpuEnabled = False

    classifier = SyntheticClassifier(metadata_file=metadata_file, model_name=model_name, pretrained=isPretrained, lr=lr,
                                     decay=decay,
                                     batch_size=batch_size,
                                     optimizer=optimizer,
                                     is_gpu_enabled=isGpuEnabled, mode=mode, is_validation_enabled=isValidationEnabled)

    # Disable_stats=True or BSOD :()
    wandb.init(settings=wandb.Settings(
        _disable_stats=True),
        config={"epochs": epochs, "learning_rate": lr, "batch_size": batch_size, "optimizer": optimizer,
                "isPretrained": isPretrained,
                "decay": decay, "metadata_file": metadata_file, "model_name": model_name}, project="test_relazione")

    logger = WandbLogger(log_model=True)

    early_stop_callback = _EarlyStopping(monitor="val_loss" if isValidationEnabled else "train_loss", mode="min",
                                         patience=3, min_delta=0.005)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=epochs,
        log_every_n_steps=1,
        accelerator="gpu" if isGpuEnabled else "cpu",
        devices="auto",
        # max_time="00:08:00:00",
        callbacks=[early_stop_callback],
        limit_val_batches=1.0 if isValidationEnabled else 0.0
    )
    trainer.logger._log_graph = True

    print("Start Training")
    trainer.fit(classifier)

    trainer.test(classifier)
    # , ckpt_path='test_relazione2/2gu0e3j8/checkpoints/epoch=13-step=16603.ckpt'
    wandb.finish()
