import lightning.pytorch as pl
import torch
import torchmetrics
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader
from torchvision import models, utils
from typing_extensions import Literal

import wandb
from constants.dataset_enum import DatasetEnum
from constants.model_enum import ModelEnum
from dataset.load_asv_19_dataset import LoadAsvSpoof19
from dataset.load_timi_dataset import LoadTimiDataset
from models.attVgg16.attention_block import visualize_attention, print_original_spec
from models.attVgg16.attention_vgg16 import AttentionVgg16
from models.passt.base import get_basic_model, get_model_passt
from utils.dataset_utils import get_dataset_base_path, extract_aug_batch
from utils.timi_tts_constants import AUDIO_KEY, CLASS_KEY, LABELS_MAP, ORIGINAL_SPEC_KEY


class SyntheticClassifier(pl.LightningModule):
    global_step = 0

    def __init__(self, pretrained: bool, metadata_file: Literal[
        "clean.csv", "dtw.csv", "aug.csv", "aug_dtw.csv", "clean_reduced.csv", "dtw_reduced.csv", "aug_reduced.csv", "aug_dtw_reduced.csv"],
                 model_name: Literal["resnet18", "resnet34", "resnet50", "att_vgg16", "passt"],
                 n_classes: int,
                 is_validation_enabled: bool = True,
                 lr: float = 0.0005,
                 decay: float = 0.00002, momentum: float = 0.99, batch_size: int = 128,
                 optimizer: str = "SGD",
                 is_gpu_enabled: bool = False,
                 is_augment_enabled: bool = False,
                 freezed: bool = False,
                 dataset: Literal["timit_tts", "asv19", "asv19_silence"] = "timit_tts",
                 return_attentions: bool = True,
                 extract_manual_spec: bool = False,
                 ):
        super().__init__()
        self.extract_manual_spec = extract_manual_spec  # only used for passt
        self.return_attentions = return_attentions
        self.dataset = dataset
        self.dataset_base_path = get_dataset_base_path(self.dataset)
        self.metadata_file = metadata_file
        self.freezed = freezed
        self.is_augment_enabled = is_augment_enabled
        self.n_classes = n_classes
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
        # self.loss_module = torch.nn.CrossEntropyLoss() if self.n_classes > 2 else torch.nn.BCELoss()
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
        elif self.model_name == ModelEnum.ATT_VGG16.value:
            backbone = AttentionVgg16(num_classes=self.n_classes, normalize_attn=True, pretrained=self.pretrained,
                                      freezed=self.freezed, return_attentions=self.return_attentions)
        elif self.model_name == ModelEnum.PASST.value:
            backbone = get_basic_model(mode="logits", extract_manual_spec=self.extract_manual_spec, pretrained=False)
            backbone.net = get_model_passt(arch="passt_s_swa_p16_128_ap476", n_classes=self.n_classes,
                                           pretrained=self.pretrained)
        else:
            raise Exception("Unsupported model ", self.model)
        return backbone

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            optimizer = Adam(self.model.parameters(),
                             lr=self.lr, weight_decay=self.decay)
        elif self.optimizer == "AdamW":
            optimizer = AdamW(self.model.parameters(),
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
            if self.dataset == DatasetEnum.TIMIT_TTS.value:
                self.training_set = LoadTimiDataset(base_path=self.dataset_base_path,
                                                    metadata_file_path=self.metadata_file, partition="training",
                                                    model_name=self.model_name,
                                                    transform=True, is_validation_enabled=self.is_validation_enabled)
                if self.is_validation_enabled:
                    self.validation_set = LoadTimiDataset(base_path=self.dataset_base_path,
                                                          metadata_file_path=self.metadata_file,
                                                          partition="validation",
                                                          model_name=self.model_name,
                                                          transform=True)
            elif self.dataset == DatasetEnum.ASV19.value:
                self.training_set = LoadAsvSpoof19(base_path=self.dataset_base_path, partition="training",
                                                   model_name=self.model_name,
                                                   transform=True,
                                                   is_augment_enabled=self.is_augment_enabled,
                                                   extract_manual_spec=self.extract_manual_spec)
                if self.is_validation_enabled:
                    self.validation_set = LoadAsvSpoof19(base_path=self.dataset_base_path,
                                                         partition="validation",
                                                         model_name=self.model_name,
                                                         transform=True,
                                                         extract_manual_spec=self.extract_manual_spec)
            elif self.dataset == DatasetEnum.ASV19_SILENCE.value:
                self.training_set = LoadAsvSpoof19(base_path=self.dataset_base_path, partition="training",
                                                   model_name=self.model_name,
                                                   transform=True,
                                                   is_asv19_silence_version=True,
                                                   is_augment_enabled=self.is_augment_enabled)
                if self.is_validation_enabled:
                    self.validation_set = LoadAsvSpoof19(base_path=self.dataset_base_path,
                                                         partition="validation",
                                                         model_name=self.model_name,
                                                         transform=True,
                                                         is_asv19_silence_version=True)
        if stage == "test" or stage is None:
            if self.dataset == DatasetEnum.TIMIT_TTS.value:
                self.test_set = LoadTimiDataset(base_path=self.dataset_base_path,
                                                metadata_file_path=self.metadata_file, partition="test",
                                                model_name=self.model_name,
                                                transform=True)
            elif self.dataset == DatasetEnum.ASV19.value:
                self.test_set = LoadAsvSpoof19(base_path=self.dataset_base_path, partition="validation",
                                               model_name=self.model_name,
                                               transform=True)
            elif self.dataset == DatasetEnum.ASV19_SILENCE.value:
                self.test_set = LoadAsvSpoof19(base_path=self.dataset_base_path, partition="validation",
                                               model_name=self.model_name,
                                               transform=True,
                                               is_asv19_silence_version=True)

    def _get_preds_loss_accuracy(self, batch, batch_idx):
        audios = batch[AUDIO_KEY]
        labels = batch[CLASS_KEY]
        if self.model_name == ModelEnum.ATT_VGG16.value:
            logits, _, _ = self.model(audios)
        else:
            logits = self.model(audios)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss_module(logits, labels)
        acc = torchmetrics.functional.accuracy(preds, labels, "multiclass",
                                               num_classes=self.n_classes)
        f1 = torchmetrics.functional.f1_score(preds, labels, "multiclass",
                                              num_classes=self.n_classes)

        if (batch_idx == 0 or batch_idx == 1) and self.model_name == ModelEnum.ATT_VGG16.value:
            # if batch_idx == self.trainer.num_training_batches - 1 and self.model_name == ModelEnum.ATT_VGG16.value:
            _, attFilter1, attFilter2 = self.model(audios[0:8])

            I_train = utils.make_grid(audios[0:8], nrow=8, padding=2, normalize=True,
                                      scale_each=True)

            first = visualize_attention(I_train, attFilter1, up_factor=2, no_attention=False)
            second = visualize_attention(I_train, attFilter2, up_factor=4, no_attention=False)

            orig_spec = batch[ORIGINAL_SPEC_KEY][0:8]
            n_rows = 1
            fig = plt.figure(figsize=(30, 10), layout="constrained")
            spec = fig.add_gridspec(3, 8)

            ax2 = fig.add_subplot(spec[1, :])
            ax3 = fig.add_subplot(spec[2, :])

            print_original_spec(orig_spec, fig, spec, n_rows, self.current_epoch)

            ax2.imshow(first)
            ax2.title.set_text("pool-3 attention")

            ax3.imshow(second)
            ax3.title.set_text("pool-4 attention")

            plt.savefig("attentions_epoch" + str(self.current_epoch) + '_vgg16_attentions.png')
            plt.show()
            plt.draw()

        return preds, loss, acc, f1

    def _get_probs_preds_loss_accuracy(self, batch):
        audios = batch[AUDIO_KEY]
        labels = batch[CLASS_KEY]

        if self.model_name == ModelEnum.ATT_VGG16.value:
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
        if self.is_augment_enabled:
            train_batch = extract_aug_batch(train_batch)
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
        all_class_names = list(LABELS_MAP.keys())
        if self.dataset == DatasetEnum.TIMIT_TTS.value:
            cm = wandb.plot.confusion_matrix(
                y_true=y_true,
                probs=probs,
                class_names=all_class_names
            )
        else:
            cm = wandb.plot.confusion_matrix(
                y_true=y_true,
                probs=probs,
                class_names=LoadAsvSpoof19.classes_names
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
        return DataLoader(self.training_set, batch_size=self.batch_size,
                          # sampler=BalancedBatchSampler(self.training_set, torch.Tensor(all_items)),
                          shuffle=False, num_workers=1,
                          pin_memory=self.is_gpu_enabled)

    def val_dataloader(self):
        return DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=False, num_workers=1,
                          pin_memory=self.is_gpu_enabled)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=1,
                          pin_memory=self.is_gpu_enabled)


class _EarlyStopping(EarlyStopping, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
