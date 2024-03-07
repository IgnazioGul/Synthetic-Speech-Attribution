import os

import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger

import wandb
from constants.dataset_enum import DatasetEnum
from synthetic_classifier import SyntheticClassifier, _EarlyStopping

if __name__ == '__main__':
    # ------- BEGIN CONFIGURATION -------
    isPretrained = True
    isValidationEnabled = True
    is_augment_enabled = False  # set to true to add augmented samples in the dataset used for training
    extract_manual_spec = False  # property used only during the attack stage, so ignore it
    isGpuEnabled = False

    epochs = 25
    lr = 0.001
    decay = 0.00008
    batch_size = 32
    optimizer = "Adam"

    metadata_file = "clean.csv"  # change to load different TIMIT data (clean, aug, dtw, dwt_aug, all)
    # model_name = "attVgg16"
    model_name = "passt"
    # dataset = DatasetEnum.ASV19.value
    dataset = DatasetEnum.TIMIT_TTS.value
    # ------- END CONFIGURATION -------

    n_classes_timi = 12
    n_classes = 7 if dataset == DatasetEnum.ASV19.value or dataset == DatasetEnum.ASV19_SILENCE.value else n_classes_timi
    os.environ["WANDB_MODE"] = "offline"  # comment this line to enable wandb cloud logging
    classifier = SyntheticClassifier(metadata_file=metadata_file, model_name=model_name, pretrained=isPretrained, lr=lr,
                                     decay=decay,
                                     batch_size=batch_size,
                                     optimizer=optimizer,
                                     is_gpu_enabled=isGpuEnabled,
                                     is_augment_enabled=is_augment_enabled,
                                     is_validation_enabled=isValidationEnabled,
                                     dataset=dataset,
                                     n_classes=n_classes,
                                     extract_manual_spec=extract_manual_spec)

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
