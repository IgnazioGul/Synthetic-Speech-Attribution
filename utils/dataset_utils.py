import os

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from typing_extensions import Literal

from constants.dataset_enum import DatasetEnum
from constants.env_var_enum import EnvVarEnum
from constants.model_enum import ModelEnum
from utils.timi_tts_constants import AUDIO_KEY, CLASS_KEY, ORIGINAL_SPEC_KEY

load_dotenv()
asv19_checkpoints = {
    ModelEnum.PASST.value: os.getenv(EnvVarEnum.PASST_ASV19_CKP.value),
    ModelEnum.ATT_VGG16.value: os.getenv(EnvVarEnum.ATT_VGG16_ASV19_CKP.value)
}

asv19_silence_checkpoints = {
    ModelEnum.PASST.value: os.getenv(EnvVarEnum.PASST_ASV19_SILENCE_CKP.value),
    ModelEnum.ATT_VGG16.value: os.getenv(EnvVarEnum.ATT_VGG16_ASV19_SILENCE_CKP_DIR.value)
}


def get_dataset_base_path(dataset: Literal["asv19", "asv19_silence", "timit_tts"]):
    """
    Returns base path of given dataset from .env variables
    :param dataset:
    :return: base path string
    """
    if dataset == DatasetEnum.ASV19.value:
        return os.getenv(EnvVarEnum.ASV19_ROOT_DIR.value)
    elif dataset == DatasetEnum.ASV19_SILENCE.value:
        return os.getenv(EnvVarEnum.ASV19_SILENCE_ROOT_DIR.value)
    elif dataset == DatasetEnum.TIMIT_TTS.value:
        return os.getenv(EnvVarEnum.TIMIT_TTS_ROOT_DIR.value)


def get_checkpoint_path(dataset: Literal["asv19", "asv19_silence"], model_name: Literal["passt", "att_vgg16"]):
    """
    Returns checkpoint path of given dataset and model, from .env variables
    :param dataset:
    :param model_name:
    :return: base path string
    """
    if dataset == DatasetEnum.ASV19.value:
        return asv19_checkpoints[model_name]
    elif dataset == DatasetEnum.ASV19_SILENCE.value:
        return asv19_silence_checkpoints[model_name]


def map_passt_spec_printable(original_spec_array):
    original_min = -1.402585
    original_max = 0.9

    target_min = -80
    target_max = -0.5712745

    return (original_spec_array - original_min) / (original_max - original_min) * (
            target_max - target_min) + target_min


def load_specific_classes(dataloader: DataLoader, target_class: int, n: int):
    """
    Filters dataloader for given target class and return the first n samples
    :param dataloader:
    :param target_class:
    :param n:
    :return: dictionary with n samples and same format as dataloader
    """
    data_audios = []
    data_classes = []
    data_specs = []
    for batch in dataloader:
        audios = batch[AUDIO_KEY]
        classes = batch[CLASS_KEY]
        specs = batch[ORIGINAL_SPEC_KEY]

        els = [value for index, value in enumerate(audios) if classes[index] == target_class]
        els_specs = [value for index, value in enumerate(specs) if classes[index] == target_class]
        if len(data_audios) + len(els) >= n:
            overflow = len(data_audios) + len(els) - n
            data_audios.extend(els[:len(els) - overflow])
            data_specs.extend(els_specs[:len(els) - overflow])
            data_classes.extend(target_class for i in range(len(els) - overflow))
            break
        else:
            data_audios.extend(els)
            data_specs.extend(els_specs)
            data_classes.extend(target_class for i in range(len(els)))
    data_audios = torch.stack(data_audios)
    data_specs = torch.stack(data_specs)
    data_classes = torch.tensor(data_classes)
    return {AUDIO_KEY: data_audios, CLASS_KEY: data_classes, ORIGINAL_SPEC_KEY: data_specs}


def extract_dict_vals(i: int, dataset: Dataset, dataset_waveform: Dataset, is_target_attack: bool):
    """
    Extract dictionary values for two different formats, based on is_target_attack value
    :param i: index of the sample in the dataset
    :param dataset:
    :param dataset_waveform:
    :param is_target_attack:
    :return: audio, spec, audio_waveform, audio_class
    """
    if is_target_attack:
        audio = dataset[AUDIO_KEY][i]
        spec = dataset[ORIGINAL_SPEC_KEY][i].numpy().astype(np.float32)
        audio_waveform = dataset_waveform[AUDIO_KEY][i]
        audio_class = dataset[CLASS_KEY][i]
    else:
        sample = dataset[i]  # Get a sample from the dataset
        audio = torch.tensor(sample[AUDIO_KEY])
        spec = sample[ORIGINAL_SPEC_KEY]
        audio_waveform = dataset_waveform[i][AUDIO_KEY]
        audio_class = sample[CLASS_KEY]
    return audio, spec, audio_waveform, audio_class


def extract_aug_batch(batch_aug):
    """
    Concatenates two augmented batches into one
    """
    batch_aug[0][AUDIO_KEY] = torch.cat((batch_aug[0][AUDIO_KEY], batch_aug[1][AUDIO_KEY]), dim=0)
    batch_aug[0][CLASS_KEY] = torch.cat((batch_aug[0][CLASS_KEY], batch_aug[1][CLASS_KEY]), dim=0)
    batch_aug[0]["original"] = torch.cat((batch_aug[0]["original"], batch_aug[1]["original"]), dim=0)
    batch_aug = batch_aug[0]
    return batch_aug
