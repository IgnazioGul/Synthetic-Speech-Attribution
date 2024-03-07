import os

import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing_extensions import Literal

from dataset.loader_utils import preprocess_item
from utils.timi_tts_constants import AUDIO_KEY, CLASS_KEY, ORIGINAL_SPEC_KEY


class LoadTimiDataset(Dataset):

    def __init__(self, base_path: str, metadata_file_path: str, partition: Literal["training", "validation", "test"],
                 model_name: Literal["resnet18", "resnet34", "resnet50", "att_vgg16"],
                 is_validation_enabled: bool = True,
                 transform: bool = False,
                 should_return_waveform: bool = False,
                 extract_manual_spec: bool = False):

        self.path = os.path.join(base_path, metadata_file_path)
        self.model_name = model_name
        self.extract_manual_spec = extract_manual_spec
        self.should_return_waveform = should_return_waveform
        self.partition = partition

        np.random.seed(123)
        data = pd.read_csv(self.path)
        data = data.sample(frac=1, random_state=42)

        if is_validation_enabled:
            split_indices = np.array([0.6, 0.8]) * len(data)
            # split in 60%, 20%, 20%
            left, middle, right = np.split(data, split_indices.astype(int))
            if partition == "training":
                self.audios_df = left
            elif partition == "validation":
                self.audios_df = middle
            elif partition == "test":
                self.audios_df = right
        else:
            percent_66 = int(0.66 * len(data))
            # split in 66%, 33%
            left, right = np.split(data, [percent_66])
            if partition == "training":
                self.audios_df = left
            elif partition == "test":
                self.audios_df = right

        self.transform = transform

    def __getitem__(self, index):
        audio_path, audio_class = self.audios_df.iloc[index]
        [audio, sr] = librosa.load(audio_path, sr=None)
        spec_img, spec = self.preprocess_item(audio, sr)

        return {AUDIO_KEY: spec_img, CLASS_KEY: audio_class, ORIGINAL_SPEC_KEY: spec}

    def preprocess_item(self, audio, sr: int):
        return preprocess_item(audio, sr, model_name=self.model_name,
                               should_return_waveform=self.should_return_waveform, transform=self.transform,
                               partition=self.partition, extract_manual_spec=self.extract_manual_spec)

    def __len__(self):
        return self.audios_df.index.size
