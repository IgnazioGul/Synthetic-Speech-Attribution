import os

import cv2
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing_extensions import Literal

from utils.audio_utils import AudioUtils
from utils.timi_tts_constants import AUDIO_KEY, CLASS_KEY


class LoadTimiDataset(Dataset):

    def __init__(self, base_path: str, metadata_file_path: str, partition: Literal["training", "validation", "test"],
                 model_name: Literal["resnet18", "resnet34", "resnet50", "attVgg16"],
                 mode: Literal["normal", "reduced"] = "normal", transform: bool = False):
        self.path = os.path.join(base_path, metadata_file_path)
        self.model_name = model_name
        data = pd.read_csv(self.path)
        np.random.seed(123)
        data = data.sample(frac=1, random_state=42)

        split_indices = np.array([0.6, 0.8]) * len(data)
        # split in 60%, 20%, 20%
        left, middle, right = np.split(data, split_indices.astype(int))
        if partition == "training":
            # take 60% dataframe
            self.audios_df = left
        elif partition == "validation":
            # take middle 20% of the dataframe
            self.audios_df = middle
        elif partition == "test":
            # take end 20% of the  dataframe
            self.audios_df = right

        if mode == "reduced":
            self.audios_df = self.audios_df.loc[(self.audios_df[CLASS_KEY] == 2) | (self.audios_df[CLASS_KEY] == 3)]

        self.transform = transform

    def __getitem__(self, index):
        audio_path, audio_class = self.audios_df.iloc[index]
        [audio, sr] = librosa.load(audio_path, sr=None)
        if self.transform:
            audio = AudioUtils.pad_trunc(audio, sr, 5000)
        spec = AudioUtils.get_mel_spect(audio, sr, 128)
        spec_img = AudioUtils.spec_to_image(spec)
        if self.model_name == "attVgg16":
            spec_img = cv2.resize(spec_img, (224, 224))
        return {AUDIO_KEY: spec_img[np.newaxis, ...], CLASS_KEY: audio_class}

    def __len__(self):
        return self.audios_df.index.size


if __name__ == "__main__":
    data = pd.read_csv(r"C:\Users\IgnazioGulino\Desktop\my\IG\Thesis MASTER DEGREE\Datasets\TIMIT-TTS\clean.csv")
    # x0, y0 = x[0]
    p, p1 = data.iloc[0]

    LoadTimiDataset(base_path=r"C:/Users/IgnazioGulino/Desktop/my/IG/Thesis MASTER DEGREE/Datasets/TIMIT-TTS",
                    metadata_file_path="clean.csv", partition="training",
                    transform=True, mode="reduced")
    # split_indices = np.array([0.6, 0.8]) * len(data)
    # # split in 60%, 20%, 20%
    # left, middle, right = np.split(data, split_indices.astype(int))
    # out = left[(left[CLASS_KEY] == 3) | (left[CLASS_KEY] == 4)]
    # print(out, out.shape)
    # len_data = len(x)
    # split_indices = np.array([0.6, 0.8]) * len_data
    # left, middle, right = np.split(x, split_indices.astype(int))
    # take 60% dataframe

    # print(left.shape, middle.shape, right.shape, middle)
