import os

import cv2
import librosa
import numpy as np
from audiomentations import PitchShift, TimeStretch, AddGaussianNoise
from torch.utils.data import Dataset, DataLoader
from typing_extensions import Literal

from utils.audio_utils import AudioUtils
from utils.timi_tts_constants import AUDIO_KEY, CLASS_KEY


class LoadAsvSpoof19(Dataset):
    classes_names = ["-", "A01", "A02", "A03", "A04", "A05", "A06"]
    classes_labels = {val: idx for idx, val in enumerate(classes_names)}

    partitions_path = {
        "training": os.path.join("ASVspoof2019_LA_train", "flac"),
        "validation": os.path.join("ASVspoof2019_LA_dev", "flac"),
        "test": os.path.join("ASVspoof2019_LA_eval", "flac")
    }

    labels_folder = "ASVspoof2019_LA_cm_protocols"
    partitions_labels = {
        "training": os.path.join(labels_folder, "ASVspoof2019.LA.cm.train.trn.txt"),
        "validation": os.path.join(labels_folder, "ASVspoof2019.LA.cm.dev.trl.txt"),
        "test": os.path.join(labels_folder, "ASVspoof2019.LA.cm.eval.trl.txt")
    }

    def __init__(self, base_path: str, partition: Literal["training", "validation", "test"],
                 model_name: Literal["resnet18", "resnet34", "resnet50", "attVgg16", "passt2"],
                 is_validation_enabled: bool = True,
                 mode: Literal["normal", "reduced"] = "normal",
                 is_augment_enabled: bool = False,
                 transform: bool = False):

        self.base_path = base_path
        self.model_name = model_name
        self.is_augment_enabled = is_augment_enabled
        # self.mode = mode
        self.transform = transform
        self.clean_train_audios_df = None
        self.partition = partition

        self.audios = np.loadtxt(os.path.join(
            self.base_path, self.partitions_labels[self.partition]), dtype=str, delimiter=' ')

        np.random.seed(123)
        np.random.shuffle(self.audios)

    def __getitem__(self, index):
        """
        Metadata file syntax
        1) SPEAKER_ID: 		LA_****, a 4-digit speaker ID
        2) AUDIO_FILE_NAME: 	LA_****, name of the audio file
        3) -: 			This column is NOT used for LA.
        4) SYSTEM_ID: 		ID of the speech spoofing system (A01 - A19),  or, for bonafide speech SYSTEM-ID is left blank ('-')
        5) KEY: 		'bonafide' for genuine speech, or, 'spoof' for spoofing speech \n
        :param index:
        :return:
        """
        _, audio_filename, _, system_id, key = self.audios[index]
        [audio, sr] = librosa.load(
            os.path.join(self.base_path, self.partitions_path[self.partition], audio_filename + ".flac"),
            sr=None)
        spec_img, spec = self._preprocess_item(audio, sr)
        audio_class = self.classes_labels[system_id]
        return {AUDIO_KEY: spec_img, CLASS_KEY: audio_class, "original": spec}
    def __len__(self):
        return len(self.audios)

    def _preprocess_item(self, audio, sr):
        if self.transform:
            audio = AudioUtils.pad_trunc(audio, sr, 5000)
        spec = AudioUtils.get_mel_spect(audio, sr, 128)
        # spec_img = AudioUtils.spec_to_image(spec)
        spec_img = spec
        if self.model_name == "attVgg16":
            spec_img = cv2.resize(spec_img, (224, 224))

        if self.model_name == "passt2":
            return audio, spec
        else:
            return spec_img[np.newaxis, ...], spec
