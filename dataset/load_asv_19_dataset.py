import os

import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing_extensions import Literal

from constants.env_var_enum import EnvVarEnum
from constants.model_enum import ModelEnum
from dataset.loader_utils import preprocess_item
from utils.augmentations import apply_time_shift, asv19_attack_class_based
from utils.timi_tts_constants import AUDIO_KEY, CLASS_KEY, ORIGINAL_SPEC_KEY


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

    silenced_partitions_labels = {
        "training": "unsupported",
        "validation": os.path.join(labels_folder, "ASVspoof2019.LA.cm.dev.filtered.trl.txt"),
        "test": "unsupported"
    }

    def __init__(self, base_path: str, partition: Literal["training", "validation", "test"],
                 model_name: Literal["resnet18", "resnet34", "resnet50", "att_vgg16", "passt"],
                 is_augment_enabled: bool = False,
                 transform: bool = False,
                 should_return_waveform: bool = False,
                 extract_manual_spec: bool = False,
                 is_asv19_silence_version: bool = False):

        self.is_asv19_silence_version = is_asv19_silence_version
        self.extract_manual_spec = extract_manual_spec
        self.should_return_waveform = should_return_waveform
        self.base_path = base_path
        self.model_name = model_name
        self.is_augment_enabled = is_augment_enabled
        self.transform = transform
        self.clean_train_audios_df = None
        self.partition = partition
        partition_labels_path = self.silenced_partitions_labels[self.partition] if self.is_asv19_silence_version else \
            self.partitions_labels[self.partition]
        self.audios = np.loadtxt(os.path.join(
            self.base_path, partition_labels_path), dtype=str, delimiter=' ')
        # RUN TO GENERATE FILTERED LABEL FILE
        # if self.is_asv19_silence_version:
        #     self._filter_missing_audios()

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
        full_file_path = os.path.join(self.base_path, self.partitions_path[self.partition],
                                      audio_filename + ".wav" if self.is_asv19_silence_version else audio_filename + ".flac")
        [audio, sr] = librosa.load(full_file_path, sr=None)
        spec_img, spec = self.preprocess_item(audio, sr)
        audio_class = self.classes_labels[system_id]
        if self.is_augment_enabled and self.partition == "training":
            if np.random.rand() <= 0.7:
                aug_audio = asv19_attack_class_based(audio, audio_class, (2, 1), 40, 0.2, sr=sr)
            else:
                ts_sec = np.random.uniform(1, 3)
                aug_audio = apply_time_shift(audio, sr, ts_sec)
            aug_spec_img, aug_spec = self.preprocess_item(audio, sr)
            return [{AUDIO_KEY: aug_spec_img, CLASS_KEY: audio_class, ORIGINAL_SPEC_KEY: aug_spec},
                    {AUDIO_KEY: spec_img, CLASS_KEY: audio_class, ORIGINAL_SPEC_KEY: spec}]
        return {AUDIO_KEY: spec_img, CLASS_KEY: audio_class, ORIGINAL_SPEC_KEY: spec}

    def __len__(self):
        return len(self.audios)

    def _filter_missing_audios(self):
        filtered_audios = []
        for row in self.audios:
            _, audio_filename, _, system_id, key = row
            file_full_path = os.path.join(self.base_path, self.partitions_path[self.partition],
                                          audio_filename + ".wav" if self.is_asv19_silence_version else ".flac")
            if os.path.exists(file_full_path):
                filtered_audios.append(row)
        np.savetxt("LA_CM_FILTERED.txt", np.array(filtered_audios), delimiter=" ", fmt='%s')
        self.audios = np.array(filtered_audios)

    def preprocess_item(self, audio, sr: int):
        return preprocess_item(audio, sr, model_name=self.model_name,
                               should_return_waveform=self.should_return_waveform, transform=self.transform,
                               partition=self.partition, extract_manual_spec=self.extract_manual_spec)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # root dir should be .../ASV_ROOT/LA
    dt = LoadAsvSpoof19(base_path=os.getenv(EnvVarEnum.ASV19_ROOT_DIR.value),
                        partition="training",
                        transform=True, model_name=ModelEnum.PASST.value)

    dt = DataLoader(dt, batch_size=32, shuffle=False, num_workers=0,
                    pin_memory=False)

    iterator = iter(dt)
    for batch in dt:
        batch = next(iterator)

        labels = batch[CLASS_KEY]
        print("Labels: ", labels)
