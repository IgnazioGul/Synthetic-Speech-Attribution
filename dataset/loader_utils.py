import cv2
import numpy as np
import torch
from typing_extensions import Literal

from constants.model_enum import ModelEnum
from models.passt.preprocess import AugmentMelSTFT
from utils.audio_utils import AudioUtils


def preprocess_item(audio, sr: int, model_name: Literal["att_vgg16", "passt"],
                    partition: Literal["training", "validation", "test"], transform: bool = False,
                    should_return_waveform: bool = False, extract_manual_spec: bool = False):
    if transform:
        audio = AudioUtils.pad_trunc(audio, sr, 5000)
    spec = AudioUtils.get_mel_spect(audio, sr, 128, model_name)
    # spec_img = AudioUtils.spec_to_image(spec)
    spec_img = spec
    if model_name == ModelEnum.ATT_VGG16.value:
        spec_img = cv2.resize(spec_img, (224, 224))
    elif model_name == ModelEnum.PASST.value and extract_manual_spec:
        mel = AugmentMelSTFT(n_mels=128, sr=16000, win_length=800, hopsize=320, n_fft=1024, freqm=48,
                             timem=192,
                             htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
                             fmax_aug_range=2000,
                             training=True if partition == "training" else False
                             )
        passt_audio = torch.Tensor(audio).unsqueeze(0)
        passt_spec = mel(passt_audio)
        return passt_spec, spec
    if model_name == ModelEnum.PASST.value or should_return_waveform:
        return audio, spec
    else:
        return spec_img[np.newaxis, ...], spec
