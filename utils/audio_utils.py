import csv

import librosa
import librosa.display
import librosa.feature
import numpy as np
import torch
from matplotlib import pyplot as plt


class AudioUtils:

    @staticmethod
    def get_mel_spect(audio: np.ndarray, sr: int, n_mels: int = 128, model_name="attVgg16"):
        """
        Returns mel spectrogram in Db scale of input audio \n
        :param audio: np.ndarray audio
        :param sr: sampling rate
        :param n_mels: number of Mel bands to generate
        :param model_name:
        :return: spectrogram: numpy.ndarray
        """
        if model_name == "passt":
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, win_length=800, hop_length=320,
                                                      n_fft=1024, htk=False, fmax=sr / 2)
        else:
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmax=sr / 2)

        # convert the power spectrogram to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec_db

    @staticmethod
    def spec_to_image(spec, eps=1e-6):
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        # norm values between 0-255
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
        spec_scaled = spec_scaled.astype(np.float32)
        return spec_scaled

    @staticmethod
    def pad_trunc(sig: np.ndarray, sr: int, max_ms: int) -> np.ndarray:
        """
        Append pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds \n
        :param sig: np.ndarray signal
        :param sr: sample rate of the signal
        :param max_ms:
        :return: new_sig= np.ndarray padded signal
        """
        sig = torch.from_numpy(sig)

        num_rows = 1 if sig.dim() == 1 else sig.size()[1]
        sig_len = sig.size()[0]
        max_len = int(sr // 1000 * max_ms)

        if sig_len > max_len:
            # Truncate the signal to the given length
            sig = sig[:max_len] if sig.dim() == 1 else sig[:, :max_len]

        elif sig_len < max_len:
            # Length of padding to add at the beginning and end of the signal
            # pad_begin_len = random.randint(0, max_len - sig_len)
            # pad_end_len = max_len - sig_len - pad_begin_len
            pad_end_len = max_len - sig_len
            # Pad with 0s
            # pad_begin = torch.zeros(pad_begin_len) if num_rows == 1 else torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros(pad_end_len) if num_rows == 1 else torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((sig, pad_end), 0 if num_rows == 1 else 1)

        return sig.numpy()


if __name__ == '__main__':
    pass

    #
    # with open("C:/Users/IgnazioGulino/Desktop/my/IG/Thesis MASTER DEGREE/Datasets/TIMIT-TTS/all.csv", "r",
    #           newline='') as file:
    #     reader = csv.reader(file, delimiter=',')
    #     next(reader)
    #     for row in reader:
    #         wave_sample_rate, wave_audio = wav.read(row[0])
    #         print(wave_sample_rate)
    #         print(wave_audio)
    #         [y, sr] = librosa.load(row[0], sr=None)
    #         print(y, sr)
    #         break
    #         AudioUtils.get_mel_spect(
    #             row[0],
    #             16000,
    #             128)
