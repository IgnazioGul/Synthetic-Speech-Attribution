import random

import librosa
import librosa.display
import librosa.feature
import numpy as np
import torch


class AudioUtils:

    @staticmethod
    def get_mel_spect(audio_full_path: str, n_mels: int = 128):
        """
        Returns mel spectrogram in Db scale of input audio \n
        :param audio_full_path:
        :param n_mels:
        :return: spectrogram: numpy.ndarray, sr:int sampling rate
        """
        y: np.ndarray
        sr: int
        # scales y values in [-1, 1] range
        [y, sr] = librosa.load(audio_full_path, sr=None)
        d = librosa.get_duration(y=y, sr=sr)

        # Compute the Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        # convert the power spectrogram to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec_db, sr

    @staticmethod
    def pad_trunc(sig, sr, max_ms):
        """
        Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds \n
        :param sig: torch.Tensor signal
        :param sr: sample rate of the signal
        :param max_ms:
        :return: new_sig= torch.Tensor padded signal
        """
        num_rows = 1 if sig.dim() == 1 else sig.size()[1]
        sig_len = sig.size()[0]
        max_len = int(sr // 1000 * max_ms)

        if sig_len > max_len:
            # Truncate the signal to the given length
            sig = sig[:, :max_len]

        elif sig_len < max_len:
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros(pad_begin_len) if num_rows == 1 else torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros(pad_end_len) if num_rows == 1 else torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 0 if num_rows == 1 else 1)

        return sig


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
