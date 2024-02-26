import os
import tkinter as tk

import librosa
import matplotlib
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from dataset.load_asv_19_dataset import LoadAsvSpoof19
from utils.audio_utils import AudioUtils
from utils.augmentations import introduce_entropy

sr = 16000


def _play_audio(audio_waveform):
    sd.play(audio_waveform, samplerate=sr)


def print_audios(audios, aug_audios, labels, aug_type):
    root = tk.Tk()
    root.title(f"Augmentation viewer: {aug_type}")
    root.configure(pady=10)
    frame1 = tk.Frame(root)
    frame1.pack(side="left")
    frame2 = tk.Frame(root)
    frame2.pack(side="right")
    label1 = tk.Label(frame1, text="Clean audios")
    label1.pack()
    label2 = tk.Label(frame2, text="Augmented audios")
    label2.pack()
    for i in range(len(audios)):
        label = tk.Label(frame1, text=f"Clean {labels[i]}", padx=10, pady=3)
        label.pack()
        play_button = tk.Button(frame1, text="Play", command=lambda file=audios[i]: _play_audio(file), pady=1)
        play_button.pack()

        label = tk.Label(frame2, text=f"Aug {labels[i]}", padx=10, pady=3)
        label.pack()
        play_button = tk.Button(frame2, text="Play", command=lambda file=aug_audios[i]: _play_audio(file), pady=1)
        play_button.pack()

    # Start the Tkinter main loop
    root.mainloop()


def print_specs(original_spec, aug_audio_waveform, sr):
    aug_audio_spec = AudioUtils.pad_trunc(aug_audio_waveform, sr, 5000)
    aug_spec = AudioUtils.get_mel_spect(aug_audio_spec, sr, 128)
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.2, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    img = librosa.display.specshow(original_spec, sr=sr, x_axis='time', y_axis='mel', hop_length=512, n_fft=2048,
                                   ax=ax1, fmax=sr / 2)
    ax1.set_title('Original spec')
    ax1.invert_yaxis()
    fig.colorbar(img, format='%+2.0f dB', ax=ax1)

    gs.update(wspace=0.13)
    ax2 = fig.add_subplot(gs[0, 2])
    img = librosa.display.specshow(aug_spec, sr=sr, x_axis='time', y_axis='mel', hop_length=512, n_fft=2048, ax=ax2,
                                   fmax=sr / 2)
    ax2.set_title('Aug spec')
    ax2.invert_yaxis()
    fig.colorbar(img, format='%+2.0f dB', ax=ax2)
    plt.show(block=True)


if __name__ == "__main__":
    load_dotenv()
    base_path = os.getenv("ASV-19-ROOT-DIR")
    partition = "training"
    full_path = os.path.join(base_path, LoadAsvSpoof19.partitions_labels[partition])

    data = np.loadtxt(full_path, dtype=str, delimiter=' ')
    np.random.seed(123)
    np.random.shuffle(data)
    audios = np.array([row[1] for row in data[0:10]])
    labels = np.array([row[3] for row in data[0:10]])
    clean_audios = []
    aug_audios = []

    for i, audio_filename in enumerate(audios):
        np.random.seed(123)
        full_audio_path = os.path.join(base_path, LoadAsvSpoof19.partitions_path[partition],
                                       str(audio_filename) + ".flac")
        [audio_waveform, sr] = librosa.load(full_audio_path, sr=None)
        clean_audios.append(audio_waveform)

        # aug_audio = apply_high_pass(audio_waveform, sr, 700)
        # aug_audio = apply_noise(audio_waveform, 0.01, 0.1)
        # aug_audio = apply_pitch_shift(audio_waveform, 0.1, 0.1, sr)
        # aug_audio = apply_time_stretch(audio_waveform, 0.8, 0.8, sr)
        # aug_audio = apply_partial_noise(audio_waveform=audio_waveform, ampl=0.01, start_s=0, duration_s=1, sr=sr)
        # aug_audio = apply_amplitude_low_pass(audio_waveform, sr, min_cutoff_db=-20, max_cutoff_db=0)
        # aug_audio = apply_band_stop(audio_waveform=audio_waveform, sr=sr, center_frequency=450, bandwidth=600, order=4)
        # aug_audio = test_sft(audio_waveform)
        # aug_audio = increment_variance(audio_waveform, (14, 1), 40)
        # aug_audio = increment_variance_2(aug_audio, (2, 1), 40, 0.01, mode="low")
        # aug_audio = apply_band_noise(aug_audio, 0.001, low_cutoff=30, high_cutoff=600, sr=sr)
        aug_audio = introduce_entropy(audio_waveform, sr)

        sf.write(f"{i}-CLEAN.wav", audio_waveform, sr)
        sf.write(f"{i}-AUG.wav", aug_audio, sr)
        aug_audios.append(aug_audio)
        if i == 0:
            print_specs(AudioUtils.get_mel_spect(audio_waveform, sr, 128), aug_audio, sr)

    print_audios(clean_audios, aug_audios, labels, "foo")
