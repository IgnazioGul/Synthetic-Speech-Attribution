import random

import librosa
import numpy as np
import sox
from scipy.signal import butter, lfilter
from typing_extensions import Literal


def _set_seed():
    np.random.seed(123)


def _get_spec(audio_waveform: np.ndarray):
    '''
    Returns spectrogram ndarray in magnitude and angle
    :param audio_waveform:
    :return: magnitude, angle
    '''
    n_fft = 1024
    hop_length = 320
    win_length = 800
    spectrogram = librosa.stft(audio_waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    angle = np.angle(spectrogram)
    magnitude = np.abs(spectrogram)

    return magnitude, angle


def _get_audio_from_spec(spec: np.ndarray, angle: np.float64):
    '''
    Converts spectrogram back to an audio waveform
    :param spec: spcetrogram ndarray
    :param angle: angle of the spectrogram
    :return: audio waveform as ndarray
    '''
    n_fft = 1024
    hop_length = 320
    win_length = 800
    return librosa.istft(spec * np.exp(1j * angle), hop_length=hop_length, n_fft=n_fft, win_length=win_length)


def apply_pitch_shift(audio_waveform: np.ndarray, pitch_shift_semitones: float, sr: int = 16000):
    '''
    Applies pitch shift filter of given semitones
    :param audio_waveform:
    :param pitch_shift_semitones:
    :param sr:
    :return:
    '''
    _set_seed()
    tfm = sox.Transformer()
    tfm.pitch(pitch_shift_semitones)
    return tfm.build_array(input_array=audio_waveform, sample_rate_in=sr)


def apply_time_stretch(audio_waveform: np.ndarray, time_rate: float, sr: int = 16000):
    '''
    Applies time shift of given rate
    :param audio_waveform:
    :param time_rate: ratio of the new speed to the old speed
    :param sr:
    :return:
    '''
    _set_seed()
    tfm = sox.Transformer()
    tfm.speed(factor=time_rate)
    return tfm.build_array(input_array=audio_waveform, sample_rate_in=sr)


def apply_noise(audio_waveform: np.ndarray, ampl: float):
    '''
    Add gaussian noise to the waveform
    :param audio_waveform:
    :param ampl: amplitude of the noise introduced
    :return: noisy audio waveform
    '''
    _set_seed()
    noise = ampl * np.random.randn(len(audio_waveform))
    return audio_waveform + noise


def apply_band_noise(audio_waveform: np.ndarray, ampl: float, low_cutoff: int = 1, high_cutoff: int = 512,
                     sr: int = 16000):
    '''
    Applies a noise filter within a given frequencies band
    :param audio_waveform:
    :param ampl:
    :param low_cutoff:
    :param high_cutoff:
    :param sr:
    :return: noisy audio waveform
    '''
    noise_length = len(audio_waveform)
    noise = ampl * np.random.randn(noise_length)
    # Design a bandpass Butterworth filter
    order = 4  # Filter order
    nyquist_frequency = 0.5 * sr
    low_cutoff /= nyquist_frequency
    high_cutoff /= nyquist_frequency
    b, a = butter(order, [low_cutoff, high_cutoff], btype='band')

    # Apply the filter to the noise signal
    filtered_noise = lfilter(b, a, noise)
    return audio_waveform + filtered_noise


def apply_partial_noise(audio_waveform: np.ndarray, ampl: float, start_s: float, duration_s: float, sr=16000):
    '''
    Applies a noise filter within a given time range of the audio
    :param audio_waveform:
    :param ampl:
    :param start_s:
    :param duration_s:
    :param sr:
    :return: noisy audio waveform
    '''
    _set_seed()
    samples_to_jump = int(start_s * sr)
    samples_to_noise = int(sr * duration_s)
    noise = ampl * np.random.normal(0, 0.1, samples_to_noise)
    noisy_audio = audio_waveform.copy()
    noisy_audio[samples_to_jump:samples_to_jump + samples_to_noise] += noise
    return noisy_audio


def apply_high_pass(audio_waveform: np.ndarray, sr: int, cutoff_frequency: int):
    '''
    Applies a high pass frequencies filter
    :param audio_waveform:
    :param sr:
    :param cutoff_frequency:
    :return: resulting filtered audio waveform
    '''
    return band_pass(audio_waveform, sr, cutoff_frequency, "high")


def apply_low_pass(audio_waveform: np.ndarray, sr: int, cutoff_frequency: int):
    '''
    Applies a low pass frequencies filter
    :param audio_waveform:
    :param sr:
    :param cutoff_frequency:
    :return: resulting filtered audio waveform
    '''
    return band_pass(audio_waveform, sr, cutoff_frequency, "low")


def apply_band_stop(audio_waveform: np.ndarray, sr: int, center_frequency: int, bandwidth: int, order=4):
    '''
    Applies a band stop frequencies filter
    :param audio_waveform:
    :param sr:
    :param center_frequency: center of the band stop frequency range
    :param bandwidth:
    :param order: order of the filter
    :return: resulting filtered audio waveform
    '''
    nyquist_frequency = 0.5 * sr
    low_cutoff = (center_frequency - 0.5 * bandwidth) / nyquist_frequency
    high_cutoff = (center_frequency + 0.5 * bandwidth) / nyquist_frequency
    b, a = butter(order, [low_cutoff, high_cutoff], btype='bandstop')

    # Apply the filter to the audio data
    filtered_audio = lfilter(b, a, audio_waveform)
    return filtered_audio


def band_pass(audio_waveform: np.ndarray, sr: int, cutoff_frequency: int, btype: str):
    '''
    Applies a generic band pass frequencies filter
    :param btype:
    :param audio_waveform:
    :param sr:
    :param cutoff_frequency:
    :param btype: the type of filter.  {'lowpass', 'highpass', 'bandpass', 'bandstop'}
    :return: resulting filtered audio waveform
    '''
    _set_seed()

    # Design a high-pass Butterworth filter
    order = 16  # Filter order
    nyquist_frequency = 0.5 * sr
    cutoff = cutoff_frequency / nyquist_frequency
    b, a = butter(order, cutoff, btype=btype)

    # Apply the filter to the audio data
    return lfilter(b, a, audio_waveform)


def temporal_ampl_low_pass(audio_waveform: np.ndarray, sr: int = 16000, low_pass_duration_s: float = 1,
                           threshold_dB: int = -40):
    '''
    Applies a low pass of the amplitudes (dB) within a given time range of the audio
    :param audio_waveform:
    :param sr:
    :param low_pass_duration_s:
    :param threshold_dB:
    :return: resulting filtered audio waveform
    '''
    magnitude, angle = _get_spec(audio_waveform)
    aug_magn = np.copy(magnitude)
    min_magn = np.min(magnitude)
    max_magn = np.max(magnitude)
    magn_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    start, end = _spot_harmonic_content_indexes(magn_db, -65)

    n_samples_per_bin = audio_waveform.shape[0] / magnitude.shape[1]
    bin_per_sec = sr / n_samples_per_bin

    duration = int(bin_per_sec * low_pass_duration_s)
    duration_end = start + duration + 1
    duration_end = magnitude.shape[1] if duration_end > magnitude.shape[1] else duration_end
    # preserve one max magnitude sample to preserve original dB range
    save_one_max_sample = True

    for i in range(0, magnitude.shape[0]):
        for j in range(start, duration_end):
            if magnitude[i, j] == max_magn and save_one_max_sample:
                save_one_max_sample = False
                continue
            if magn_db[i, j] >= threshold_dB:
                aug_magn[i, j] = min_magn
    np.random.seed(123)
    rand_freq_index = np.random.randint(0, aug_magn.shape[0])
    rand_time_index = np.random.randint(0, aug_magn.shape[1])
    aug_magn[rand_freq_index, rand_time_index] = max_magn
    return _get_audio_from_spec(aug_magn, angle)


def _spot_harmonic_content_indexes(spectrogram: np.ndarray, threshold_dB: int = -55):
    '''
    Helper to spot the beginning of the harmonic content within a spectrogram
    :param spectrogram:
    :param threshold_dB:
    :return: start_index,end_index of the samples that contains harmonic content
    '''
    # mean across time bins
    mean_power = np.mean(spectrogram, axis=0)

    above_threshold = mean_power >= threshold_dB

    # Find the start and end indices
    start_index = np.argmax(above_threshold)
    end_index = len(above_threshold) - np.argmax(above_threshold[::-1]) - 1

    return start_index, end_index


def apply_time_shift(audio_waveform: np.ndarray, sr: int, time_shift_s: float):
    '''
    Applies a time shift to the audio waveform of given seconds
    :param audio_waveform:
    :param sr:
    :param time_shift_s:
    :return: shifted audio
    '''
    samples_to_shift = int(sr * time_shift_s)
    shifted_audio = audio_waveform[:len(audio_waveform) - samples_to_shift]
    return np.pad(shifted_audio, (samples_to_shift, 0), 'constant')


def apply_amplitude_low_pass(audio_waveform: np.ndarray, sr, min_cutoff_db: int, max_cutoff_db: int):
    '''
    Applies a low pass of the amplitudes (dB) within a given range, in the entire duration of the audio
    :param audio_waveform:
    :param sr:
    :param min_cutoff_db:
    :param max_cutoff_db:
    :return: resulting filtered audio waveform
    '''
    magnitude, angle = _get_spec(audio_waveform)

    magn_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    min_magn = np.min(magnitude)
    max_magn = np.max(magnitude)
    n_row, n_col = magnitude.shape
    save_one_max_sample = True
    for i in range(n_row):
        for j in range(n_col):
            if magnitude[i, j] == max_magn and save_one_max_sample:
                save_one_max_sample = False
                continue
            if min_cutoff_db <= magn_db[i, j] <= max_cutoff_db:
                magnitude[i, j] = min_magn

    return _get_audio_from_spec(magnitude, angle)


def get_magnitude_for_target_db(ampl_max: float, target_db: int = -10):
    '''
    Computes the magnitude of given dB for a given audio waveform
    :param ampl_max: max magnitude of the waveform
    :param target_db:
    :return: target magnitude
    '''
    if target_db <= -80:
        target_db = -80
    return ampl_max * 10 ** (target_db / 20)


def increment_variance(audio_waveform: np.ndarray,
                       region_size: tuple[int, int] = (3, 3),
                       db_alteration: int = -15,
                       db_threshold: int = 20,
                       mode: Literal["low", "high"] = "low"):
    '''
    Implements the increment_variance attack described in the thesis
    :param audio_waveform:
    :param region_size: window size of the sliding window
    :param db_alteration: db alteration of the selected windows
    :param db_threshold: threshold to select windows
    :param mode: high to select windows above the threshold, low otherwise
    :return: resulting filtered audio waveform
    '''
    n_fft = 512
    hop_length = 256
    original_spectrogram = librosa.stft(audio_waveform, n_fft=n_fft,
                                        hop_length=hop_length)
    angle = np.angle(original_spectrogram)
    original_magnitude = np.abs(original_spectrogram)
    orig_magnitude_max = np.max(original_magnitude)
    aug_magnitude = np.copy(original_magnitude)
    original_db = librosa.amplitude_to_db(original_magnitude,
                                          ref=np.max)
    # preserve one max value sample to preserve the magnitude range
    preserve_max = True
    # Iterate over the spectrogram in regions
    for i in range(0,
                   original_magnitude.shape[0] - region_size[0] + 1,
                   region_size[0]):
        for j in range(0,
                       original_magnitude.shape[1] - region_size[1] + 1,
                       region_size[1]):
            region = original_magnitude[i:i + region_size[0],
                     j:j + region_size[1]]
            region_db = original_db[i:i + region_size[0],
                        j:j + region_size[1]]
            # Check if the region has some harmonic content
            if np.all(region_db <= -70):
                continue

            region_db_max = np.max(region_db)
            region_db_min = np.min(region_db)

            if (mode == "low" and np.abs(region_db_max - region_db_min) <= db_threshold) or (
                    mode == "high" and np.abs(region_db_max - region_db_min) >= db_threshold):
                for k in range(region.shape[0]):
                    for l in range(region.shape[1]):
                        if region[k, l] == orig_magnitude_max and preserve_max:
                            preserve_max = False
                            continue
                        new_db_ampl = region_db[k, l] - db_alteration
                        new_ampl = get_magnitude_for_target_db(orig_magnitude_max,
                                                               new_db_ampl)
                        region[k, l] = new_ampl

                region = np.clip(region, 0, orig_magnitude_max)
                aug_magnitude[i:i + region_size[0],
                j:j + region_size[1]] = region

    return librosa.istft(aug_magnitude * np.exp(1j * angle),
                         hop_length=hop_length,
                         n_fft=n_fft)


def increment_variance_2(audio_waveform, region_size: tuple[int, int] = (3, 3), db_alteration: float = -15,
                         variance_threshold: float = 0.05, mode: Literal["low", "high"] = "low", spec_fraction_end=1):
    '''
    Implements the increment_variance_v2 attack described in the thesis
    :param audio_waveform:
    :param region_size: window size of the sliding window
    :param db_alteration: db alteration of the selected windows
    :param variance_threshold: threshold to select windows
    :param mode: high to select windows above the threshold, low otherwise
    :param spec_fraction_end: fraction of the spectrogram to attack. Use 1 for entire spectrogram
    :return: resulting filtered audio waveform
    '''
    n_fft = 512
    hop_length = 256
    original_spectrogram = librosa.stft(audio_waveform, n_fft=n_fft, hop_length=hop_length)
    angle = np.angle(original_spectrogram)

    original_magnitude = np.abs(original_spectrogram)
    orig_magnitude_max = np.max(original_magnitude)
    aug_magnitude = np.copy(original_magnitude)
    original_db = librosa.amplitude_to_db(original_magnitude, ref=np.max)
    preserve_max = True
    max_row_index = int(original_magnitude.shape[0] * spec_fraction_end)
    # Iterate over the spectrogram in regions
    for i in range(0, original_magnitude.shape[0] - region_size[0] + 1, region_size[0]):
        for j in range(0, original_magnitude.shape[1] - region_size[1] + 1, region_size[1]):
            region = original_magnitude[i:i + region_size[0], j:j + region_size[1]]
            region_db = original_db[i:i + region_size[0], j:j + region_size[1]]
            # Check if the region has low variance
            if i >= max_row_index:
                break
            if np.all(region_db <= -70):
                continue

            if (mode == "low" and np.var(region) <= variance_threshold) | (
                    mode == "high" and np.var(region) >= variance_threshold):
                half_db_alteration = db_alteration / 2.
                for k in range(region.shape[0]):
                    for l in range(region.shape[1]):
                        if region[k, l] == orig_magnitude_max and preserve_max:
                            preserve_max = False
                            continue

                        new_ampl = get_magnitude_for_target_db(orig_magnitude_max,
                                                               region_db[k, l] - half_db_alteration * 2.)
                        region[k, l] = new_ampl if abs(new_ampl - region[k, l]) >= 0.0001 else region[k, l]

                region = np.clip(region, 0, orig_magnitude_max)
                aug_magnitude[i:i + region_size[0], j:j + region_size[1]] = region

    return librosa.istft(aug_magnitude * np.exp(1j * angle), hop_length=hop_length, n_fft=n_fft)


def introduce_entropy(audio_waveform: np.ndarray, sr: int = 16000):
    '''
   Implements the introduce_entropy attack described in the thesis
   :param audio_waveform:
   :param sr:
   :return: resulting filtered audio waveform
   '''
    aug_audio_waveform = apply_high_pass(audio_waveform, sr, 700)
    aug_audio_waveform = increment_variance_2(aug_audio_waveform, (1, 4), 40, 0.01, mode="low")
    aug_audio_waveform = apply_band_noise(aug_audio_waveform, 0.001, low_cutoff=30, high_cutoff=600, sr=sr)
    return aug_audio_waveform


def test_sft(audio_waveform: np.ndarray):
    '''
    Tests the conversion from audio to spectrogram ad back to audio, to ensure no quality is lost
    :param audio_waveform:
    :return:
    '''
    magnitude, angle = _get_spec(audio_waveform)

    return _get_audio_from_spec(magnitude, angle)
