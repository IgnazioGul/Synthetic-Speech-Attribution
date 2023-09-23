import numpy as np
import sox


def _set_seed():
    np.random.seed(123)


def apply_pitch_shift(audio_waveform, pitch_shift_semitones, sr=16000):
    _set_seed()
    tfm = sox.Transformer()
    tfm.pitch(pitch_shift_semitones)
    return tfm.build_array(input_array=audio_waveform, sample_rate_in=sr)


def apply_time_stretch(audio_waveform, time_rate: float, sr=16000):
    _set_seed()
    tfm = sox.Transformer()
    tfm.speed(factor=time_rate)
    return tfm.build_array(input_array=audio_waveform, sample_rate_in=sr)


def apply_noise(audio_waveform, ampl):
    _set_seed()
    noise = ampl * np.random.randn(len(audio_waveform))
    return audio_waveform + noise
