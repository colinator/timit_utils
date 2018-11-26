import math
import numpy as np
import scipy as sp
import pandas as pd
from python_speech_features import mfcc, fbank, logfbank
from typing import *

def averaged_sequence(s: np.ndarray, R: float) -> np.ndarray:
    '''sampling, with averaging
       r2 = R // 2
       return [sum(s[i-r2:i+r2]) for i in range(0, len(s), R)]

       true sampling, no averaging
       return [s[i] for i in range(0, len(s), R)]

       sampling done with np - first pad, then reshape/sample, then unpad'''

    pad_size = math.ceil(float(s.size)/R)*R - s.size
    s_padded = np.append(s, np.zeros(pad_size)*np.NaN)
    sampled = sp.nanmean(s_padded.reshape(-1,R), axis=1)
    sampled_nopad = sampled[:-pad_size]
    retval = sampled_nopad.reshape((len(sampled_nopad),))
    return retval

def down_sampled_audio(data: np.ndarray, sample_rate: int, to_sample_rate: int) -> np.ndarray:
    return averaged_sequence(data, sample_rate // to_sample_rate)

def signal_0scaled_to(signal: np.ndarray, time_axis:int =0, scaled_to: float = 1.0) -> np.ndarray:
    new_shape = (1, signal.shape[1]) if time_axis == 0 else (signal.shape[0], 1)
    mins = signal.min(axis=time_axis).reshape(new_shape)
    maxs = signal.max(axis=time_axis).reshape(new_shape)
    return signal / (maxs - mins) * scaled_to



AudioFeatureOrig = 0
AudioFeatureSampled = 1
AudioFeatureAbs = 2
AudioFeatureMel = 3
AudioFeatureMelBank = 4
AudioFeatureMelBankEnergies = 5
AudioFeatureMelLogBank = 6

AudioFeaturesMelAll = [ AudioFeatureMelBankEnergies, AudioFeatureMelBank, AudioFeatureMelLogBank, AudioFeatureMel ]
AudioFeaturesMelBanks = [ AudioFeatureMelBankEnergies, AudioFeatureMelBank, AudioFeatureMelLogBank ]


window_length = 0.025
window_step = 0.001
nfft = 512
preemphasis = 0.97
scale_to = 10.0
nfilt = 26


def audio_features(data: np.ndarray, sample_rate: int, feature_types: List[int] = AudioFeaturesMelAll) -> np.ndarray:

    features = []
    mel_filterbank = None
    mel_energies = None
    for feature in feature_types:
        if feature == AudioFeatureOrig:
            new_data = data
        elif feature == AudioFeatureSampled:
            new_data = down_sampled_audio(data, sample_rate=sample_rate, to_sample_rate=math.ceil(1.0 / window_step))
        elif feature == AudioFeatureMel:
            new_data = mfcc(data, sample_rate, winlen=window_length, winstep=window_step,
                            nfilt=nfilt, nfft=nfft, lowfreq=0, highfreq=None, preemph=preemphasis)
        elif feature in AudioFeaturesMelBanks:
            if mel_filterbank is None:
                mel_filterbank, mel_en = fbank(data, samplerate=sample_rate, winlen=window_length, winstep=window_step, nfilt=nfilt, nfft=nfft, lowfreq=0, highfreq=None, preemph=preemphasis)
                mel_filterbank = mel_filterbank
                mel_energies = mel_en.reshape((len(mel_en),1))
            if feature == AudioFeatureMelBank:
                new_data = mel_filterbank
            elif feature == AudioFeatureMelBankEnergies:
                new_data = mel_energies
            elif feature == AudioFeatureMelLogBank:
                new_data = np.log(mel_filterbank + 1)
        features.append(new_data)

    return np.concatenate(features, axis=1)


def audio_zero_padded(start_0s: int, data: np.ndarray, end_0s: int) -> np.ndarray:
    return np.concatenate([np.zeros((start_0s,)), data, np.zeros((end_0s,))])

def audio_gained(data: np.ndarray, gain: float) -> np.ndarray:
    return data * gain

def features_replicated(data: np.ndarray, by: int) -> np.ndarray:
    return np.repeat(data, by, axis=1)

def padded_process(left_pad: float, process: np.ndarray) -> np.ndarray:
    return process + left_pad

def resampled_process(process: np.ndarray, from_sample_rate: float, left_pad: float=0) -> np.ndarray:
    return padded_process(left_pad, process) / (window_step * from_sample_rate) - window_length * 1000.0 / 2.0

def resampled_phones_df(phones_df: pd.DataFrame, from_sample_rate: float, left_pad: float=0) -> pd.DataFrame:
    new_starts = resampled_process(phones_df.start.values, from_sample_rate, left_pad=left_pad)
    new_ends = resampled_process(phones_df.end.values, from_sample_rate, left_pad=left_pad)
    return pd.DataFrame(np.array([new_starts, new_ends]).T, index=phones_df.index, columns=phones_df.columns)

def resampled_audio(audio_orig: np.ndarray, sample_rate: int, pad: int, to_sample_rate: int=1000) -> np.ndarray:
    # start_0s: int, data: np.ndarray, end_0s: int
    az = audio_zero_padded(pad, audio_orig, pad)
    v = down_sampled_audio(az, sample_rate=sample_rate, to_sample_rate=to_sample_rate)
    return v
