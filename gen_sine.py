import librosa
import numpy as np
import random

"""
Small dataset generation tool for mixtures of sine waves. Used for
development.
"""

SAMPLE_RATE = 22050

def write_wav(waveform, filename, sample_rate=SAMPLE_RATE):
    librosa.output.write_wav(filename, waveform, sample_rate)

def make_signal(sig_length, sig_freq, sample_rate=SAMPLE_RATE):
    """
    Generates numpy array of signal

    Args;
        sig_length: length of signal in seconds
        sig_freq: frequency of signal in Hz
        sample_rate: sample rate of audio in Hz

    Returns:
        numpy array of signal
    """
    x = np.linspace(0, sig_length, sample_rate * sig_length)
    y = np.sin(x * np.pi * sig_freq * 2)

    return y

def make_lc_signal(signals, sample_rate=SAMPLE_RATE):
    """
    Generates numpy array of signal with multiple frequencies
    separated in time.

    Args;
        signals: list of tuples of (frequency of signal in Hz,
        duration in seconds)
        sample_rate: sample rate of audio in Hz

    Returns:
        numpy array of signal
    """
    waves = []
    for freq, duration in signals:
        x = np.linspace(0, duration, sample_rate * duration)
        waves.append(np.sin(x * np.pi * freq * 2))
    return np.concatenate(waves)

def make_durations(n):
    """
    Makes random length durations of every note in the 4th octave.
    Concatenates them into random order.

    Args:
        n: number of examples to generate
    """

    freqs = [440.000, 493.883, 523.251, 587.330, 659.255, 698.456, 783.991]

    waves = []
    for i in range(n):
        dur = [(x, y) for x, y in zip(freqs, np.random.gamma(
            shape=2, scale=0.5, size=len(freqs)))]
        random.shuffle(dur)
        waves.append(make_lc_signal(dur))

    return waves

if __name__ == '__main__':
    n_examples = 100
    f_names = []
    for i, j in enumerate(make_durations(n_examples)):
        f_name = 'mixhz_{:04}'.format(i)
        f_names.append(f_name)
        write_wav(j, f_name + '.wav')

    with open('metadata.csv', 'w') as f:
        for f_name in f_names:
            f.write('{}|None|None\n'.format(f_name))
