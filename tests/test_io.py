import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from frequency_analysis import read_text, read_wav


def test_read_headered_three_channel_file(tmp_path):
    fs = 2000
    n = 100
    t = np.arange(n)/fs
    f = tmp_path/'recording.tsv'
    pd.DataFrame({'t': t, 'x': np.sin(2*np.pi*50*t),
                  'y': 2*t, 'z': 3*t}).to_csv(f, sep='\t', index=False)
    r = read_text(f, time_column='t', channels=['x','z'], unit='m/s²')
    assert r.samples.shape == (100,2)
    assert r.sampling_rate_hz == pytest.approx(fs)
    assert r.channel_names == ('x','z')
    assert r.unit == 'm/s²'


def test_text_without_time_requires_rate(tmp_path):
    f = tmp_path/'values.csv'
    pd.DataFrame({'x': range(10)}).to_csv(f, index=False)
    with pytest.raises(ValueError):
        read_text(f)
    assert read_text(f, sampling_rate_hz=500).time_s[3] == .006


def test_reject_nonuniform_time(tmp_path):
    f = tmp_path/'irregular.csv'
    pd.DataFrame({'t':[0,.1,.2,.32,.4], 'x':[1,2,3,4,5]}).to_csv(f, index=False)
    with pytest.raises(ValueError, match='irregular'):
        read_text(f, time_column='t')


def test_normalized_pcm_wav(tmp_path):
    f = tmp_path/'mic.wav'
    wavfile.write(f, 4000, np.array([0, 8192, -8192, 0, 32767], dtype=np.int16))
    r = read_wav(f)
    assert r.samples.shape == (5,1)
    assert r.samples[1,0] == pytest.approx(.25)
    assert r.sampling_rate_hz == 4000
