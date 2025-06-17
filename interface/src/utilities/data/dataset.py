# Dataset
import csv
from xml.etree.ElementInclude import default_loader
import numpy as np
from torch.utils.data import Dataset
import os
from pathlib import Path
import torch
import torchaudio
import h5py
import random


def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, "r") as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row["mid"]] = row["index"]
            line_count += 1
    return index_lookup


class TextDataset(Dataset):
    def __init__(self, data, logfile):
        super().__init__()
        self.data = data
        self.logfile = logfile

    def __getitem__(self, index):
        data_dict = {}
        # construct dict
        data_dict['fname'] = f"infer_file_{index}"
        data_dict['fbank'] = np.zeros((1024, 64))
        data_dict['waveform'] = np.zeros((32000))
        data_dict['text'] = self.data[index]
        if index == 0:
            with open(os.path.join(self.logfile), 'w') as f:
                f.write(f"{data_dict['fname']}: {data_dict['text']}")
        else:
            with open(os.path.join(self.logfile), 'a') as f:
                f.write(f"\n{data_dict['fname']}: {data_dict['text']}")
        return data_dict

    def __len__(self):
        return len(self.data)


class BandDataset(Dataset):
    def __init__(self, data_folder, mode, config, dataset_iter=1000):
        super().__init__()

        self.mode = mode
        self.data = [list((band_dir / mode).glob('*.h5'))
                     for band_dir in Path(data_folder).iterdir() if band_dir.is_dir()]
        self.data = [self.data[0], self.data[3]]
        self.config = config
        self.dataset_iter = dataset_iter

        audio_cfg = config['preprocessing']['audio']
        stft_cfg = config['preprocessing']['stft']
        mel_cfg = config['preprocessing']['mel']

        self.sample_rate = audio_cfg['sampling_rate']
        self.target_length = mel_cfg['target_length']
        self.n_fft = stft_cfg['filter_length']
        self.hop_length = stft_cfg['hop_length']
        self.win_length = stft_cfg['win_length']
        self.n_mels = mel_cfg['n_mel_channels']
        self.f_min = mel_cfg['mel_fmin']
        self.f_max = mel_cfg['mel_fmax']
        self.n_samples = self.hop_length * \
            (self.target_length - 1) + self.n_fft

        self.fbank_transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                f_min=self.f_min,
                f_max=self.f_max,
                power=2.0,
                center=False,
            ),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80.0)
        )

    def __getitem__(self, index):
        band = self.data[index % len(self.data)]
        path = random.choice(band)
        data_dict = {'fname': path.name}

        # .h5 파일에서 waveform 읽기
        with h5py.File(path, "r") as f:
            waveform = f["wav"][:]  # (C, T) 또는 (T,)
        # 채널 평균 또는 첫 채널 사용
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        # 랜덤 샘플링: 길이가 짧으면 패딩, 길면 crop
        if len(waveform) < self.n_samples:
            pad_width = self.n_samples - len(waveform)
            waveform = np.pad(waveform, (0, pad_width))
        else:
            start = np.random.randint(
                0, len(waveform) - self.n_samples + 1)
            waveform = waveform[start:start + self.n_samples]

        # Tensor 변환 및 fbank 추출
        waveform_tensor = torch.tensor(
            waveform, dtype=torch.float32).unsqueeze(0)  # (1, L)
        fbank = self.fbank_transform(waveform_tensor)  # (1, mel, T)
        fbank = fbank.squeeze(0).T       # → (T, mel)

        # data_dict['waveform'] = waveform_tensor.squeeze(0)  # (L,)
        data_dict['fbank'] = fbank                          # (T, mel)
        data_dict['text'] = path.stem.split('_')[0]
        return data_dict

    def __len__(self):
        return max(self.dataset_iter * len(self.data), 4)
