import os
import re
import torch
import numpy as np
import soundfile as sf
import librosa
from torch.utils.data import Dataset

# MERGED MAPPING: Calm and Neutral are now both 0.
# All other classes shifted down by 1 to maintain a 0-6 range (7 classes total).
EMOTION_MAP = {
    "neutral": 0, "01": 0,
    "calm": 0,    "02": 0, 
    "happy": 1,   "03": 1,
    "sad": 2,     "04": 2,
    "angry": 3,   "05": 3,
    "fear": 4,    "06": 4, "fearful": 4,
    "disgust": 5, "07": 5,
    "surprise": 6,"08": 6, "surprised": 6, "pleasant_surprise": 6, "pleasant_surprised": 6,

    "neu": 0, "hap": 1, "sad": 2, "ang": 3, "fea": 4, "dis": 5,
    "n": 0, "h": 1, "sa": 2, "a": 3, "f": 4, "d": 5, "su": 6
}

DOMAIN_MAP = {
    "ravdess": 0,
    "crema": 1, 
    "tess": 2,
    "savee": 3
}

class UnifiedWaveLMDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, duration=3.0):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * duration)
        
        self.file_paths = []
        self.emotion_labels = []
        self.domain_labels = []
        
        self._scan_files()
        print(f"Dataset initialized. Total valid samples: {len(self.file_paths)}")

    def _determine_domain(self, path):
        path_lower = path.lower()
        for domain_name, domain_idx in DOMAIN_MAP.items():
            if domain_name in path_lower:
                return domain_idx
        return -1

    def _scan_files(self):
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if not file.lower().endswith(".wav"):
                    continue
                
                full_path = os.path.join(root, file)
                domain_label = self._determine_domain(full_path)
                
                if domain_label == -1:
                    continue 
                
                emotion_label = None
                file_name_no_ext = os.path.splitext(file)[0]
                file_lower = file_name_no_ext.lower()

                try:
                    if domain_label == 0:
                        parts = file_name_no_ext.split("-")
                        if len(parts) >= 3 and parts[2] in EMOTION_MAP:
                            emotion_label = EMOTION_MAP[parts[2]]
                    elif domain_label == 1:
                        parts = file_name_no_ext.split("_")
                        if len(parts) >= 3:
                            emo_code = parts[2].lower()
                            if emo_code in EMOTION_MAP:
                                emotion_label = EMOTION_MAP[emo_code]
                    elif domain_label == 2:
                        for emo_key in EMOTION_MAP:
                            if emo_key in file_lower:
                                emotion_label = EMOTION_MAP[emo_key]
                                break
                    elif domain_label == 3:
                        parts = file_name_no_ext.split("_")
                        if len(parts) >= 2:
                            emo_part = parts[1].lower() 
                            emo_code = re.sub(r'[0-9]+', '', emo_part) 
                            if emo_code in EMOTION_MAP:
                                emotion_label = EMOTION_MAP[emo_code]
                except Exception:
                    continue 

                if emotion_label is not None:
                    self.file_paths.append(full_path)
                    self.emotion_labels.append(emotion_label)
                    self.domain_labels.append(domain_label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio, sr = sf.read(self.file_paths[idx])

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        if sr != self.sample_rate:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=self.sample_rate)

        if len(audio) > self.max_length:
            audio = audio[:self.max_length] 
        else:
            pad_len = self.max_length - len(audio)
            audio = np.pad(audio, (0, pad_len), mode='constant')

        waveform = torch.tensor(audio, dtype=torch.float32)
        emo_label = torch.tensor(self.emotion_labels[idx], dtype=torch.long)
        dom_label = torch.tensor(self.domain_labels[idx], dtype=torch.long)

        return waveform, emo_label, dom_label