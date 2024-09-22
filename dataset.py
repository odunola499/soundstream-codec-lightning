from torch.utils.data import IterableDataset,Dataset, DataLoader
import torch
import random
from datasets import load_dataset, Audio, concatenate_datasets
from omegaconf import OmegaConf
import numpy as np
import os

config = OmegaConf.load('config.yaml')

class Data(IterableDataset):
    def __init__(self,config = config, data = None):
        super(Data, self).__init__()
        self.config = config
        self.data = data.cast_column('audio', Audio(sampling_rate = 24000))
        self.split = False
        self.segment_size = config.segment_size
        self.audio_norm_scale = config.audio_norm_scale

    def process(self, array:np.ndarray):
        tensor = torch.FloatTensor(array)
        tensor = tensor * self.audio_norm_scale
        if tensor.size(1) >= self.segment_size:
            max_audio_start = tensor.size(1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            tensor = tensor[:, audio_start:audio_start+self.segment_size]
        else:
            tensor = torch.nn.functional.pad(tensor, (0, self.segment_size - tensor.size(1)), 'constant')

        return tensor

    def __iter__(self):
        for row in self.data:
            audio = row['audio']
            array = audio['array']
            tensor = self.process(array)
            yield tensor

def get_loaders(batch_size = 4):
    # urls = [
    #     'internal_dataset',
    #     'internal_dataset',
    #     'internal_dataset'
    # ]
    # datas = [load_dataset(i, cache_dir = 'data', streaming = True)['train'] for i in urls]
    # data = concatenate_datasets(datas)
    data = load_dataset('gpt-omni/VoiceAssistant-400K', cache_dir = 'data', streaming = True)['train'].rename_column('question_audio','audio')
    return DataLoader(data, batch_size=batch_size, num_workers = os.cpu_count())

