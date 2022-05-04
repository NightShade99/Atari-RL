
import pickle 
import numpy as np
from torch.utils import data

__all__ = ['load_data', 'ExperienceDataset', 'MultiEpochsDataLoader', 'numpy_collate']


def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    states = data['states']
    actions = data['actions']
    del data 
    return states, actions 


class ExperienceDataset(data.Dataset):
    
    def __init__(self, states, actions):
        super().__init__()
        self.states = states 
        self.actions = actions 
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx] / 255.0, self.actions[idx]


class MultiEpochsDataLoader(data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
            
            
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)