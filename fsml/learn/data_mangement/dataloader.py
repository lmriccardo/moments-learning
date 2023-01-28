import torch
from torch.utils.data import DataLoader

from fsml.learn.data_mangement.dataset import FSMLDataset
from typing import Optional, List, Generic, Tuple, Iterator

from . import T


def collate(_batch: Generic[T]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    r"""
    A function that replace the default collate function of the DataLoader.
    This function takes as input a batch of any size and creates two lists.
    The first list will contain all the input element of the batch, while
    the second will contains all its output elements. Recall that each
    element of the batch is a tuple (input, output) in the way the
    :class:`FSMLDataset` has been setup.
    
    :param _batch: The current given as input by the dataloader
    :return: a couple of list such that the first element are the input and second the outptus
    """
    input_data, output_data = [], []
    for (idata, odata) in _batch:
        input_data.append(idata)
        output_data.append(odata)

    return input_data, output_data


class FSMLDataLoader(DataLoader):
    r""" Just a custom dataloader for our dataset """
    def __init__(self, dataset      : FSMLDataset, 
                       batch_size   : int  = 1,
                       shuffle      : bool = True,
                       follow_batch : Optional[List[str]] = None,
                       exclude_keys : Optional[List[str]] = None,
                       **kwargs
    ) -> None:
        """
        :param dataset     : the input dataset to iterate
        :param batch_size  : The size of each minibatch
        :param shuffle     : True if we want to shuffle the dataset
        :param follow_batch: (To be defined)
        :param exclude_keys: (To be defined)
        """
        # Let's remove a possible input collate_fn
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super(FSMLDataLoader, self).__init__(dataset, 
                                             batch_size, 
                                             shuffle,
                                             collate_fn=collate,
                                             **kwargs)

    def __call__(self) -> Iterator:
        """ Makes the dataloader class callable """
        return super().__iter__()