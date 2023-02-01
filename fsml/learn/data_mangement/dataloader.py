import torch
from torch.utils.data import DataLoader

from fsml.learn.data_mangement.dataset import FSMLMeanStdDataset, FSMLOneMeanStdDataset
from typing import Optional, List, Generic, Tuple, Iterator

from . import T


def collate(_batch: Generic[T]) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    A function that replace the default collate function of the DataLoader.
    This function takes as input a batch of any size and creates two tensor
    by just vstacking all the input datas and the output datas.

    :param _batch: The current given as input by the dataloader
    :return: (Input data, Output data)
    """
    input_data, output_data = None, None
    output_shapes = []
    for (idata, odata, output_shape) in _batch:
        input_data = idata if input_data is None else torch.vstack((input_data, idata))
        output_data = odata if output_data is None else torch.vstack((output_data, odata))
        output_shapes.append(output_shape)

    return input_data, output_data, output_shapes


class FSMLDataLoader(DataLoader):
    r""" Just a custom dataloader for our dataset """
    def __init__(self, dataset      : FSMLMeanStdDataset | FSMLOneMeanStdDataset,
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
        
        # Make sure that whenever a FSMLMeanStdDataset is
        # given then the mixup attribute is set to true
        if isinstance(dataset, FSMLMeanStdDataset):
            assert dataset.mixup, \
                "[!!!] ERROR: Calling dataloader for a FSMLMeanStdDataset" + \
                " with mixup attribute set to False. Please set it to True"

        super(FSMLDataLoader, self).__init__(dataset, 
                                             batch_size, 
                                             shuffle,
                                             collate_fn=collate,
                                             **kwargs)

    def __call__(self) -> Iterator:
        """ Makes the dataloader class callable """
        return super().__iter__()