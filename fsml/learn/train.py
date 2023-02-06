import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import _Loss, _WeightedLoss

import fsml.utils as utils
from fsml.learn.data_mangement.dataset import FSMLOneMeanStdDataset,  \
                                              get_dataset_by_indices, \
                                              FSMLMeanStdDataset

from fsml.learn.data_mangement.dataloader import FSMLDataLoader
from sklearn.model_selection import KFold
import fsml.learn.models.mlp as mlp
import fsml.learn.models.rbf as rbf
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Optional
from tqdm import tqdm
from functools import wraps
import os
import os.path as opath
import time
import config
    

class KFoldCrossValidationWrapper:
    r""" A simple wrapper class for K-Fold Cross Validation """

    @staticmethod
    def setup_kFold_validation(dataset    : FSMLOneMeanStdDataset, 
                               kf_split   : int, 
                               batch_size : int) -> List[Tuple[int, FSMLDataLoader]]:
        r"""
        Setup the kfold validation, i.e., returns a list of
        triple (fold index, train dataloader, validation dataloader)

        :param dataset: The dataset to split
        :param kf_split: The total number of split
        :param batch_size: the batch_size argument to the dataloader
        :return: a list of triple (fold index, train dataloader, validation dataloader)
        """
        # Create the splitter and the dataloader list
        kfold_splitter = KFold(n_splits=kf_split, shuffle=True)
        tt_list = []

        for fold_num, (train_ids, test_ids) in enumerate(kfold_splitter.split(dataset)):
            fold_dataset = get_dataset_by_indices(dataset, train_ids, test_ids)
            fold_dataloader = FSMLDataLoader(fold_dataset, batch_size=batch_size)
            tt_list.append((fold_num, fold_dataloader))

        return tt_list
    
    @staticmethod
    def kFoldValidation(dataset    : FSMLOneMeanStdDataset,
                        model      : mlp.FSML_MLP_Predictor,
                        epoch      : int,
                        kf_split   : int,
                        batch_size : int,
                        use        : bool=True) -> Callable:
        r""" Run kFold Cross Validation """
        def _kFoldValidation(func):
            
            @wraps(func)
            def wrapper(*args, **kwargs) -> Tuple[float, float]:
                """ The wrapper that returns the train and test total loss """
                # If use is set to False then just run the train set and returns
                if not use:
                    train_loss, test_acc = func()
                    return train_loss, test_acc
                
                # Set up the KFold validation and returns the dataloaders
                tt_list = KFoldCrossValidationWrapper.setup_kFold_validation(
                    dataset, kf_split, batch_size
                )

                progress_bar = tqdm(tt_list, desc=f"Epoch: {epoch} -- ", leave=True, position=0)
                fold_total_train_loss = 0.0
                fold_total_train_accuracy = 0.0
                for fold_num, fold_dataloader in progress_bar:
                    
                    # Create a new progress bar for the train step
                    dl_progress_bar = tqdm(fold_dataloader(), desc=f"Fold: {fold_num} -- ", leave=False, position=1)

                    # Get the final train loss
                    fold_train_loss, _ = func(fold_dataloader, dl_progress_bar)
                    fold_total_train_loss += fold_train_loss

                    # Set the dataset for testing
                    fold_dataloader.dataset.test()

                    # Validate
                    total_accuracy = 0.0
                    with torch.no_grad():
                        model.eval()  # Set the model for evaluation
                        for i_batch, (test_x, test_y, _) in enumerate(fold_dataloader):
                            current_accuracy = utils.compute_accuracy(model(test_x), test_y)
                            total_accuracy += current_accuracy
                        
                    total_accuracy /= (i_batch + 1)
                    progress_bar.set_postfix_str(f"Train Loss: {fold_train_loss}, Test Acc: {total_accuracy}")
                    progress_bar.refresh()

                    fold_total_train_accuracy += total_accuracy

                fold_total_train_loss /= (fold_num + 1)
                fold_total_train_accuracy /= (fold_num + 1)

                progress_bar.set_description_str(f"Epoch: {epoch} -- ")
                progress_bar.set_postfix_str(
                    f"Train Loss: {fold_total_train_loss}, Train Acc: {fold_total_train_accuracy}")
                progress_bar.refresh()

                return fold_total_train_loss, fold_total_train_accuracy
            
            return wrapper
        
        return _kFoldValidation


class Trainer:
    def __init__(self, train_dataset      : FSMLOneMeanStdDataset,              # The input train dataset
                       train_dataloader   : FSMLDataLoader,                     # The input train dataloader
                       optimizer          : optim.Optimizer,                    # The optimizer to use
                       model              : nn.Module,                          # The predictor
                       lr_scheduler       : optim.lr_scheduler._LRScheduler,    # The Learning Rate Scheduler
                       grad_clip          : int,                                # Gradient clipping value
                       num_epochs         : int                   = 30,         # Total number of epochs
                       criterion          : _Loss | _WeightedLoss = nn.MSELoss, # The Loss function to be used
                       k_fold             : int                   = 5,          # The number of fold for KFoldCrossValidation
                       accuracy_threshold : float                 = 0.94,       # Stop for accuracy grater than this
                       
                       imgs_path          : str = os.path.join(os.getcwd(), "img"),     # the path where to store the images
                       model_path         : str = os.path.join(os.getcwd(), "models")   # The path where to store the models
    ) -> None:
        """
        :param train_dataset: The input train dataset
        :param train_dataloader: The input train dataloader
        :param optimizer: The optimizer to use
        :param model: The predictor
        :param lr_scheduler: The Learning Rate Scheduler
        :param grad_clip: Gradient clipping value
        :param num_epochs: Total number of epochs
        :param criterion: The Loss function to be used
        :param num_hidden_input: Number of hidden layer input side
        :param num_hidden_output: Number of hidden layers output side
        :param hidden_input_size: Size of hidden input layers
        :param hidden_output_size: Size of hidden output layers
        :param model_path: The path where to store the models
        :param imgs_path: the path where to store the images
        :param k_fold: number of KFold Cross Validation runs
        :param accuracy_threshold: Stop when the current accuracy overcome a value
        """
        self.train_dataset      = train_dataset
        self.train_dataloder    = train_dataloader
        self.optimizer          = optimizer
        self.num_epochs         = num_epochs
        self.criterion          = criterion()
        self.model_path         = model_path
        self.imgs_path          = imgs_path
        self.model              = model
        self.lr_scheduler       = lr_scheduler
        self.grad_clip          = grad_clip
        self.k_fold             = k_fold
        self.use_kfold          = (k_fold != 0)
        self.accuracy_threshold = accuracy_threshold

        self.train_losses = []
        self.train_accs   = []

        filepath_linux_format = opath.basename(self.train_dataset.csv_file).replace('\\', '/')
        csv_filename = opath.basename(filepath_linux_format)
        self.model_filename = f"{csv_filename}_{self.model.__class__.__name__}"

        if self.k_fold != 0:
            self.model_filename += f"_KFoldCrossValidation{self.k_fold}"

        self.model_filepath = opath.join(self.model_path, self.model_filename + ".pth")

    def _run_epoch(self, epoch: int) -> Tuple[float, float]:
        """ Run one single epoch of training """

        @KFoldCrossValidationWrapper.kFoldValidation(
            self.train_dataset, self.model, epoch, self.k_fold, self.train_dataloder.batch_size, self.use_kfold)
        def __run_epoch(
            dataloader: Optional[FSMLDataLoader]=None, prog_bar: Optional[tqdm]=None
        ) -> Tuple[float, float]:
            # If the input dataloader is None then use the default one
            # The default dataloader is the one already defined in the class
            if not dataloader:
                dataloader = self.train_dataloder

            if not prog_bar:
                prog_bar = tqdm(dataloader(), desc=f"Epoch: {epoch} -- ", leave=True)
            
            train_loss = torch.zeros(1)
            train_acc  = 0.0
            train_step = 0
            self.optimizer.zero_grad()
            for _, (train_x, train_y, _) in enumerate(prog_bar):
                output      = self.model(train_x)
                loss        = self.criterion(output, train_y)
                train_loss += loss

                with torch.no_grad():
                    acc = utils.compute_accuracy(output, train_y)
                    train_acc += acc
                
                loss.backward()

                # Apply the gradient clipping
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                train_step += 1
                prog_bar.set_postfix_str(f"Train Loss: {(train_loss / train_step).item()}")
            
            train_loss /= train_step
            train_acc  /= train_step
            
            return train_loss.item(), train_acc
        
        return __run_epoch(self.train_dataloder)
    
    def run(self) -> str:
        """ Run the training and return the path of the model """
        # Set the model for training
        self.model.train()

        print("[*] Start training the model")
        start_time = time.time()
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self._run_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            if train_acc > self.accuracy_threshold:
                print(f"[*] Accuracy threshold reached. Stop")
                self.num_epochs = epoch + 1
                break

            self.lr_scheduler.step(train_loss)
        
        end_time = time.time()
        print(f"[*] Training procedure ended in {end_time - start_time} msec")

        print(f"[*] Saving the model {self.model_filepath}")
        torch.save(self.model.state_dict(), self.model_filepath)

        return self.model_filepath
    
    def plot(self) -> None:
        """ Plot the result (train loss and train acc over epochs) """
        img_filepath = opath.join(self.imgs_path, self.model_filename + ".png")

        epochs = list(range(self.num_epochs))
        torch_train_losses = torch.tensor(self.train_losses)
        torch_train_accs = torch.tensor(self.train_accs)
        torch_train_losses = torch_train_losses[torch_train_losses <= 1.0]

        start_index = len(epochs) - torch_train_losses.shape[0]
        epochs = epochs[start_index:]
        torch_train_accs = torch_train_accs[start_index:]

        figure = plt.figure(figsize=[15.0, 8.0])
        plt.plot(epochs, torch_train_losses, label="Train losses")
        plt.plot(epochs, torch_train_accs, label="Train accuracies")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Train losses and Accuracies")
        plt.legend(loc="upper right")
        figure.savefig(img_filepath)


def __train_one(train_dataset     : FSMLOneMeanStdDataset,
                criterion         : _Loss | _WeightedLoss,
                batch_size        : int,
                k_fold            : int,
                num_epochs        : int,
                model_type        : str,
                accuracy_threshold: float,
                patience          : float,
                min_lr            : float,
                grad_clip         : float,
                factor            : float,
                mode              : str,
                *args) -> Tuple[str, FSMLDataLoader]:
    """ Run one training with the input dataset and configuration """
    # Log the dataset for the training
    print("[*] Called training procedure with dataset")
    print(train_dataset)

    print("[*] Creating the dataloader")
    train_dataloader = FSMLDataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, drop_last=True
    )

    print("[*] Instantiating the Predictor")
    if model_type.lower() == "mlp":
        num_hidden_input, hidden_input_size, num_hidden_output, hidden_output_size = args
        predictor = mlp.FSML_MLP_Predictor(
            train_dataset.input_size,  num_hidden_input,  hidden_input_size,
            train_dataset.output_size, num_hidden_output, hidden_output_size
        )
    else:
        n_hidden_layer, hidden_sizes, basis_func = args
        predictor = rbf.FSML_RBF_Predictor(
            train_dataset.input_size, train_dataset.output_size,
            n_hidden_layer, hidden_sizes, basis_func
        )
    print(predictor)
    print(f"Models Parameters: {predictor.count_parameters()}")

    print("[*] Creating the Adam Optimizer and the LR scheduler")
    optimizer = optim.Adam(predictor.parameters(), lr=config.LR)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode=mode, factor=factor,
        patience=patience, verbose=True, min_lr=min_lr)
    print(optimizer, lr_scheduler)

    print("[*] Instantiating the Trainer")
    trainer = Trainer(
        train_dataset, train_dataloader,
        optimizer, predictor,
        lr_scheduler, grad_clip,
        num_epochs, criterion,
        k_fold, accuracy_threshold
    )

    model_path = trainer.run()
    trainer.plot()

    return model_path, train_dataloader


def train(path              : str,
          criterion         : _Loss | _WeightedLoss= config.CRITERION,
          batch_size        : int                  = config.BATCH_SIZE,
          k_fold            : int                  = config.KF_SPLIT,
          num_epochs        : int                  = config.NUM_EPOCHS,
          model_type        : str                  = config.MODEL_TYPE,
          accuracy_threshold: float                = config.ACCURACY_THRESHOLD,
          patience          : float                = config.PATIENCE,
          min_lr            : float                = config.MIN_LR,
          grad_clip         : float                = config.GRAD_CLIP,
          factor            : float                = config.FACTOR,
          mode              : str                  = config.MODE,
          *args) -> List[Tuple[str, FSMLDataLoader]]:
    r"""
    Run the training. The input `path` can be either a 
    path to a CSV file that contains the dataset, or to
    a folder with multiple CSV files (so multiple datasets).
    In the first case the training procedure will be runned
    only for that file, otherwise, in the second case,
    multiple times as the number of files. 

    :param path: the path to a CSV file or CSV folder
    :param criterion: the PyTorch type of loss (or a custom one)
    :param batch_size: the Size of the batch for the dataloader
    :param k_fold: number of cross fold validation
    :param num_epochs: The total number of epochs
    :param model_type: the type of the model to use
    :param accuracy_threshold: Stop when the current accuracy overcome a value
    :param patience: Number of epochs with no improvement after which learning rate will be reduced
    :param min_lr: A lower bound on the learning rate of all param groups
    :param grad_clip: the gradient clipping value
    :param factor: Factor by which the learning rate will be reduced
    :param mode: The mode with the scheduler will reduce the learning rate
    :return: A list of tuple (model_path, dataloader)
    """
    # Check if the input path is a file or a folder
    input_abspath = opath.abspath(path)
    assert opath.exists(input_abspath), \
        f"[!!!] ERROR: The input path: {path} does not exists."
    
    outputs = []
    
    # If it is a file then create a FSMLOneMeanStdDataset 
    # and run the training
    if opath.isfile(input_abspath):
        train_dataset = FSMLOneMeanStdDataset(input_abspath)
        return [__train_one(
            train_dataset, criterion, batch_size, 
            k_fold, num_epochs, model_type, accuracy_threshold, 
            patience, min_lr, grad_clip, factor, mode, *args
        )]
    
    # Otherwise it is a folder
    print(f"[*] Received in input a path: {path}")
    train_multi_dataset = FSMLMeanStdDataset(input_abspath)
    print(f"[*] Created the Multi-Dataset")
    print(train_multi_dataset)

    for idx, train_dataset in enumerate(train_multi_dataset):
        print(f": ---------------- : ({idx}) Using {train_dataset.csv_file} : ---------------- :")
        output = __train_one(
            train_dataset, criterion, batch_size, 
            k_fold, num_epochs, model_type, accuracy_threshold, 
            patience, min_lr, grad_clip, factor, mode, *args
        )
        outputs.append(output)

    return outputs


def main() -> None:
    file = os.path.join(os.getcwd(), "data/meanstd/")
    train(file, k_fold=3, num_epochs=150)


if __name__ == "__main__":
    main()