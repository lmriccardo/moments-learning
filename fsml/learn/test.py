import torch

from fsml.learn.data_mangement.dataloader import FSMLDataLoader
from fsml.learn.models.mlp import FSML_MLP_Predictor
from fsml.utils import compute_accuracy
import os.path as opath
from typing import List, Tuple
import fsml.learn.config as config


class Tester:
    def __init__(self, model_path         : str,             # The path of the model
                       test_dataloader    : FSMLDataLoader,  # The dataloader for testing
                       **kwargs
    ) -> None:
        """
        :param model_path: the path where the model has been saved
        :param train_dataloader: the dataloader for testing
        """
        self.model_path         = model_path
        self.test_dataloader    = test_dataloader

        self.num_hidden_input   = kwargs["num_hidden_input"]
        self.hidden_input_size  = kwargs["hidden_input_size"]
        self.num_hidden_output  = kwargs["num_hidden_output"]
        self.hidden_output_size = kwargs["hidden_output_size"]

        # Check that the model path really exists
        assert opath.exists(opath.abspath(model_path)), \
            f"[!!!] ERROR: The model path {model_path} does not exists"
        
    def test(self) -> float:
        """ Just run the test and return the final accuracy """
        print("[*] Starting testing procedure")
        # Initialize the result
        test_step = 1
        test_accuracy = 0.0

        # If the is_train attribute of the dataset is 
        # set to True, we need to set it to false
        if self.test_dataloader.dataset.is_train:
            self.test_dataloader.dataset.test()

        # Create an initial predictor
        test_dataset = self.test_dataloader.dataset

        predictor = FSML_MLP_Predictor(
            test_dataset.input_size,  self.num_hidden_input,  self.hidden_input_size,
            test_dataset.output_size, self.num_hidden_output, self.hidden_output_size,
        )

        # Load the state dict from the saved model
        predictor.load_state_dict(torch.load(opath.abspath(self.model_path)))
        predictor.eval()

        print("[*] Saved model loaded into a new predictor")
        print("[*] Computing")

        for x_data, y_data, _ in self.test_dataloader():
            output         = predictor(x_data)
            acc            = compute_accuracy(output, y_data)
            test_accuracy += acc
            test_step     += 1

        test_accuracy /= test_step

        print("[*] Testing procedure ended succesfully")
        return test_accuracy
    

def test(paths_and_dataloaders: List[Tuple[str, FSMLDataLoader]],
         model_type           : str = config.MODEL_TYPE,
         **kwargs) -> List[float]:
    r"""
    Run the testing phase against some pre-trained models.

    :param paths_and_dataloaders: a list of tuple (model_path, dataloader)
    :param num_hidden_input: The number of hidden layer in input side
    :param num_hidden_output: The number of hidden layer in output side
    :param hidden_input_size: The number of neurons for each input hidden layer
    :param hidden_output_size: The number of neurons for each output hidden layer
    :return: The list with all the accuracies
    """
    final_accuracies = []
    for idx, (model_path, dataloader) in enumerate(paths_and_dataloaders):
        print(f": ---------------- : ({idx}) Test Model {model_path} : ---------------- :")
        tester = Tester(model_path, dataloader, model_type, **kwargs)

        final_acc = tester.test()
        print(f"FINAL ACCURACY: {final_acc * 100:.5f}")
        final_accuracies.append(final_acc)
    
    return final_accuracies