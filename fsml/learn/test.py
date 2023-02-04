import torch

from fsml.learn.data_mangement.dataloader import FSMLDataLoader
from fsml.learn.models.nets import FSML_MLP_Predictor
from fsml.utils import compute_accuracy
import os.path as opath


class Tester:
    def __init__(self, model_path         : str,             # The path of the model
                       test_dataloader    : FSMLDataLoader,  # The dataloader for testing
                       num_hidden_input   : int,             # Number of hidden layer input side
                       num_hidden_output  : int,             # Number of hidden layers output side
                       hidden_input_size  : int,             # Size of hidden input layers
                       hidden_output_size : int              # Size of hidden output layers
    ) -> None:
        """
        :param model_path: the path where the model has been saved
        :param train_dataloader: the dataloader for testing
        """
        self.model_path         = model_path
        self.test_dataloader    = test_dataloader
        self.num_hidden_input   = num_hidden_input
        self.num_hidden_output  = num_hidden_output
        self.hidden_input_size  = hidden_input_size
        self.hidden_output_size = hidden_output_size

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