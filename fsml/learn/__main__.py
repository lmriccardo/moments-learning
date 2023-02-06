from fsml.learn.train import train
from fsml.learn.test import test
import os


def main() -> None:
    file = os.path.join(os.getcwd(), "data/meanstd/BIOMD00006_MeanStd.csv")
    outputs = train(file, k_fold=5, num_epochs=200)
    test(outputs)


main()