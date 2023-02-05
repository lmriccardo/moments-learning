from fsml.learn.train import train
from fsml.learn.test import test
import os


def main() -> None:
    file = os.path.join(os.getcwd(), "data/meanstd/")
    outputs = train(file, k_fold=3, num_epochs=150)
    test(outputs)


main()