import fsml.learn.data_mangement.dataset as dataset
import os.path as opath
import os 


def iter_dataset():
    data_dir = opath.join(os.getcwd(), "data/simulations/")
    ds = dataset.SimulationDataset(data_dir)
    print(ds)

    for (x, y) in ds:
        ...


if __name__ == "__main__":
    iter_dataset()