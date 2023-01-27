import fsml.learn.data_mangement.dataset as dataset
import fsml.learn.data_mangement.dataloader as dataloader
import os.path as opath
import os 


def iter_dataset():
    data_dir = opath.join(os.getcwd(), "data/simulations/")
    ds = dataset.FSMLDataset(data_dir)
    dl = dataloader.FSMLDataLoader(ds, 3, True)
    for data in dl():
        print(data[0])
        print(data[1])
        print()


if __name__ == "__main__":
    iter_dataset()