import fsml.learn.data_mangement.dataset as dataset
import fsml.learn.data_mangement.dataloader as dataloader
import fsml.learn.models.net as nets
import os.path as opath
import os 


def iter_dataset():
    # data_dir = opath.join(os.getcwd(), "data/meanstd/")
    # ds = dataset.FSMLMeanStdDataset(data_dir)
    # dl = dataloader.FSMLDataLoader(ds, 3, True)
    # model = nets.FSMLSimpleNetwork(input_size=ds.max_parameters)
    # for data in dl():
    #     input_data, _ = data
    #     output = model(input_data)
    #     print(output)

    data_dir = opath.join(os.getcwd(), "data/denseoutputs/")
    denseoutput_ds = dataset.FSMLOneStepAheadDataset(data_dir)
    for x_data, y_data in denseoutput_ds:
        print(x_data, y_data)
        print()
        break


if __name__ == "__main__":
    iter_dataset()