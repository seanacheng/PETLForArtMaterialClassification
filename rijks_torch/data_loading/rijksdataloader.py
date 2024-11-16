from .rijksdataset import RijksDataset
import os
from torch.utils.data import DataLoader
import pandas as pd

class RijksDataloader:
    """
    Generates dataloaders for the training, validation, and testing set
    """

    def make_data_loaders(batch_size: int, transform):

        # where the images are in the HPC
        img_dir = '/cluster/tufts/cs152l3dclass/shared/rijksdata'

        # Get a list of possible materials:
        materials = pd.read_csv('data_annotations/all-hist.csv')["material"].to_list()

        train_dataset = RijksDataset("data_annotations/all-train.csv", materials, os.path.join(img_dir, "train_jpg"), transform)
        val_dataset  = RijksDataset("data_annotations/all-val.csv", materials, os.path.join(img_dir, "val_jpg"), transform)
        test_dataset = RijksDataset("data_annotations/all-test.csv", materials, os.path.join(img_dir, "test_jpg"), transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader