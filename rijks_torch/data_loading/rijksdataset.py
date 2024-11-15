from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import pandas as pd
import os
import re

class RijksDataset(Dataset):
    """A class that encapsulates the Rijksmuseum Challenge dataset."""
    
    def __init__(self, csv_file, materials, img_dir, transform=None, target_transform=None):
        """
        ## Default constructor

        :param csv_file:  A file containing [*.jpg, 'material'] pairs for each element in the dataset\n
        :param materials: A list containing all the materials, such that a ML model can learn to
                    predict indeces into this list\n
        :param img_dir:   Directory containing all .jpg files mentioned in :csv_file:\n
        :param transform and target_transform: apply transforms to input and output resp.\n
        """
        self._df = RijksDataset._processTable(csv_file, materials)
        self._img_dir = img_dir
        self._transform = transform
        self._target_transform = target_transform
    
    def _processTable(csv_file, materials):
        """
        Returns pd.DataFrame containing [*.jpg, idx] pairs, such that materials[idx] == 'material'.
        """
        df = pd.read_csv(csv_file, index_col=False)

        try:
            df["idx"] = df["material"].map(lambda mat: materials.index(mat))
        except Exception as e:
            raise Exception("Can't map all material strings to indeces.") from e

        df.drop(columns="material", inplace=True)

        return df
    
    def __len__(self):
        """Returns number of samples in dataset"""
        return len(self._df)
    
    def __getitem__(self, idx):
        """Get x (image) and y (material index into materials list) at idx"""
        x = read_image(
            path = os.path.join(self._img_dir, self._df.loc[idx, "jpg"]),
            mode = ImageReadMode.RGB
        ).float() / 255
        if self._transform:
            x = self._transform(x)
        
        y = self._df.loc[idx, "idx"]
        if self._target_transform:
            y = self._target_transform(y)
        
        return x, y