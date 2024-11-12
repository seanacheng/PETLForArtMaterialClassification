from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import pandas as pd
import os
import tarfile
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

        # Better to check now than to find out while training:
        self.subdir = re.search(r'all-(.+)\.csv', csv_file).group(1) + "_jpg"
    
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
        
        with tarfile.open(self._img_dir, 'r') as tar:
            # Extract the file object for the specified file within the tar
            img_path = os.path.join(self.subdir, self._df.loc[idx, "jpg"])
            file_obj = tar.extractfile(img_path)
            if file_obj is None:
                raise FileNotFoundError(f"The file '{img_path}' was not found in {self._img_dir}.")
            
            img = read_image(
                path = file_obj,
                mode = ImageReadMode.RGB
            ).float() / 255
            
        x = img
        if self._transform:
            x = self._transform(x)
        
        y = self._df.loc[idx, "idx"]
        if self._target_transform:
            y = self._target_transform(y)
        
        return x, y