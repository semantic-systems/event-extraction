import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class DatasetTRECIS(Dataset):
    classification_type: str = "single_label"

    def __init__(self, df: pd.Dataframe, sentence_column: str, label_column: str):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label