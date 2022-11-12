import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import pandas as pd

class LD(Dataset):
    def __init__(self, img_path: str, labels: pd.DataFrame, 
                vid_list: list, newsize = [256, 256]):
        
        # Construct a list of paths to train/val images
        self.img_paths = []
        
        for vid in vid_list:
            img_paths = glob.glob(img_path + vid.split('.')[0] + '_*')
            self.img_paths += img_paths

        self.newsize = newsize
        self.labels = labels

    def __len__(self):
        return len(self.img_paths)

    def preprocess(self, pil_img):
        newW, newH = int(self.newsize[0]), int(self.newsize[1])

        pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)

        img_ndarray = np.asarray(pil_img)
        img_ndarray = img_ndarray.transpose((2, 0, 1))
        img_ndarray = img_ndarray / 255

        return img_ndarray

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img = Image.open(img_path)
        img = self.preprocess(img)

        vidname = img_path.split('/')[-1].split('_')[0] + '.mp4'

        label = int(self.labels.loc[vidname]['liveness_score'])

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'label': torch.tensor(label)
        }

class LD_test(Dataset):
    def __init__(self, img_path: str, newsize = [256, 256]):
        self.img_paths = glob.glob(img_path + '*')
        self.newsize = newsize

    def __len__(self):
        return len(self.img_paths)

    def preprocess(self, pil_img):
        newW, newH = int(self.newsize[0]), int(self.newsize[1])

        pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)

        img_ndarray = np.asarray(pil_img)
        img_ndarray = img_ndarray.transpose((2, 0, 1))
        img_ndarray = img_ndarray / 255

        return img_ndarray

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img = Image.open(img_path)
        img = self.preprocess(img)

        vidname = img_path.split('/')[-1].split('_')[0]

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'vidname': torch.tensor(int(vidname))
        }

def split_data(label_file: str = 'data/train/label.csv', val_frac: float = 0.1):
    df = pd.read_csv(label_file)
    df.set_index("fname", inplace = True)

    val_split = df.sample(frac=val_frac)
    train_split = df.drop(val_split.index).index.to_list()
    val_split = val_split.index.to_list()

    return train_split, val_split, df