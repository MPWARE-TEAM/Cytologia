import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import transformers
import torchvision
from cdc.common.constants import LABEL


class CDCDataset(Dataset):
    def __init__(self, df, config, mode='train', preprocess=None, augment=None, prepare=None, background_preprocess=None):
        self.df = df
        self.mode = mode
        self.config = config
        self.preprocess = preprocess
        self.augment = augment
        self.prepare = prepare
        self.background_preprocess = background_preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.iloc[idx]
        filename = row[self.config.filename_col]

        # Read RGB image
        if isinstance(self.config.images_home, list):
            for images_home in self.config.images_home:
                image_path = os.path.join(images_home, filename)
                if os.path.exists(image_path):
                    break
        else:
            image_path = os.path.join(self.config.images_home, filename)

        if os.path.exists(image_path):
            image = np.array(Image.open(image_path))
        else:
            raise(Exception("Image not found: %s" % image_path))

        sample = {
            'image': image,
            'filename': filename,
        }
        if self.mode in ['train', 'valid']:
            # OHE
            if self.config.label_col == LABEL:
                sample['label'] = row[self.config.label_col]
            else:
                class_ = int(row[self.config.label_col])
                label = np.zeros(self.config.num_labels, dtype=np.float32)
                label[class_] = 1.
                sample['label'] = label

        # Optional preprocessing on background RGB image
        if (self.background_preprocess) and ('background' in filename):
            tmp = self.background_preprocess(image=sample['image'])
            sample['image'] = tmp["image"]  # Apply on full image

        # Optional preprocessing on RGB image
        if self.preprocess:
            tmp = self.preprocess(image=sample['image'])
            sample['image'] = tmp["image"]  # Apply on full image

        # Optional augmentation on RGB image
        if self.augment:
            tmp = self.augment(image=sample['image'])
            sample['image'] = tmp["image"]  # Apply on full image

        # Mandatory to feed model (normalization, convert to CHW)
        if self.prepare:
            if isinstance(self.prepare, transformers.models.bit.image_processing_bit.BitImageProcessor):
                tmp = self.prepare(images=sample['image'], return_tensors="pt")['pixel_values'][0]
                sample['image'] = tmp
            elif isinstance(self.prepare, torchvision.transforms.transforms.Compose):
                img = sample['image']
                if isinstance(self.prepare.transforms[0], torchvision.transforms.transforms.Resize):
                    img = Image.fromarray(img)  # To PIL image
                tmp = self.prepare(img)
                sample['image'] = tmp
            else:
                tmp = self.prepare(image=sample['image'])
                sample['image'] = tmp["image"]  # Apply on full image

        return sample
