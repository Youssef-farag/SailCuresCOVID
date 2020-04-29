import numpy as np
import torch
import torchvision.transforms as tvt
from skimage.color import rgb2gray
from torch.utils.data import Dataset


class CovidDatasetTrain(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        sample = self.imgs[idx] * 255.

        transforms = tvt.Compose([tvt.ToPILImage(),
                     tvt.RandomChoice([tvt.RandomRotation((0,0)),
                                       tvt.RandomRotation((90,90)),
                                       tvt.RandomRotation((180,180)),
                                       tvt.RandomRotation((-90,-90))]),
                     tvt.RandomChoice([tvt.RandomHorizontalFlip(),
                                       tvt.RandomVerticalFlip()]),
                     # tvt.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
                     tvt.Resize(256)])

        sample = transforms(sample)
        original = np.asarray(sample)
        grayscale = rgb2gray(original)
        sample = tvt.ToTensor()(grayscale).float()

        return sample, self.labels[idx]


class CovidDatasetVal(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        sample = tvt.ToPILImage()(self.imgs[idx] * 255.)
        rotate = [tvt.RandomRotation(angle)(sample) for angle in [(0, 0), (90, 90), (180, 180), (-90, -90)]]
        hflip = [tvt.RandomHorizontalFlip(p=1.)(img) for img in rotate]
        vflip = [tvt.RandomVerticalFlip(p=1.)(img) for img in rotate]
        batch = rotate + hflip + vflip
        batch = [tvt.Resize(256)(img) for img in batch]
        batch = [tvt.ToTensor()(rgb2gray(np.asarray(img))).float().unsqueeze(0) for img in batch]

        labels = torch.ones((len(batch), 1)) * self.labels[idx]

        return torch.cat(batch, dim=0), labels


class CovidDatasetTest(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, imgs):
        self.imgs = imgs

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        sample = tvt.ToPILImage()(self.imgs[idx] * 255.)
        rotate = [tvt.RandomRotation(angle)(sample) for angle in [(0,0),(90,90),(180,180),(-90,-90)]]
        hflip = [tvt.RandomHorizontalFlip(p=1.)(img) for img in rotate]
        vflip = [tvt.RandomVerticalFlip(p=1.)(img) for img in rotate]
        batch = rotate + hflip + vflip
        batch = [tvt.Resize(256)(img) for img in batch]
        batch = [tvt.ToTensor()(rgb2gray(np.asarray(img))).float().unsqueeze(0) for img in batch]

        return torch.cat(batch, dim=0)
