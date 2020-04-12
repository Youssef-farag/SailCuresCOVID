import pickle
from dataset_classes import *
from torch.utils.data import DataLoader


def make_data_loaders(train_imgs, train_labels, test_imgs):
    train_dataset = CovidDatasetTrain(train_imgs[:50], train_labels[:50])
    val_dataset = CovidDatasetTrain(train_imgs[50:], train_labels[50:])
    test_dataset = CovidDatasetTest(test_imgs)

    return {
        "train": DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1),
        "val": DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1),
        "test": DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1),
    }


def prep_data():
    test_imgs = pickle.load(open("../data/test_images_512.pk", 'rb'), encoding='bytes')
    train_imgs = pickle.load(open("../data/train_images_512.pk", 'rb'), encoding='bytes')
    train_labels = pickle.load(open("../data/train_labels_512.pk", 'rb'), encoding='bytes')

    data_loaders = make_data_loaders(train_imgs, train_labels, test_imgs)
    return data_loaders

