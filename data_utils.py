import pickle
from dataset_classes import *
from torch.utils.data import DataLoader


def make_train_data_loader(indices):
    imgs = pickle.load(open("../data/train_images_512.pk", 'rb'), encoding='bytes')[indices]
    labels = pickle.load(open("../data/train_labels_512.pk", 'rb'), encoding='bytes')[indices]
    dataset = CovidDatasetTrain(imgs, labels)

    return DataLoader(dataset, batch_size=69, shuffle=False, num_workers=1, drop_last=False)


def make_val_data_loader(indices):
    imgs = pickle.load(open("../data/train_images_512.pk", 'rb'), encoding='bytes')[indices]
    labels = pickle.load(open("../data/train_labels_512.pk", 'rb'), encoding='bytes')[indices]
    dataset = CovidDatasetVal(imgs, labels)

    return DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False)


def make_test_data_loader():
    imgs = pickle.load(open("../data/test_images_512.pk", 'rb'), encoding='bytes')
    dataset = CovidDatasetTest(imgs)

    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
