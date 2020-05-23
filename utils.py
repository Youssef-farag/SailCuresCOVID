import pickle
import pretrainedmodels
import torch.nn as nn
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


def prep_model(args):
    model = pretrainedmodels.__dict__['resnet18'](num_classes=1000, pretrained='imagenet')
    model.last_linear = nn.Linear(model.last_linear.in_features, 1)
    model.inplanes = 64
    model.conv1 = nn.Conv2d(1, model.inplanes, kernel_size=7, stride=2, padding=3,
                            bias=False)
    model.to(args['device'])

    return model