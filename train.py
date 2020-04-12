from __future__ import print_function, division

import torch
import pretrainedmodels
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tvt
import numpy as np
from tqdm import tqdm
import random

from data_utils import *

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


def val_phase(model, data_loaders):
    correct = 0
    with torch.no_grad():
        model.eval()
        for sample, label in tqdm(data_loaders['val']):
            preds = model(sample.to(device))
            if nn.Sigmoid()(preds).round().item() == label.numpy():
                correct += 1

    print('Accuracy before training is {}, {} correct'.format(correct/len(data_loaders['val']), correct))
    model.train()


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_loaders = prep_data()

    model_name = 'resnet18'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model.last_linear = nn.Linear(model.last_linear.in_features, 1)
    model.to(device)

    bigfella = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([0.3]))
    correct = 0

    val_phase(model, data_loaders)

    for epoch in tqdm(range(150)):
        preds = []
        labels = []

        bigfella.zero_grad()
        for sample, label in data_loaders["train"]:
            sample = tvt.ToTensor()(tvt.RandomCrop(350)(tvt.ToPILImage()(sample.squeeze(0)))).unsqueeze(0)
            pred = model(sample.to(device))
            preds.append(pred.cpu().squeeze(0))
            labels.append(label)
            if label == 0:
                print(nn.Sigmoid()(pred).item())

        loss = criterion(torch.stack(preds, dim=1), torch.stack(labels, dim=1).float())
        print('Loss is {}'.format(loss.item()))
        loss.backward()
        bigfella.step()

    val_phase(model, data_loaders)
