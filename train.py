from __future__ import print_function, division
import torch
import pretrainedmodels
import torch.nn as nn
from data_handlers import *
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision.transforms as tvt
import pickle


def prep_data():
    test_imgs = pickle.load(open("../data/test_images_512.pk", 'rb'), encoding='bytes')
    train_imgs = pickle.load(open("../data/train_images_512.pk", 'rb'), encoding='bytes')
    train_labels = pickle.load(open("../data/train_labels_512.pk", 'rb'), encoding='bytes')

    data_loaders = make_data_loaders(train_imgs, train_labels, test_imgs)
    return data_loaders


if __name__=='__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_loaders = prep_data()

    model_name = 'resnet18'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model.last_linear = nn.Linear(model.last_linear.in_features, 1)
    model.to(device)

    bigfella = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([0.3]))
    correct = 0

    with torch.no_grad():
        model.eval()
        for sample, label in data_loaders['val']:
            preds = model(sample.to(device))
            if np.rint(nn.Sigmoid()(preds.cpu().numpy())) == label.numpy():
                correct += 1

    print('Accuracy before training is {}, {} correct'.format(correct/len(data_loaders['val']), correct))
    model.train()

    for epoch in range(15):
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

    correct = 0

    with torch.no_grad():
        model.eval()
        for sample, label in data_loaders['val']:
            preds = nn.Sigmoid()(model(sample.to(device)))
            if np.rint(preds.cpu().numpy()) == label.numpy():
                correct += 1

    print('Accuracy after training is {}, {} correct'.format(correct/len(data_loaders['val']), correct))
    model.train()
