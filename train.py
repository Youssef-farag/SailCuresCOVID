from __future__ import print_function, division
import pretrainedmodels
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random

from data_utils import *

random.seed(0)
torch.manual_seed(0)
torch.set_printoptions(precision=20)
np.set_printoptions(suppress=True)
np.random.seed(0)


def val_phase(model, data_loaders):
    correct = 0
    benign = 0
    with torch.no_grad():
        model.eval()
        for samples, labels in tqdm(data_loaders['val']):
            samples, labels = samples.squeeze(0), labels.squeeze(0)
            preds = model(samples.to(device))

            if len(preds[np.where(labels == 0)[0], :]) > 1:
                print(np.asarray(nn.Sigmoid()(preds[np.where(labels == 0)[0], :]).detach().cpu().double()))

            for pred, label in zip(preds, labels):
                if nn.Sigmoid()(pred).round().item() == label.numpy():
                    correct += 1

    print('Accuracy is {}, {} correct'.format(correct / (len(data_loaders['val'])*12), correct))
    model.train()


def test_phase(model, data_loaders):

    with torch.no_grad():
        model.eval()
        for samples in tqdm(data_loaders['test']):
            samples = samples.squeeze(0)
            preds = nn.Sigmoid()(model(samples.to(device)))

    model.train()


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_loaders = prep_data()

    model_name = 'resnet18'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model.last_linear = nn.Linear(model.last_linear.in_features, 1)
    model.inplanes = 64
    model.conv1 = nn.Conv2d(1, model.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
    model.to(device)

    bigfella = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([0.3]))
    correct = 0

    for epoch in tqdm(range(20)):
        preds = []
        labels = []
        benigns = []

        bigfella.zero_grad()
        for sample, label in data_loaders["train"]:
            pred = model(sample.to(device))
            preds.append(pred.cpu())
            labels.append(label.unsqueeze(1))
            if len(pred[np.where(label == 0)[0], :]) > 0:
                print(np.asarray(nn.Sigmoid()(pred[np.where(label == 0)[0], :]).detach().cpu().double()))

        if len(preds) > 1:
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0).float()
        else:
            preds = preds[0]
            labels = labels[0].float()

        loss = criterion(preds, labels)
        print('Loss is {}'.format(loss.item()))
        loss.backward()
        bigfella.step()

    val_phase(model, data_loaders)
    test_phase(model, data_loaders)
