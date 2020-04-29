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
from cross_validation import k_folds

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


def eval_ensemble(model):
    correct = 0
    # iterate over the 70 models
    for index in range(70):
        print("Evaluating model " + str(index) + "...")
        # load model from disk
        model.load_state_dict(torch.load("./models/model_" + str(index)))
        with torch.no_grad():
            model.eval()
            # get the data
            dataset_test = indexDataset(indices=[index])
            test_loader = make_test_data_loader(dataset_test)
            for data in test_loader:
                sample, label = data
                preds = model(sample.to(device))
                if nn.Sigmoid()(preds).round().item() == label.numpy():
                    correct += 1

    print('Accuracy is {}, {} correct'.format(correct / 70, correct))


def val_phase(model, data_loaders):

    correct = 0
    with torch.no_grad():
        model.eval()
        for sample, label in tqdm(data_loaders['val']):
            preds = model(sample.to(device))
            if nn.Sigmoid()(preds).round().item() == label.numpy():
                correct += 1

    print('Accuracy is {}, {} correct'.format(correct / len(data_loaders['val']), correct))
    model.train()


def val_phase_cv(model, data_loaders):
    correct = 0
    with torch.no_grad():
        model.eval()
        for sample, label in tqdm(data_loaders):
            preds = model(sample.to(device))
            if nn.Sigmoid()(preds).round().item() == label.numpy():
                correct += 1

    print('Accuracy is {}, {} correct'.format(correct / len(data_loaders), correct))
    model.train()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_loaders = prep_data()

    model_name = 'resnet18'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model.last_linear = nn.Linear(model.last_linear.in_features, 1)
    model.to(device)

    opt = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([0.3]))
    correct = 0

    val_phase(model, data_loaders)

    for train_idx, test_idx in tqdm(k_folds(n_splits=70)):
        dataset_train = indexDataset(indices=train_idx)
        dataset_test = indexDataset(indices=test_idx)
        train_loader = make_train_data_loader(dataset_train)
        test_loader = make_test_data_loader(dataset_test)

        for epoch in range(0):
            preds = []
            labels = []

            opt.zero_grad()
            for sample, label in train_loader:
                sample = tvt.ToTensor()(tvt.RandomCrop(350)(tvt.ToPILImage()(sample.squeeze(0)))).unsqueeze(0)
                pred = model(sample.to(device))
                preds.append(pred.cpu().squeeze(0))
                labels.append(label)
                if label == 0:
                    print(nn.Sigmoid()(pred).item())

            loss = criterion(torch.stack(preds, dim=1), torch.stack(labels, dim=1).float())
            print('Loss is {}'.format(loss.item()))
            loss.backward()
            opt.step()

        # evaluate val
        val_phase_cv(model, test_loader)
        # save model to disk
        torch.save(model.state_dict(), "./models/model_" + str(test_idx[0]))

    # evaluate the 70 models
    eval_ensemble(model)
