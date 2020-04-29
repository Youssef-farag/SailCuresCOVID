import gc
import pandas as pd
import torch.nn as nn
import pretrainedmodels
from data_utils import *


def eval_ensemble():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0

    # iterate over the 70 models
    for index in range(70):

        model = pretrainedmodels.__dict__['resnet18'](num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Linear(model.last_linear.in_features, 1)
        model.inplanes = 64
        model.conv1 = nn.Conv2d(1, model.inplanes, kernel_size=7, stride=2, padding=3,
                                bias=False)
        model.to(device)

        print("Evaluating model " + str(index) + "...")
        # load model from disk
        model.load_state_dict(torch.load("./models/model_" + str(index)))
        with torch.no_grad():
            model.eval()
            # get the data
            test_loader = make_val_data_loader(indices=[index])

            for data in test_loader:
                samples, labels = data[0].squeeze(0), data[1].squeeze(0)
                preds = model(samples.to(device))

                if len(preds[np.where(labels == 0)[0], :]) > 1:
                    print('Prediction on benign is: ',
                          np.asarray(nn.Sigmoid()(preds[np.where(labels == 0)[0], :]).detach().cpu().double()))

                for pred, label in zip(preds, labels):
                    if nn.Sigmoid()(pred).round().item() == label.numpy():
                        correct += 1

    print('Accuracy is {}, {} correct'.format(correct / 70, correct))


def generate_predictions():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loader = make_test_data_loader()

    pred_dict = {}

    # iterate over the 70 models
    for index in range(70):

        model = pretrainedmodels.__dict__['resnet18'](num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Linear(model.last_linear.in_features, 1)
        model.inplanes = 64
        model.conv1 = nn.Conv2d(1, model.inplanes, kernel_size=7, stride=2, padding=3,
                                bias=False)
        model.to(device)

        print("Evaluating model " + str(index) + "...")
        # load model from disk
        model.load_state_dict(torch.load("./models/model_" + str(index)))
        with torch.no_grad():
            model.eval()
            # get the data

            for idx, samples in enumerate(test_loader):
                samples = samples.squeeze(0)
                preds = model(samples.to(device))
                preds = np.mean(nn.Sigmoid()(preds).detach().cpu().numpy(), axis=1)

                if index == 0:
                    pred_dict[str(idx)] = [preds]
                else:
                    pred_dict[str(idx)].append(preds)

    for i in range(20):
        pred_dict[str(i)] = np.mean(pred_dict[str(i)])

    np.save('./models/final_predictions.npy', pred_dict)
    print(pred_dict)
    df = pd.DataFrame(data=pred_dict, index=[0])
    df.to_csv('./models/final_predictions.csv')


def val_phase_cv(model, data_loader, device):
    correct = 0

    with torch.no_grad():
        model.eval()

        for samples, labels in data_loader:
            samples, labels = samples.squeeze(0), labels.squeeze(0)

        preds = model(samples.to(device))

        print('Label is {}, predictions are: '.format(labels[0]),
              np.asarray(nn.Sigmoid()(preds).detach().cpu().double()))

        for pred, label in zip(preds, labels):
            if nn.Sigmoid()(pred).round().item() == label.numpy():
                correct += 1

    print('Accuracy is {}, {} correct, average pred {}'.
          format(correct / (len(data_loader)*12), correct,
                 np.asarray(nn.Sigmoid()(preds).detach().cpu().double()).mean()))
    model.train()


def train(model, data_loader, criterion, optimizer, device, epochs):

    for epoch in range(epochs):
        preds = []
        labels = []

        optimizer.zero_grad()
        for sample, label in data_loader:
            pred = model(sample.to(device))
            preds.append(pred.cpu())
            labels.append(label.unsqueeze(1))
            # if len(pred[np.where(label == 0)[0], :]) > 0:
                # print(np.asarray(nn.Sigmoid()(pred[np.where(label == 0)[0], :]).detach().cpu().double()))

        if len(preds) > 1:
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0).float()
        else:
            preds = preds[0]
            labels = labels[0].float()

        loss = criterion(preds, labels)
        # print('Loss is {}'.format(loss.item()))
        loss.backward()
        optimizer.step()
        gc.collect()