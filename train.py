from __future__ import print_function, division
import pretrainedmodels
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random

from data_utils import *
from cross_validation import k_folds

random.seed(0)
torch.manual_seed(0)
torch.set_printoptions(precision=20)
np.set_printoptions(suppress=True)
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
    benign = 0
    with torch.no_grad():
        model.eval()
        for samples, labels in tqdm(data_loaders['val']):
            samples, labels = samples.squeeze(0), labels.squeeze(0)
            preds = model(samples.to(device))

            if len(preds[np.where(labels == 0)[0], :]) > 1:
                print('Prediction on benign is: ', np.asarray(nn.Sigmoid()(preds[np.where(labels == 0)[0], :]).detach().cpu().double()))

            for pred, label in zip(preds, labels):
                if nn.Sigmoid()(pred).round().item() == label.numpy():
                    correct += 1

    print('Accuracy is {}, {} correct'.format(correct / (len(data_loaders['val'])*12), correct))
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
            opt.step()

        # evaluate val
        val_phase_cv(model, test_loader)
        # save model to disk
        torch.save(model.state_dict(), "./models/model_" + str(test_idx[0]))

    # evaluate the 70 models
    eval_ensemble(model)