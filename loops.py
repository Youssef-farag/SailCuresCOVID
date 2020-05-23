import gc
import os
import pandas as pd
import torch.optim as optim
from utils import *


def eval_ensemble(args):
    device = args['device']
    correct = 0

    # iterate over the 70 models
    for index in range(70):
        print("Evaluating model " + str(index) + "...")

        model = prep_model(args)

        # load model from disk
        model.load_state_dict(torch.load(os.path.join(args['results_dir'], "model_" + str(index))))
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

    print('Accuracy is {}, {} correct'.format(correct / (70*12), correct))


def generate_predictions(args):
    device = args['device']
    test_loader = make_test_data_loader()

    pred_dict = {}

    # iterate over the 70 models
    for index in range(70):

        model = prep_model(args)

        print("Evaluating model " + str(index) + "...")
        # load model from disk
        model.load_state_dict(torch.load(os.path.join(args['results_dir'], "model_" + str(index))))
        with torch.no_grad():
            model.eval()
            # get the data

            for idx, samples in enumerate(test_loader):
                samples = samples.squeeze(0)
                preds = model(samples.to(device))
                preds = np.mean(nn.Sigmoid()(preds).detach().cpu().numpy(), axis=1)

                if index == 0:
                    pred_dict[idx] = [preds]
                else:
                    pred_dict[idx].append(preds)

    np.save(os.path.join(args['results_dir'], 'final_predictions.npy'), pred_dict)

    for i in range(20):
        pred_dict[i] = np.mean(pred_dict[i]).round() == 1
    df = pd.DataFrame(data=pred_dict, index=[0]).T
    df.to_csv(os.path.join(args['results_dir'], 'final_predictions.csv'))


def val_phase_cv(model, model_idx, criterion, args, verbose=False):
    device = args['device']

    data_loader = make_val_data_loader(model_idx)

    correct = 0

    with torch.no_grad():
        model.eval()

        for samples, labels in data_loader:
            samples, labels = samples.squeeze(0), labels.squeeze(0)

        preds = model(samples.to(device))

        if verbose:
            print('Label is {}, predictions are: '.format(labels[0]),
                  np.asarray(nn.Sigmoid()(preds).detach().cpu().double()))

        loss = criterion(preds.detach().cpu(), labels)

        for pred, label in zip(preds, labels):
            if nn.Sigmoid()(pred).round().item() == label.numpy():
                correct += 1
    if verbose:
        print('Loss is {}, {} correct, average pred {}'.
              format(loss, correct,
                     np.asarray(nn.Sigmoid()(preds).detach().cpu().double()).mean()))
    model.train()
    return loss, labels[0]


def train(train_loader, model_idx, args):
    model = prep_model(args)
    device = args['device']

    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    # learning rate scheduler
    if args['schedule']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                         patience=args['scheduler']['patience'])
    else:
        scheduler = None

    # loss function with weighting
    criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([args['loss_weight']]))

    best_val = np.inf

    for epoch in range(args['train_epochs']):
        preds = []
        labels = []
        optimizer.zero_grad()

        for sample, label in train_loader:
            pred = model(sample.to(device))
            preds.append(pred.cpu())
            labels.append(label.unsqueeze(1))

        # check to see if epoch contains multiple batches
        if len(preds) > 1:
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0).float()
        else:
            preds = preds[0]
            labels = labels[0].float()

        training_loss = criterion(preds, labels)
        training_loss.backward()
        optimizer.step()

        val_loss, val_label = val_phase_cv(model, model_idx, criterion, args,
                                           verbose=(epoch == args['train_epochs'] - 1))

        # save model with best validation loss
        if val_loss < best_val:
            torch.save(model.state_dict(), os.path.join(args['results_dir'],"model_" + str(model_idx[0])))
            best_val = val_loss

        if scheduler is not None:
            scheduler.step(val_loss)

        label = 'Benign' if val_label == 0 else 'COVID'
        print('Epoch {} Train loss is {}, Val loss is {} on a {} example'.
              format(epoch, training_loss.item(), val_loss.item(), label))
        gc.collect()

    print('Best val loss is {}'.format(best_val))
    return best_val
