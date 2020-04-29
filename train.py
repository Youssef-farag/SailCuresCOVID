from __future__ import print_function, division
import pretrainedmodels
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import gc

import loops
from data_utils import *
from cross_validation import k_folds

random.seed(0)
torch.manual_seed(0)
torch.set_printoptions(precision=20)
np.set_printoptions(suppress=True)
np.random.seed(0)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for train_idx, val_idx in tqdm(k_folds(n_splits=70)):

        model = pretrainedmodels.__dict__['resnet18'](num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Linear(model.last_linear.in_features, 1)
        model.inplanes = 64
        model.conv1 = nn.Conv2d(1, model.inplanes, kernel_size=7, stride=2, padding=3,
                                bias=False)
        model.to(device)

        opt = optim.Adam(model.parameters(), lr=0.00001)
        criterion = nn.BCEWithLogitsLoss()

        train_loader = make_train_data_loader(indices=train_idx)
        val_loader = make_val_data_loader(indices=val_idx)

        # train model
        loops.train(model, train_loader, criterion, opt, device, epochs=25)
        # evaluate val
        loops.val_phase_cv(model, val_loader, device)
        # save model to disk
        torch.save(model.state_dict(), "./models/model_" + str(val_idx[0]))
        gc.collect()

    # evaluate the 70 models
    loops.eval_ensemble()
    loops.generate_predictions()


if __name__ == '__main__':
    main()