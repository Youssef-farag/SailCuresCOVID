from __future__ import print_function, division
from tqdm import tqdm
import gc
import os
import yaml
import random

import loops
from utils import *
from cross_validation import k_folds

random.seed(0)
torch.manual_seed(0)
torch.set_printoptions(precision=20)
np.set_printoptions(suppress=True)
np.random.seed(0)


def main(args):
    val_losses = []

    for train_idx, val_idx in tqdm(k_folds(n_splits=70)):

        train_loader = make_train_data_loader(train_idx)

        # train model
        val_loss = loops.train(train_loader, val_idx, args)

        val_losses.append(val_loss)
        gc.collect()

    # evaluate the 70 models
    loops.eval_ensemble(args)
    loops.generate_predictions(args)
    np.save(os.path.join(args['results_dir'], 'val_losses.npy'), val_losses)


if __name__ == '__main__':
    paths = [
        "./models-0.3-sch/config.yml",
        "./models-0.2/config.yml",
        "./models-0.5/config.yml",
        "./models-0.3/config.yml",
        "./models-0.5-sch/config.yml"
    ]
    for path in tqdm(paths):
        with open("./models-0.3-sch/config.yml", "r") as ymlfile:
            args = yaml.load(ymlfile)
        main(args)
