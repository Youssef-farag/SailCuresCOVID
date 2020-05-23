from __future__ import print_function, division
from tqdm import tqdm
import gc
import os
import yaml

import loops
from utils import *
from cross_validation import k_folds

torch.manual_seed(0)
np.random.seed(0)


def main(args):
    val_losses = []

    for train_idx, val_idx in tqdm(k_folds(n_splits=70)):

        if val_idx[0] < args['resume']:
            continue

        train_loader = make_train_data_loader(train_idx)

        # train model
        val_loss = loops.train(train_loader, val_idx, args)

        val_losses.append(val_loss)
        gc.collect()

    # evaluate the 70 models on their validation sets
    loops.eval_ensemble(args)

    # generate prediction csv and dictionary of final predictions
    loops.generate_predictions(args)

    # save val losses during training
    np.save(os.path.join(args['results_dir'], 'val_losses.npy'), val_losses)


if __name__ == '__main__':
    config_paths = [
        "./models-0.5/config.yml",
        "./models-0.3/config.yml",
        "./models-0.5-sch/config.yml"
    ]
    for config in tqdm(config_paths):
        with open(config, "r") as ymlfile:
            args = yaml.load(ymlfile)
        main(args)
