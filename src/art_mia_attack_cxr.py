import os, sys, pickle

import argparse
import random
from tqdm import tqdm
from functools import reduce

import torch
from torchvision import datasets, transforms
from torchvision import models as pt_models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from sklearn import metrics

import numpy as np

from TabularDatasets import AdultDataset, CompasDataset, TexasDataset
from FEMNISTDataset import create_leaf_fldataset
from utils import membership_advantage
from explanation_ensemble_cxr import Ensemble, get_models
from models import SmallCNN, MLP

from art.attacks.inference.membership_inference import ShadowModels
from art.attacks.inference.membership_inference import MembershipInferenceBlackBoxRuleBased, MembershipInferenceBlackBox
from art.estimators.classification import PyTorchClassifier
from art.utils import to_categorical
from art_mia_attack import calc_precision_recall
from cxr import load_data, load_model
from XrayDataset import GazeXrayDataset

import warnings

try:
    from opacus.validators import ModuleValidator
except ImportError:
    run_validation = False
    warnings.warn("Running old Opacus version, model will not be checked for compatibility.")


def torch_dataset_to_sklearn(dataset):
    """
    Split a PyTorch dataset into x and y tensors
    :param dataset:
    :return:
    """

    try:
        y_shape = dataset[0][2].shape
    except AttributeError:
        y_shape = (1,)

    x = torch.zeros((len(dataset), *dataset[0][0].shape))
    y = torch.zeros((len(dataset), *y_shape))

    for i in range(len(dataset)):
        x_sample = dataset[i][0]
        y_sample = dataset[i][2]

        x[i] = x_sample
        y[i] = torch.tensor(y_sample)

    del dataset, x_sample, y_sample

    return x, y


def main():
    parser = argparse.ArgumentParser(description='Run Membership Inference Attacks on pre-trained FL models using the'
                                                 'Adversarial Robustness Toolkit')

    parser.add_argument('--model-path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--model', choices=['densenet'], default='densenet',
                        help='Model to train (default: large-cnn)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--ee', action='store_true', help='Load and attack an explanation ensemble model')
    parser.add_argument('--dp', action='store_true')


    parser.add_argument('--dataset', choices=['egd'], default='egd')
    parser.add_argument('--train-idx-path', type=str, default=None,
                        help='Path to indices of data to use for training/test split')
    parser.add_argument('--test-idx-path', type=str, default=None,
                        help='Path to indices of data to use for training/test split')

    parser.add_argument('--attack', choices=['rule', 'bbox', 'shadow'], default='rule', help='Type of attack to use')
    parser.add_argument('--num-shadow-models', type=int, default=5, help='Number of shadow models to train if using')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for shadow models if using')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    num_classes = 3
    num_features = 267456

    args.model_type = args.model

    # Load data
    with open('/media/hdd/mimic-cxr-eye/x.pt', 'rb') as f:
        x = pickle.load(f)

    with open('/media/hdd/mimic-cxr-eye/y.pt', 'rb') as f:
        y = pickle.load(f)

    with open(args.train_idx_path, 'rb') as f:
        train_indices = pickle.load(f)

    with open(args.test_idx_path, 'rb') as f:
        test_indices = pickle.load(f)

    x_train = x[train_indices].numpy()
    y_train = y[train_indices].numpy()

    x_test = x[test_indices].numpy()
    y_test = y[test_indices].numpy()

    del x, y

    if args.attack == 'shadow':
        x_shadow, y_shadow = x_test, y_test
    else:
        x_shadow, y_shadow = None, None

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    # Labels should be of the shape (nb_samples,) or (nb_samples, nb_classes)
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    print(y_train.shape)
    print(y_test.shape)

    # Load the model
    if args.ee:
        # Add some args that are needed to load the model
        args.dropout = 0
        args.verbose = False

        models = get_models(args, train=False, as_ensemble=False, model_file=args.model_path)
        model = Ensemble(models)

        params = []
        for m in models:
            params += list(m.parameters())

        optimiser = optim.Adam(params, lr=0.001)
    else:
        model = load_model('densenet', num_classes)

        if args.dp:
            errors = ModuleValidator.validate(model, strict=False)

            if len(errors) != 0:
                # Let Opacus try and fix our model
                print('WARNING: Model {} is not able to be used with differential privacy (most likely due to '
                      'normalisation. Attempting to fix.')

                model = ModuleValidator.fix(model)

        model.load_state_dict(torch.load(args.model_path))

        optimiser = optim.Adam(model.parameters(), lr=0.01)

    print('============ Loaded model from {}'.format(args.model_path))

    # Wrap model in ART wrapper
    loss_fn = nn.NLLLoss()
    input_shape = (3, 224, 398)

    art_model = PyTorchClassifier(model, loss=loss_fn, optimizer=optimiser, input_shape=input_shape,
                                  nb_classes=num_classes)

    if args.attack == 'rule':
        attack = MembershipInferenceBlackBoxRuleBased(art_model)

        #y_train = to_categorical(y_train, nb_classes=num_classes)
        #y_test = to_categorical(y_test, nb_classes=num_classes)
    elif args.attack == 'bbox':
        attack = MembershipInferenceBlackBox(art_model, attack_model_type='nn')
        attack_train_size = int(0.7 * len(x_train))
        attack_test_size = int(0.7 * len(x_test))

        # Fix for the texas dataset, which doesn't work properly due to a bug in ARTS code
        #y_train = to_categorical(y_train, nb_classes=num_classes)
        #y_test = to_categorical(y_test, nb_classes=num_classes)

        attack.fit(x_train[:attack_train_size], y_train[:attack_train_size], x_test[attack_test_size:],
                   y_test[attack_test_size:])

        # Set x_train, x_test etc. to our test set
        x_train, y_train = x_train[attack_train_size:], y_train[attack_train_size:]
        x_test, y_test = x_test[attack_test_size:], y_test[attack_test_size:]
    elif args.attack == 'shadow':
        #y_train = to_categorical(y_train, nb_classes=num_classes)
        #y_test = to_categorical(y_test, nb_classes=num_classes)
        y_shadow = y_shadow.squeeze()

        attack_train_size = int(0.7 * len(x_train))
        shadow_models = ShadowModels(art_model, num_shadow_models=args.num_shadow_models)
        shadow_dataset = shadow_models.generate_shadow_dataset(x_shadow, y_shadow)

        (member_x, member_y, member_predictions), (nonmember_x, nonmember_y, nonmember_predictions) = shadow_dataset

        # Shadow model accuracy
        for i, model in enumerate(shadow_models.get_shadow_models()):
            outputs = model.predict(x_train[attack_train_size:], batch_size=args.batch_size)
            preds = outputs.argmax(axis=1)
            true_labels = y_train.argmax(axis=1)
            acc = np.equal(preds, true_labels[attack_train_size:].reshape(preds.shape)).sum() / len(preds)

            print('Shadow model {} accuracy: {:.3f}'.format(i, acc))

        attack = MembershipInferenceBlackBox(art_model, attack_model_type='nn')
        attack.fit(member_x, member_y, nonmember_x, nonmember_y, member_predictions, nonmember_predictions)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # Infer membership in train set
    inferred_train = attack.infer(x_train, y_train)
    inferred_test = attack.infer(x_test, y_test)

    # Check attack accuracy
    train_acc = np.sum(inferred_train) / len(inferred_train)
    test_acc = 1 - (np.sum(inferred_test) / len(inferred_test))

    attack_acc = (train_acc * len(inferred_train) + test_acc * len(inferred_test)) / (len(inferred_train) +
                                                                                      len(inferred_test))
    print('train accuracy:', train_acc)
    print('test accuracy:', test_acc)
    print('attack accuracy:', attack_acc)

    labels = np.concatenate((np.ones(len(inferred_train)), np.zeros(len(inferred_test))))
    preds = np.concatenate((inferred_train, inferred_test))

    precision, recall = calc_precision_recall(preds, labels)

    print('precision: {}\nrecall: {}'.format(precision, recall))

    confusion_matrix = metrics.confusion_matrix(labels, preds)
    print('membership advantage: {}'.format(membership_advantage(confusion_matrix)))


if __name__ == '__main__':
    main()
