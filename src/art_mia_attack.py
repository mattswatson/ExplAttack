import pickle

import argparse
import random
from tqdm import tqdm
from functools import reduce
import warnings

import torch
from torchvision import datasets, transforms
from torchvision import models as pt_models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from sklearn import metrics

import numpy as np

from train_target_model import load_model
from TabularDatasets import AdultDataset, CompasDataset, TexasDataset
from FEMNISTDataset import create_leaf_fldataset
from utils import membership_advantage
from explanation_ensemble import Ensemble, get_models
from models import SmallCNN, MLP
from cxr import load_model as load_densenet_model


from art.attacks.inference.membership_inference import ShadowModels
from art.attacks.inference.membership_inference import MembershipInferenceBlackBoxRuleBased, MembershipInferenceBlackBox
from art.estimators.classification import PyTorchClassifier
from art.utils import to_categorical

try:
    from opacus.validators import ModuleValidator
except ImportError:
    run_validation = False
    warnings.warn("Running old Opacus version, model will not be checked for compatibility.")


def get_dataset(dataset, data_path):
    if dataset == 'mnist':
        train_set = datasets.MNIST(data_path, train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

        test_set = datasets.MNIST(data_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))

        num_classes = 10
        num_features = 28 * 28
        input_shape = (1, 28, 28)
    elif dataset == 'cifar10':
        train_set = datasets.CIFAR10(data_path, train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ]))
        test_set = datasets.CIFAR10(data_path, train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

        num_classes = 10
        num_features = 3 * 32 * 32
        input_shape = (3, 32, 32)
    elif dataset == 'compas':
        train_set = CompasDataset(data_path, train=True)
        test_set = CompasDataset(data_path, normalise=False, train=False)
        test_set.apply_scaler(train_set.scaler)

        num_classes = 2
        num_features = 466
        input_shape = (466,)
    elif dataset == 'adult':
        train_set = AdultDataset(data_path, train=True)
        test_set = AdultDataset(data_path, normalise=False, train=False)
        test_set.apply_scaler(train_set.scaler)

        num_classes = 2
        num_features = 205
        input_shape = (205,)
    elif dataset == 'texas':
        train_set = TexasDataset(data_path, train=True)
        test_set = TexasDataset(data_path, normalise=False, train=False)
        test_set.apply_scaler(train_set.scaler)

        num_classes = 100
        num_features = 252
        input_shape = (252,)
    elif dataset == 'femnist':
        train_set, _ = create_leaf_fldataset('femnist', data_path, split='train', transforms=transforms.Compose([
            transforms.Normalize((0.9641,), (0.1592,))
        ]))

        test_set, _ = create_leaf_fldataset('femnist', data_path, split='test', transforms=transforms.Compose([
                transforms.Normalize((0.9641,), (0.1592,))
            ]))

        num_classes = 62
        num_features = 28 * 28
        input_shape = (1, 28, 28)
    elif dataset == 'leaf':
        train_set, _ = create_leaf_fldataset('synthetic', data_path, split='train')

        test_set, _ = create_leaf_fldataset('synthetic', data_path, split='test')

        num_classes = 12
        num_features = len(train_set[0][0])
        input_shape = (num_features,)
    elif dataset == 'nature':
        train_set = datasets.INaturalist(data_path, version='2021_train_mini', target_type='kingdom',
                                         transform=transforms.Compose([
                                             transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                         ]))

        test_set = datasets.INaturalist(data_path, version='2021_valid', target_type='kingdom',
                                         transform=transforms.Compose([
                                             transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                         ]))

        num_classes = 10
        num_features = 3 * 224 * 224
        input_shape = (3, 224, 224)
    else:
        raise AttributeError('Dataset {} is not supported'.format(dataset))

    return train_set, test_set, num_classes, num_features, input_shape


def torch_dataset_to_sklearn(dataset, max_num_samples=None):
    """
    Split a PyTorch dataset into x and y tensors
    :param dataset:
    :return:
    """

    try:
        y_shape = dataset[0][1].shape
    except AttributeError:
        y_shape = (1,)

    num_samples = len(dataset)
    if max_num_samples:
        num_samples = min(max_num_samples, num_samples)

    print(dataset[0][0].shape)
    x = torch.zeros((num_samples, *dataset[0][0].shape))
    y = torch.zeros((num_samples, *y_shape))

    for i in range(num_samples):
        x_sample = dataset[i][0]
        y_sample = dataset[i][1]

        x[i] = x_sample
        y[i] = y_sample

    del dataset, x_sample, y_sample

    return x, y


def calc_precision_recall(predicted, actual, positive_value=1):
    # From https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/attack_membership_inference.ipynb
    score = 0  # both predicted and actual are positive
    num_positive_predicted = 0  # predicted positive
    num_positive_actual = 0  # actual positive
    for i in range(len(predicted)):
        if predicted[i] == positive_value:
            num_positive_predicted += 1
        if actual[i] == positive_value:
            num_positive_actual += 1
        if predicted[i] == actual[i]:
            if predicted[i] == positive_value:
                score += 1

    if num_positive_predicted == 0:
        precision = 1
    else:
        precision = score / num_positive_predicted  # the fraction of predicted “Yes” responses that are correct
    if num_positive_actual == 0:
        recall = 1
    else:
        recall = score / num_positive_actual  # the fraction of “Yes” responses that are predicted correctly

    return precision, recall


def main():
    parser = argparse.ArgumentParser(description='Run Membership Inference Attacks on pre-trained FL models using the'
                                                 'Adversarial Robustness Toolkit')

    parser.add_argument('--model-path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--model', choices=['large-cnn', 'small-cnn', 'mlp', 'syft-cnn', 'densenet'],
                        default='large-cnn', help='Model to train (default: large-cnn)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--ee', action='store_true', help='Load and attack an explanation ensemble model')
    parser.add_argument('--dp', action='store_true')


    parser.add_argument('--dataset', choices=['mnist', 'femnist', 'leaf', 'cifar10', 'compas', 'adult',
                                              'texas', 'nature'], default='mnist')
    parser.add_argument('--data-path', type=str, default='../data', help='Path to download MNIST data to')
    parser.add_argument('--data-idx-path', type=str, default=None,
                        help='Path to indices of data to use for training/test split')

    parser.add_argument('--attack', choices=['rule', 'bbox', 'shadow'], default='rule', help='Type of attack to use')
    parser.add_argument('--num-shadow-models', type=int, default=5, help='Number of shadow models to train if using')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for shadow models if using')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Load data
    train_set, test_set, num_classes, num_features, input_shape = get_dataset(args.dataset, args.data_path)

    # ART expects data in the form x_train, y_train, x_test, y_test
    x_train, y_train = torch_dataset_to_sklearn(train_set, max_num_samples=5000)
    x_test, y_test = torch_dataset_to_sklearn(test_set, max_num_samples=500)

    if args.attack == 'shadow':
        x_shadow, y_shadow = x_test.cpu().numpy(), y_test.cpu().numpy()

        if not args.data_idx_path:
            raise ValueError('argument data_idx_path must be passed if using a shadow dataset!')
    else:
        x_shadow, y_shadow = None, None

    # If we're given data_idx_path, then split train/test using that not default dataset split
    if args.data_idx_path:
        with open(args.data_idx_path, 'rb') as f:
            train_indices = pickle.load(f)

        print('============ Loaded train indices from {}\n{} training samples'.format(args.data_idx_path,
                                                                                      len(train_indices)))

        test_indices = [i for i in range(len(x_train)) if i not in train_indices]

        train_indices = [x % 5000 for x in train_indices][:5000]
        test_indices= [x % 500 for x in test_indices][:500]

        x_test, y_test = x_train[test_indices], y_train[test_indices]
        x_train, y_train = x_train[train_indices], y_train[train_indices]

    del train_set, test_set

    # Labels should be of the shape (nb_samples,) or (nb_samples, nb_classes)
    try:
        if y_test.shape[1] == 1:
            y_test = y_test.reshape(-1)
        elif y_test.shape[1] != num_classes:
            y_test = y_test.reshape(-1, num_classes)

        if y_train.shape[1] == 1:
            y_train = y_train.reshape(-1)
        elif y_train.shape[1] != num_classes:
            y_train = y_train.reshape(-1, num_classes)
    except IndexError:
        pass

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

        optimiser = optim.Adam(params, lr=0.0001)
    else:
        if 'syft' in args.model:
            if args.model == 'syft-cnn':
                model = SmallCNN(num_classes=num_classes)
                model.load_state_dict(torch.load(args.model_path))
        elif args.model == 'densenet':
            model = load_densenet_model('densenet', num_classes)

            if args.dp:
                errors = ModuleValidator.validate(model, strict=False)

                if len(errors) != 0:
                    # Let Opacus try and fix our model
                    print('WARNING: Model {} is not able to be used with differential privacy (most likely due to '
                          'normalisation. Attempting to fix.')

                    model = ModuleValidator.fix(model)

            model.load_state_dict(torch.load(args.model_path))
            model = model.cuda()
        else:
            model = load_model(args.model_path, args.model, num_classes, num_features, True, 'cuda')


        optimiser = optim.Adam(model.parameters(), lr=0.001)

    print('============ Loaded model from {}'.format(args.model_path))

    # Wrap model in ART wrapper
    loss_fn = nn.NLLLoss()

    if args.model == 'mlp':
        # Model takes a flat input
        input_shape = (reduce(lambda x, y: x * y, input_shape),)

        # Need to make sure data is flat too
        x_train = x_train.reshape(-1, *input_shape)
        x_test = x_test.reshape(-1, *input_shape)

    art_model = PyTorchClassifier(model, loss=loss_fn, optimizer=optimiser, input_shape=input_shape,
                                  nb_classes=num_classes)

    if args.attack == 'rule':
        attack = MembershipInferenceBlackBoxRuleBased(art_model)

        if args.dataset == 'texas' or args.dataset == 'leaf':
            y_train = to_categorical(y_train, nb_classes=num_classes)
            y_test = to_categorical(y_test, nb_classes=num_classes)
    elif args.attack == 'bbox':
        attack = MembershipInferenceBlackBox(art_model, attack_model_type='nn')
        attack_train_size = int(0.7 * len(x_train))
        attack_test_size = int(0.7 * len(x_test))

        # Fix for the texas dataset, which doesn't work properly due to a bug in ARTS code
        if args.dataset == 'texas' or args.dataset == 'leaf':
            y_train = to_categorical(y_train, nb_classes=num_classes)
            y_test = to_categorical(y_test, nb_classes=num_classes)

        attack.fit(x_train[:attack_train_size], y_train[:attack_train_size], x_test[attack_test_size:],
                   y_test[attack_test_size:])

        # Set x_train, x_test etc. to our test set
        x_train, y_train = x_train[attack_train_size:], y_train[attack_train_size:]
        x_test, y_test = x_test[attack_test_size:], y_test[attack_test_size:]
    elif args.attack == 'shadow':
        if args.dataset == 'texas' or args.dataset == 'leaf':
            y_train = to_categorical(y_train, nb_classes=num_classes)
            y_test = to_categorical(y_test, nb_classes=num_classes)

        attack_train_size = int(0.7 * len(x_train))
        shadow_models = ShadowModels(art_model, num_shadow_models=args.num_shadow_models)
        shadow_dataset = shadow_models.generate_shadow_dataset(x_shadow, y_shadow)

        (member_x, member_y, member_predictions), (nonmember_x, nonmember_y, nonmember_predictions) = shadow_dataset

        # Shadow model accuracy
        for i, model in enumerate(shadow_models.get_shadow_models()):
            outputs = model.predict(x_train[attack_train_size:], batch_size=args.batch_size)
            preds = outputs.argmax(axis=1)
            acc = np.equal(preds, y_train[attack_train_size:].reshape(preds.shape)).sum() / len(preds)

            print('Shadow model {} accuracy: {:.3f}'.format(i, acc))

        attack = MembershipInferenceBlackBox(art_model, attack_model_type='nn')
        attack.fit(member_x, member_y, nonmember_x, nonmember_y, member_predictions, nonmember_predictions)

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
