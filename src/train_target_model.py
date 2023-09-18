import argparse
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import pickle
import pickletools
import random
import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from packaging import version

try:
    import opacus
except ImportError:
    warnings.warn('Opacus is not installed, you will not be able to use the --dp argument.')

from utils import AverageMeter, VisdomLinePlotter
from FEMNISTDataset import create_leaf_fldataset
from TabularDatasets import CompasDataset, AdultDataset, TexasDataset
from cxr import load_model as load_densenet_model


class Net(nn.Module):
    def __init__(self, num_classes=10, dropout_prob1=0.25, dropout_prob2=0.25):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(dropout_prob1)
        self.dropout2 = nn.Dropout2d(dropout_prob2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(774400, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = self.softmax(x)
        return output


class SmallNet(nn.Module):
    def __init__(self, num_classes=10, dropout_prob1=0.25, softmax=True):
        super(SmallNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.dropout1 = nn.Dropout2d(dropout_prob1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(5408, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        self.do_softmax = softmax

    def forward(self, x, features=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x1 = torch.flatten(x, 1)
        x = self.fc1(x1)

        if self.do_softmax:
            return self.softmax(x)

        if not features:
            return x

        return x, x1

class MLP(nn.Module):
    def __init__(self, num_features=784, hidden_1_size=412, hidden_2_size=512, num_classes=10, dropout_prob=0.25):
        super(MLP, self).__init__()
        self.num_features = num_features

        self.fc1 = nn.Linear(num_features, hidden_1_size)
        self.fc2 = nn.Linear(hidden_1_size, hidden_2_size)
        self.fc3 = nn.Linear(hidden_2_size, num_classes)
        self.droput = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Flatten the input if needed
        x = x.view(-1, self.num_features)

        x = self.relu(self.fc1(x))

        x = self.droput(x)
        x = self.relu(self.fc2(x))

        x = self.droput(x)

        x = self.fc3(x)

        output = self.softmax(x)
        return output


def get_opacus_info(args, optimiser, privacy_engine):
    """
    Get the epsilon and delta for differential privacy during training. Support multiple Opacus versions
    :param args:
    :param optimiser:
    :param privacy_engine:
    :return: string
    """

    output_string = ' '
    if args.dp:
        if version.parse(opacus.__version__) > version.parse('1.0'):
            output_string += '[Epsilon: {:.6f}, Delta: {:.6f}'.format(privacy_engine.get_epsilon(args.delta),
                                                                      args.delta)
        else:
            output_string += '[Epsilon: {:.6f}, Delta: {:.6f}'\
                .format(optimiser.privacy_engine.get_privacy_spent(args.delta), args.delta)

    return output_string


def train(args, model, device, train_loader, optimiser, loss_fn, epoch, verbose=False, privacy_engine=None):
    losses = AverageMeter()

    model.train()
    len_train = len(train_loader.dataset) if len(train_loader.dataset) <= len(train_loader.sampler) \
        else len(train_loader.sampler)

    for batch_idx, (data, target) in enumerate(train_loader):
        # Make sure we have the correct data type
        data = data.type(torch.FloatTensor)
        data, target = data.to(device), target.to(device)

        optimiser.zero_grad()
        output = model(data)

        if args.model == 'densenet':
            output = F.log_softmax(output)

        loss = loss_fn(output, target)
        loss.backward()
        optimiser.step()

        losses.update(loss.item(), len(target))

        if batch_idx % args.log_interval == 0 and verbose:
            output_string = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len_train,
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item())

            output_string += get_opacus_info(args, optimiser, privacy_engine)

            print(output_string)

    return losses.avg


def test(model, device, test_loader, loss_fn, verbose=False):
    model.eval()
    test_loss = 0
    correct = 0
    num_samples = 0

    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            # Make sure we have the correct data type
            data = data.type(torch.FloatTensor)

            data, target = data.to(device), target.to(device)

            output = F.log_softmax(model(data))
            test_loss += loss_fn(output, target).item()  # sum up batch loss

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()
            num_samples += len(data)

            y_true += list(target.flatten().detach().cpu().numpy())
            y_pred += list(pred.flatten().detach().cpu().numpy())


    test_loss /= num_samples

    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                     num_samples,
                                                                                     100. * correct /
                                                                                    num_samples))

    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    print(f"Macro F1: {f1}, precision: {precision}, recall: {recall}\n")

    acc = 100. * correct / num_samples
    return test_loss, acc, f1, precision, recall


def load_model(path, model_type, num_classes, num_features, state_dict, device):
    if state_dict:
        if model_type == 'large-cnn':
            model = Net(num_classes=num_classes).to(device)
        elif model_type == 'small-cnn':
            model = SmallNet(num_classes=num_classes).to(device)
        elif model_type == 'mlp':
            model = MLP(num_features=num_features, num_classes=num_classes).to(device)
        else:
            raise NotImplementedError()

        model.load_state_dict(torch.load(path))
    else:
        model = torch.load(path)
        model = model.to(device)

    return model


def load_test_train_split(split_path):
    with open(split_path, 'rb') as f:
        train_indices = pickle.load(f)

    print('============ Loaded train indices from {}\n{} training samples'.format(split_path, len(train_indices)))

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

    return train_sampler


def get_dataset(args, kwargs):
    if args.dataset == 'mnist':
        train_set = datasets.MNIST(args.data_path, train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.test_batch_size, shuffle=args.shuffle, **kwargs)

        num_classes = 10
        num_features = 28 * 28
    elif args.dataset == 'cifar10':
        train_set = datasets.CIFAR10(args.data_path, train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ]))
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_path, train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])),
            batch_size=args.test_batch_size, shuffle=args.shuffle, **kwargs
        )

        num_classes = 10
        num_features = 3 * 32 * 32
    elif args.dataset == 'compas':
        train_set = CompasDataset(args.data_path, train=True)
        test_set = CompasDataset(args.data_path, normalise=False, train=False)
        test_set.apply_scaler(train_set.scaler)
        test_loader = torch.utils.data.DataLoader(test_set,
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

        num_classes = 2
        num_features = 466

        if args.model != 'mlp':
            raise NotImplementedError('Cannot use a CNN with flat data (using dataset {})!'.format(args.dataset))
    elif args.dataset == 'adult':
        train_set = AdultDataset(args.data_path, normalise=False, train=True)
        test_set = AdultDataset(args.data_path, normalise=False, train=False)
        #test_set.apply_scaler(train_set.scaler)
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=args.test_batch_size, shuffle=True, **kwargs)

        num_classes = 2
        num_features = 205

        if args.model != 'mlp':
            raise NotImplementedError('Cannot use a CNN with flat data (using dataset {})!'.format(args.dataset))
    elif args.dataset == 'texas':
        train_set = TexasDataset(args.data_path, train=True)
        test_set = TexasDataset(args.data_path, normalise=False, train=False)
        test_set.apply_scaler(train_set.scaler)
        test_loader = torch.utils.data.DataLoader(test_set,
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

        num_classes = 100
        num_features = 252

        if args.model != 'mlp':
            raise NotImplementedError('Cannot use a CNN with flat data (using dataset {})!'.format(args.dataset))
    elif args.dataset == 'femnist':
        train_set, _ = create_leaf_fldataset('femnist', args.data_path, split='train', transforms=transforms.Compose([
            transforms.Normalize((0.9641,), (0.1592,))
        ]))

        test_loader = torch.utils.data.DataLoader(
            create_leaf_fldataset('femnist', args.data_path, split='test', transforms=transforms.Compose([
                transforms.Normalize((0.9641,), (0.1592,))
            ]))[0],
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

        num_classes = 62
        num_features = 28 * 28
    elif args.dataset == 'leaf':
        train_set, _ = create_leaf_fldataset('synthetic', args.data_path, split='train')

        test_loader = torch.utils.data.DataLoader(
            create_leaf_fldataset('synthetic', args.data_path, split='test')[0],
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

        num_classes = 12
        num_features = len(train_set[0][0])

        if args.model != 'mlp':
            raise NotImplementedError('Cannot use a CNN with flat data (using dataset {})!'.format(args.dataset))
    elif args.dataset == 'nature':
        print('Loading iNature data...')
        train_set = datasets.INaturalist(args.data_path, version='2021_train_mini', target_type='kingdom',
                                   transform=transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                   ]))

        test_loader = torch.utils.data.DataLoader(
            datasets.INaturalist(args.data_path, version='2021_valid', target_type='kingdom',
                                 transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])),
            batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)

        num_classes = 10
        num_features = 3 * 224 * 224
    else:
        raise AttributeError('Dataset {} is not supported'.format(args.dataset))

    if args.data_idx_path is not None:
        train_sampler = load_test_train_split(args.data_idx_path)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                                                   sampler=train_sampler, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle,
                                                   **kwargs)

    print('Data Loaded!')
    return num_classes, num_features, test_loader, train_loader


def convert_to_dp(args, model, optimiser, train_loader):
    if args.dp:

        # If we're running an up to date version of Opacus, we can validate our model first
        # (When running in an old version of PyTorch with PySyft, this won't work, so add error handling)
        run_validation = True
        try:
            from opacus.validators import ModuleValidator
        except ImportError:
            run_validation = False
            warnings.warn("Running old Opacus version, model will not be checked for compatibility.")

        if run_validation:
            # Double check our model can be used with differential privacy
            errors = ModuleValidator.validate(model, strict=False)

            if len(errors) != 0:
                # Let Opacus try and fix our model
                print('WARNING: Model {} is not able to be used with differential privacy (most likely due to '
                      'normalisation. Attempting to fix.')

                model = ModuleValidator.fix(model)
                optimiser = optim.Adam(model.parameters(), lr=0.001)

                if len(ModuleValidator.validate(model, strict=False)) != 0:
                    raise Exception('Could not fix model for DP use. Try using a different architecture.')


        # Opacus changed its API with version 1.0, but when using PySyft we're using opacus==0.15.0 due to the need
        # for torch==1.4.0. Maybe not the best solution, but we need to handle this somehow
        if version.parse(opacus.__version__) > version.parse('1.0'):
            privacy_engine = opacus.PrivacyEngine()

            model, optimiser, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimiser,
                data_loader=train_loader,
                epochs=args.epochs,
                target_epsilon=args.epsilon,
                target_delta=args.delta,
                max_grad_norm=args.max_grad_norm
            )

            print('------- Using Differential Privacy with sigma={} and C={}'.format(optimiser.noise_multiplier,
                                                                                     args.max_grad_norm))
        else:
            # Add support for Syft
            try:
                len_trainset = len(train_loader.dataset)
            except AttributeError:
                len_trainset = len(train_loader.federated_dataset)
                
            privacy_engine = opacus.PrivacyEngine(
                model,
                epochs=args.epochs,
                target_epsilon=args.epsilon,
                target_delta=args.delta,
                max_grad_norm=args.max_grad_norm,
                batch_size=args.batch_size,
                sample_size=len_trainset
            )
            privacy_engine.attach(optimiser)

            print('------- Using Differential Privacy with sigma={} and C={}'
                  .format(optimiser.privacy_engine.noise_multiplier, args.max_grad_norm))

    else:
        privacy_engine = None

    return model, optimiser, train_loader, privacy_engine


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Train PyTorch model')
    parser.add_argument('--dataset', choices=['mnist', 'femnist', 'leaf', 'cifar10', 'compas', 'adult',
                                              'texas', 'nature'], default='mnist')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--data-path', type=str, default='../data', help='Path to download MNIST data to')
    parser.add_argument('--data-idx-path', type=str, default=None,
                        help='Path to indices of data to use for training/test split')

    parser.add_argument('--epochs', type=int, default=14, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.25, help='Dropout probability to use (default: 0.25)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', type=str, default=None, help='Path to save model to (default: None)')
    parser.add_argument('--plot', type=str, default=None, help='Name of Visdom plot (default: None)')
    parser.add_argument('--visdom-server', type=str, default='localhost', help='URL to Visdom server')
    parser.add_argument('--visdom-port', type=int, default=8097, help='Visdom server port')

    parser.add_argument('--model', choices=['large-cnn', 'small-cnn', 'mlp', 'densenet'], default='large-cnn',
                        help='Model to train (default: large-cnn)')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle dataset for training')

    parser.add_argument('--dp', action='store_true', help='Use differential privacy during training')
    parser.add_argument('--epsilon', type=float, default=50, help='Target epsilon for DP guarantee')
    parser.add_argument('--max-grad-norm', type=float, default=1.2, help='Max L2 norm of per-sample gradients')
    parser.add_argument('--delta', type=float, default=0.00001,
                        help='Target delta for DP guarantee. Should be < inverse of |training|')

    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if args.plot is not None:
        plotter = VisdomLinePlotter(args.plot, server=args.visdom_server, port=args.visdom_port)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    num_classes, num_features, test_loader, train_loader = get_dataset(args, kwargs)

    loss_fn = nn.NLLLoss()
    resnet = False
    if args.model == 'large-cnn':
        model = Net(dropout_prob1=args.dropout, dropout_prob2=args.dropout, num_classes=num_classes).to(device)
    elif args.model == 'small-cnn':
        model = SmallNet(dropout_prob1=args.dropout, num_classes=num_classes).to(device)
    elif args.model == 'mlp':
        model = MLP(num_features=num_features, dropout_prob=args.dropout, num_classes=num_classes).to(device)
    elif args.model == 'densenet':
        model = load_densenet_model('densenet', num_classes)
        model = model.to(device)
    else:
        raise NotImplementedError()

    num_params = sum(p.numel() for p in model.parameters())
    print('Model arch {} has {} parameters'.format(args.model, num_params))
    torch.manual_seed(args.seed)

    optimiser = optim.Adam(model.parameters(), lr=args.lr)

    model, optimiser, train_loader, privacy_engine = convert_to_dp(args, model, optimiser, train_loader)

    for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
        train_loss = train(args, model, device, train_loader, optimiser, loss_fn, epoch, verbose=args.verbose,
                           privacy_engine=privacy_engine)
        test_loss, acc, f1, precision, recall = test(model, device, test_loader, loss_fn, verbose=args.verbose)

        if args.plot is not None:
            plotter.plot('loss', 'train', 'loss', epoch, train_loss)
            plotter.plot('loss', 'test', 'loss', epoch, test_loss)
            plotter.plot('accuracy', 'test', 'acc', epoch, acc)
            plotter.plot('f1', 'test', 'acc', epoch, f1)
            plotter.plot('precision', 'test', 'precision', epoch, acc)
            plotter.plot('recall', 'test', 'acc', epoch, recall)

    if args.save_model is not None:
        if args.dp:
            torch.save(model._module.state_dict(), args.save_model)
        else:
            torch.save(model.state_dict(), args.save_model)


if __name__ == '__main__':
    main()