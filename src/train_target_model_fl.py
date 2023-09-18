import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from sklearn.metrics import precision_score, recall_score, f1_score

import syft as sy

import os
from argparse import ArgumentParser
import pickle

from utils import AverageMeter, VisdomLinePlotter
from models import SmallCNN, MLP
from FEMNISTDataset import create_leaf_fldataset
from train_target_model import convert_to_dp, get_opacus_info, Net
from TabularDatasets import CompasDataset, AdultDataset, TexasDataset


def train(args, model, device, federated_train_loader, optimiser, epoch, privacy_engine=None):
    model.train()
    losses = AverageMeter()
    clients_saved = []

    for batch_idx, (data, target) in enumerate(federated_train_loader):
        if args.holdout_clients and data.location.id in args.holdout_clients:
            # Ignore this batch if we don't want to include data from this client
            continue

        # Send the model to the location of the data
        model.send(data.location)

        # Train as normal (this occurs on the remote worker!)
        data, target = data.to(device), target.to(device)
        optimiser.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimiser.step()

        # Get the model back from the remote worker
        model.get()

        # Get the loss from the remote worker so we can log it
        loss = loss.get()
        losses.update(loss.item(), len(target))

    output_string = 'Train Epoch {}: [Loss: {:.6f}]'.format(epoch, losses.avg)
    output_string += get_opacus_info(args, optimiser, privacy_engine)

    print(output_string)
    return losses.avg


def test(args, model, device, test_loader):
    # Testing is the same as it is done at the model location
    model.eval()
    losses = AverageMeter()
    correct = 0
    total = 0

    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = F.nll_loss(output, target)
            losses.update(loss, len(target))
            pred = output.argmax(1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(target)

            y_true += list(target.flatten())
            y_pred += list(pred.flatten())

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        losses.avg, correct, total, 100. * correct / total))

    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    print(f"Macro F1: {f1}, precision: {precision}, recall: {recall}\n")

    return losses.avg, 100. * correct / total, f1, precision, recall


def main():
    parser = ArgumentParser(description='Train federated learning')
    parser.add_argument('--dataset', choices=['mnist', 'femnist', 'cifar10', 'compas', 'texas', 'nature'],
                        default='mnist')
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--test-batch-size', type=int, default=128)

    parser.add_argument('--model', choices=['large-cnn', 'small-cnn', 'mlp'], default='cnn')

    parser.add_argument('--num-clients', type=int, default=2)
    parser.add_argument('--holdout-clients', type=str, default=None, nargs='*',
                        help='Holdout these clients from training')

    parser.add_argument('--epochs', type=int, default=14)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.5)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no-cuda', action='store_true')

    parser.add_argument('--dp', action='store_true', help='Use differential privacy during training')
    parser.add_argument('--epsilon', type=float, default=50, help='Target epsilon for DP guarantee')
    parser.add_argument('--max-grad-norm', type=float, default=1.2, help='Max L2 norm of per-sample gradients')
    parser.add_argument('--delta', type=float, default=0.00001,
                        help='Target delta for DP guarantee. Should be < inverse of |training|')

    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--plot', type=str, default=None, help='Name of Visdom plot (default: None)')

    args = parser.parse_args()

    print('-------- Train on {}, {} epochs, {} clients, holdout {}'.format(args.dataset, args.epochs, args.num_clients,
                                                                           args.holdout_clients))

    hook = sy.TorchHook(torch)

    plotter = None
    if args.plot is not None:
        plotter = VisdomLinePlotter(args.plot)

    cuda_available = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda_available else {}

    torch.manual_seed(args.seed)

    # Create virtual workers. For now, only two
    clients = [sy.VirtualWorker(hook, id='client{}'.format(i)) for i in range(args.num_clients)]

    # Load the data and split it between the two virtual workers
    if args.dataset == 'mnist':
        federated_train_loader = sy.FederatedDataLoader(
            datasets.MNIST(args.data_path, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])).federate(clients),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

        num_classes = 10
        num_features = 28 * 28
    elif args.dataset == 'cifar10':
        federated_train_loader = sy.FederatedDataLoader(
            datasets.CIFAR10(args.data_path, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_path, train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs
        )

        num_classes = 10
        num_features = 3 * 32 * 32
    elif args.dataset == 'compas':
        train_set = CompasDataset(args.data_path, train=True)
        federated_train_loader = sy.FederatedDataLoader(train_set,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        **kwargs)

        test_set = CompasDataset(args.data_path, normalise=False, train=False)
        test_set.apply_scaler(train_set.scaler)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

        num_classes = 2
        num_features = 466
    elif args.dataset == 'adult':
        train_set = AdultDataset(args.data_path, train=True)
        federated_train_loader = sy.FederatedDataLoader(train_set,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        **kwargs)

        test_set = AdultDataset(args.data_path, normalise=False, train=False)
        test_set.apply_scaler(train_set.scaler)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

        num_classes = 2
        num_features = 204
    elif args.dataset == 'texas':
        train_set = TexasDataset(args.data_path, train=True)
        federated_train_loader = sy.FederatedDataLoader(train_set,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        **kwargs)

        test_set = TexasDataset(args.data_path, normalise=False, train=False)
        test_set.apply_scaler(train_set.scaler)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

        num_classes = 100
        num_features = 252
    elif args.dataset == 'nature':
        train_set = datasets.INaturalist(args.data_path, version='2021_train', target_type='kingdom', download=True,
                                         transform=transforms.Compose([
                                             transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                         ]))
        federated_train_loader = sy.FederatedDataLoader(train_set,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        **kwargs)

        test_set = datasets.INaturalist(args.data_path, version='2021_valid', target_type='kingdom',
                                        transform=transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                        ]))
        test_set.apply_scaler(train_set.scaler)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

        num_classes = 10
        num_features = 3 * 224 * 224
    elif args.dataset == 'femnist':
        print('Using non-MNIST dataset, ignoring --num_clients and using 3597 clients')
        clients = [sy.VirtualWorker(hook, id='client{}'.format(i)) for i in range(3597)]
        train_dataset, _ = create_leaf_fldataset('femnist', args.data_path, split='train')

        federated_train_loader = sy.FederatedDataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            create_leaf_fldataset('femnist', args.data_path, split='test')[0],
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

        num_classes = 62
        num_features = 28 * 28
    else:
        raise NotImplementedError('Dataset {} not implemented!'.format(args.dataset))

    if args.model == 'small-cnn':
        model = SmallCNN(num_classes=num_classes)
    elif args.model == 'mlp':
        model = MLP(num_features=num_features, num_classes=num_classes)
    elif args.model == 'large-cnn':
        model = Net(num_classes=num_classes)
    else:
        raise NotImplementedError('Model architecture {} not yet implemented!'.format(args.model))

    model = model.to(device)

    optimiser = optim.SGD(model.parameters(), lr=args.lr)

    model, optimiser, train_loader, privacy_engine = convert_to_dp(args, model, optimiser, federated_train_loader)

    for epoch in range(args.epochs):
        train_loss = train(args, model, device, federated_train_loader, optimiser, epoch, privacy_engine)
        test_loss, acc, f1, precision, recall = test(args, model, device, test_loader)

        if args.plot is not None:
            plotter.plot('loss', 'train', 'loss', epoch, train_loss)
            plotter.plot('loss', 'test', 'loss', epoch, test_loss)
            plotter.plot('accuracy', 'test', 'acc', epoch, acc)
            plotter.plot('f1', 'test', 'acc', epoch, f1)
            plotter.plot('precision', 'test', 'precision', epoch, acc)
            plotter.plot('recall', 'test', 'acc', epoch, recall)

    if args.save_path is not None:
        holdout_clients = ''
        for c in args.holdout_clients:
            holdout_clients += c + '-'

        torch.save(model.state_dict(), os.path.join(args.save_path,
                                                    '{}-{}_epcohs{}_clients{}_holdout{}.pth'.format(args.dataset,
                                                                                                    args.model,
                                                                                                    args.epochs,
                                                                                                    args.num_clients,
                                                                                                    holdout_clients)))
        print('---------- Saved model to {}'.format(os.path.join(args.save_path,
                                                                 '{}-{}_epcohs{}_clients{}_holdout{}.pth'.format(
                                                                     args.dataset,
                                                                     args.model,
                                                                     args.epochs,
                                                                     args.num_clients,
                                                                     holdout_clients))))


if __name__ == '__main__':
    main()
