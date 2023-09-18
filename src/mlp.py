import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix

import os
from argparse import ArgumentParser
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils import AverageMeter, VisdomLinePlotter, membership_advantage
from SHAPDatasets import SHAPDataset


class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_1_size=412, hidden_2_size=512, dropout_prob=0.25):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_1_size)
        self.fc2 = nn.Linear(hidden_1_size, hidden_2_size)
        self.fc3 = nn.Linear(hidden_2_size, 1)
        self.droput = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))

        x = self.droput(x)
        x = self.relu(self.fc2(x))

        x = self.droput(x)

        x = self.fc3(x)

        output = self.sigmoid(x)
        return output


def train(args, model, device, train_loader, criterion, optimiser, epoch):
    model.train()
    losses = AverageMeter()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimiser.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss = loss.mean()

        loss.backward()
        optimiser.step()

        losses.update(loss.item(), len(target))

    print('Train Epoch {}: [Loss: {:.6f}]'.format(epoch, losses.avg))

    return losses.avg


def test(args, model, device, criterion, test_loader):
    model.eval()
    losses = AverageMeter()
    correct = 0
    total = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = criterion(output, target)
            losses.update(loss.item(), len(target))
            pred = torch.round(output)

            if args.verbose:
                print(target)
                print(output)
                print(pred)
                print('---------------')

            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(target)
            all_targets += target.cpu().tolist()
            all_preds += pred.cpu().tolist()

    conf_matrix = confusion_matrix(all_targets, all_preds)
    f, axes = plt.subplots(1, 1, figsize=(8, 8))
    conf_matrix_heatmap = sns.heatmap(conf_matrix, annot=True, fmt='d', ax=axes)

    m_adv = membership_advantage(conf_matrix)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Membership Advantage: {:.4f}\n'.format(
        losses.avg, correct, total, 100. * correct / total, m_adv))

    return losses.avg, correct / total, f


def main():
    parser = ArgumentParser(description='Train an MLP for membership inference on explanations from a FL model')
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--test-batch-size', type=int, default=128)

    parser.add_argument('--hidden1-size', type=int, default=1024)
    parser.add_argument('--hidden2-size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.25)

    parser.add_argument('--epochs', type=int, default=14)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--weighted', action='store_true', help='Use weighted sampling')
    parser.add_argument('--use-outputs', action='store_true', help='Use outputs if included in data')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no-cuda', action='store_true')

    parser.add_argument('--plot', type=str, default=None)
    parser.add_argument('--visdom-server', type=str, default='localhost', help='URL to Visdom server')
    parser.add_argument('--visdom-port', type=int, default=8097, help='Visdom server port')

    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--save-path', type=str, default=None)

    args = parser.parse_args()

    print('-------- Train MLP: {} epochs, hidden1: {}, hidden2: {}, dropout: {}'.format(args.epochs, args.hidden1_size,
                                                                                        args.hidden2_size,
                                                                                        args.dropout))

    cuda_available = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda_available else {}

    torch.manual_seed(args.seed)

    if args.plot is not None:
        plotter = VisdomLinePlotter(args.plot, server=args.visdom_server, port=args.visdom_port)
    else:
        plotter = None

    dataset = SHAPDataset(args.data_path, remove_output=not args.use_outputs)
    train_size = int(0.9 * len(dataset))
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    input_size = len(dataset[0][0])

    # We want to fit the scaler to the train set, then apply the same scaling across the whole dataset
    scaler = dataset.get_scaler(idx=train_indices)
    dataset.apply_scaler(scaler)

    if args.weighted:
        train_sample_count = np.array([len(np.where(dataset.targets[train_indices] == t)[0]) for t in [0, 1]])
        weights = 1. / train_sample_count
        sample_weights = [0 if i not in train_indices else weights[dataset.targets[i]] for i in range(len(dataset))]
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))
    else:
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)

    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler, **kwargs)

    model = MLP(input_size, args.hidden1_size, args.hidden2_size, args.dropout)
    model = model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    for epoch in range(args.epochs):
        train_loss = train(args, model, device, train_loader, criterion, optimiser, epoch)
        test_loss, acc, conf_matrix_heatmap = test(args, model, device, criterion, test_loader)

        if plotter is not None:
            plotter.plot('loss', 'train', 'loss', epoch, train_loss)
            plotter.plot('loss', 'test', 'loss', epoch, test_loss)

            plotter.plot('acc', 'test', 'acc', epoch, acc)
            plotter.plot_matplotlib('Confusion Matrix Epoch {}'.format(epoch), conf_matrix_heatmap)

    if args.save_path is not None:
        torch.save(model.state_dict(), os.path.join(args.save_path,
                                                    'epochs{}_hidden1size{}_hidden2size{}_dropout_mlp.pth'.format(
                                                        args.epochs, args.hidden1_size, args.hidden2_size, args.dropout
                                                    )))
        print('---------- Saved model to {}'.format(os.path.join(args.save_path,
                                                    'epochs{}_hidden1size{}_hidden2size{}_dropout_mlp.pth'.format(
                                                        args.epochs, args.hidden1_size, args.hidden2_size, args.dropout
                                                    ))))


if __name__ == '__main__':
    main()
