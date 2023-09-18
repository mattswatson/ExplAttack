import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import argparse
import os
import tqdm
import pickle
import warnings

from utils import VisdomLinePlotter, AverageMeter
from XrayDataset import GazeXrayDataset

try:
    import opacus
    from train_target_model import get_opacus_info, convert_to_dp
except ImportError:
    warnings.warn('Opacus is not installed, you will not be able to use the --dp argument.')


def load_model(model_type, num_classes, model_file=None):
    if model_type == 'densenet':
        model = models.densenet121(pretrained=True)

        # Resize final layer to have the correct number of outputs
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise NotImplementedError('Model architecture {} is not supported!'.format(model_type))

    if model_file:
        model.load_state_dict(torch.load(model_file))

    model.train()

    return model


def load_data(gaze_data_path, cxr_data_path, generated_heatmaps_path, test_split=0.2, batch_size=32, num_workers=12,
              train_idx_path=None, test_idx_path=None):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    cxr_transforms = [transforms.Resize(224), transforms.Normalize(mean=mean, std=std)]

    dataset = GazeXrayDataset(gaze_data_path, cxr_data_path, cxr_transforms=cxr_transforms,
                              generated_heatmaps_path=generated_heatmaps_path)

    indices = list(range(len(dataset)))
    split = int(np.floor(test_split * len(dataset)))

    np.random.shuffle(indices)

    train_indices, test_indices = indices[split:], indices[:split]

    if train_idx_path:
        with open(train_idx_path, 'rb') as f:
            train_indices = pickle.load(f)

        print('------ Loaded train indices from {}'.format(train_idx_path))

    if test_idx_path:
        with open(test_idx_path, 'rb') as f:
            test_indices = pickle.load(f)

        print('------ Loaded test indices from {}'.format(test_idx_path))

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              drop_last=True)

    return train_loader, test_loader


def train(model, train_loader, criterion, optimizer, device):
    losses = AverageMeter()

    model.train()

    for inputs, _, labels in tqdm.tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        _, labels = torch.max(labels, 2)
        labels = torch.squeeze(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        losses.update(loss.data, len(labels))

    return losses.avg.cpu()


def test(model, test_loader, criterion, device):
    losses = AverageMeter()
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for inputs, _, labels in tqdm.tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            _, labels = torch.max(labels, 2)
            labels = torch.squeeze(labels)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            losses.update(loss.data, len(labels))
            correct += torch.sum(preds == labels).cpu().item()
            total += len(labels)

    return losses.avg.cpu(), (correct / total) * 100


def main():
    parser = argparse.ArgumentParser(description='Finetune a densenet model on the CXR EGD dataset')

    dataset_args = parser.add_argument_group("Dataset arguments")
    model_args = parser.add_argument_group("Model/training arguments")
    plot_args = parser.add_argument_group("Plotting arguments")

    dataset_args.add_argument('--cxr-data-path', type=str, default='/media/hdd/mimic-cxr-jpg',
                              help='Path to JPG CXR data')
    dataset_args.add_argument('--gaze-data-path', type=str, default='/media/hdd/mimic-cxr-eye',
                              help='Path the EGD data')
    dataset_args.add_argument('--generated-heatmaps-path', type=str, default=None,
                              help='Path to pre-generated heatmaps. If None, generate heatmaps at runtime')
    dataset_args.add_argument('--label', choices=['Normal', 'CHF', 'Pneumothorax', 'all'], default='all',
                              help='Label to predict')
    dataset_args.add_argument('--num-workers', type=int, default=12, help='Number of dataloader workers')
    dataset_args.add_argument('--train-idx-path', type=str, default=None)
    dataset_args.add_argument('--test-idx-path', type=str, default=None)

    model_args.add_argument('--model-type', choices=['densenet'], default='densenet',
                            help='Model architecture to use for classification')
    model_args.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for')
    model_args.add_argument('--batch-size', type=int, default=32, help='Batch size during training')
    model_args.add_argument('--lr', type=float, default=0.001, help='Learning rate during training')
    model_args.add_argument('--test-split', type=float, default=0.2, help='Proportion of dataset to use for testing')
    model_args.add_argument('--save-dir', type=str, default=None, help='Directory to save model checkpoints to')

    plot_args.add_argument('--visdom-server', type=str, default='localhost', help='URL to Visdom server')
    plot_args.add_argument('--visdom-port', type=int, default=8097, help='Visdom server port')
    plot_args.add_argument('--plot', type=str, default=None, help='Name of Visdom plot')

    parser.add_argument('--dp', action='store_true', help='Use differential privacy during training')
    parser.add_argument('--epsilon', type=float, default=50, help='Target epsilon for DP guarantee')
    parser.add_argument('--max-grad-norm', type=float, default=1.2, help='Max L2 norm of per-sample gradients')
    parser.add_argument('--delta', type=float, default=0.00001,
                        help='Target delta for DP guarantee. Should be < inverse of |training|')

    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    plotter = None
    if args.plot:
        plotter = VisdomLinePlotter(args.plot, server=args.visdom_server, port=args.visdom_port)

    if args.label == 'all':
        num_classes = 3
    else:
        num_classes = 1

    # Load model
    model = load_model(args.model_type, num_classes)
    model = model.to(device)

    # Load the data
    train_loader, test_loader = load_data(args.gaze_data_path, args.cxr_data_path, args.generated_heatmaps_path,
                                          test_split=args.test_split, batch_size=args.batch_size,
                                          num_workers=args.num_workers, train_idx_path=args.train_idx_path,
                                          test_idx_path=args.test_idx_path)

    optimiser = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model, optimiser, train_loader, privacy_engine = convert_to_dp(args, model, optimiser, train_loader)

    for epoch in tqdm.tqdm(range(args.epochs)):
        train_loss = train(model, train_loader, criterion, optimiser, device)
        test_loss, acc = test(model, test_loader, criterion, device)

        print('Epoch {}: [Train loss: {:.4f}] [Test loss: {:.4f} | Test acc: {:.4f}]'.format(epoch, train_loss,
                                                                                             test_loss, acc))

        if args.plot:
            plotter.plot('loss', 'train', 'Classification loss', epoch, train_loss)
            plotter.plot('loss', 'test', 'Classification Loss', epoch, test_loss)
            plotter.plot('acc', 'test', 'Classification Accuracy', epoch, acc)

        if args.save_dir:
            save_dir = os.path.join(args.save_dir, '{}-epochs{}-bs{}-lr{}-labels{}'.format(args.model_type,
                                                                                           args.epochs,
                                                                                           args.batch_size,
                                                                                           args.lr,
                                                                                           args.label))

            os.makedirs(save_dir, exist_ok=True)
            model_save_path = os.path.join(save_dir, 'model-checkpoint{}.pth'.format(epoch))
            optim_save_path = os.path.join(save_dir, 'optim-checkpoint{}.pth'.format(epoch))
            print('---------- Saving model to {}'.format(model_save_path))

            if args.dp:
                torch.save(model._module.state_dict(), model_save_path)
            else:
                torch.save(model.state_dict(), model_save_path)
            torch.save(optimiser, optim_save_path)


if __name__ == '__main__':
    main()