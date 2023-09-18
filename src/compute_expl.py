import os
import argparse
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler
import torch.nn.functional as F

import numpy as np
from collections import OrderedDict

from captum.attr import GradientShap, GuidedGradCam

# These are for our own baselines we've created
from torchvision import datasets, transforms

from models import SmallCNN
from train_target_model import load_test_train_split, Net, SmallNet, MLP
from FEMNISTDataset import create_leaf_fldataset
from TabularDatasets import CompasDataset, AdultDataset, TexasDataset
from cxr import load_data, load_model

import warnings

try:
    from opacus.validators import ModuleValidator
except ImportError:
    run_validation = False
    warnings.warn("Running old Opacus version, model will not be checked for compatibility.")


# -------------------- GradCAM ------------------

# -- Code modified from source: https://github.com/kazuto1011/grad-cam-pytorch
class _BaseWrapper(object):
    """
    Please modify forward() and backward() depending on your task.
    """
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def generate(self):
        raise NotImplementedError

    def forward(self, image):
        """
        Simple classification
        """
        self.model.zero_grad()
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return list(zip(*self.probs.sort(0, True)))  # element: (probability, index)


class GradCam(_BaseWrapper):
    def __init__(self, model, candidate_layers=[]):
        super(GradCam, self).__init__(model)
        self.fmap_pool = OrderedDict()
        self.grad_pool = OrderedDict()
        self.candidate_layers = candidate_layers

        def forward_hook(module, input, output):
            self.fmap_pool[id(module)] = output.detach()


        def backward_hook(module, grad_in, grad_out):
            self.grad_pool[id(module)] = grad_out[0].detach()

        for module in self.model.named_modules():
            if len(self.candidate_layers) == 0 or module[0] in self.candidate_layers:
                self.handlers.append(module[1].register_forward_hook(forward_hook))
                self.handlers.append(module[1].register_backward_hook(backward_hook))

    def find(self, pool, target_layer):
        # --- Query the right layer and return it's value.
        for key, value in pool.items():
            for module in self.model.named_modules():
                # print(module[0], id(module[1]), key)
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError(f"Invalid Layer Name: {target_layer}")

    def normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads ,2))) + 1e-5
        return grads /l2_norm

    def compute_grad_weights(self, grads):
        grads = self.normalize(grads)
        return F.adaptive_avg_pool2d(grads, 1)


    def generate(self, target_layer):
        fmaps = self.find(self.fmap_pool, target_layer)
        grads = self.find(self.grad_pool, target_layer)
        weights = self.compute_grad_weights(grads)

        gcam = (fmaps[0] * weights[0]).sum(dim=0)
        gcam = torch.clamp(gcam, min=0.0)

        gcam -= gcam.min()
        gcam /= gcam.max()
        return gcam


def compute_gradCAM(probs, labels, gcam, testing_labels, criterion, target_layer='encoder.blocks.6'):
    # --- one hot encode this:
    # one_hot = torch.zeros((labels.shape[0], labels.shape[1])).float()
    one_hot = torch.zeros((probs.shape[0], probs.shape[1])).float()
    max_int = torch.max(criterion(probs), 1)[1]

    if testing_labels:
        for i in range(one_hot.shape[0]):
            one_hot[i][max_int[i]] = 1.0

    else:
        for i in range(one_hot.shape[0]):
            one_hot[i][torch.max(labels, 0)[i]] = 1.0

    probs.backward(gradient=one_hot.cuda(), retain_graph=True)
    fmaps = gcam.find(gcam.fmap_pool, target_layer)
    grads = gcam.find(gcam.grad_pool, target_layer)

    weights = F.adaptive_avg_pool2d(grads, 1)
    gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
    gcam_out = F.relu(gcam)
    return probs, gcam_out, one_hot


def main():
    parser = argparse.ArgumentParser(description='Compute and save GradCAM or SHAP attributions for a given model')

    parser.add_argument('--model-path', type=str, help='Path to model checkpoint to load', required=True)
    parser.add_argument('--model-type', default='small-cnn', choices=['small-cnn', 'large-cnn', 'small-net', 'mlp',
                                                                      'densenet'])

    parser.add_argument('--data-idx-path', type=str, default=None,
                        help='Path to indices of data to use for training/test split')

    parser.add_argument('--expl-type', choices=['gradcam', 'shap', 'c-gradcam'], default='gradcam',
                        help='Explanation method to use')
    parser.add_argument('--save-output', action='store_true', help='Also save the output of the model on each input')

    parser.add_argument('--dataset', choices=['mnist', 'femnist', 'leaf', 'compas', 'adult', 'texas', 'egd', 'nature'],
                        default='mnist')
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size to use')

    parser.add_argument('--dp', action='store_true')

    # CXR specific args
    parser.add_argument('--cxr-data-path', type=str, default='/media/hdd/mimic-cxr-jpg',
                              help='Path to JPG CXR data')
    parser.add_argument('--gaze-data-path', type=str, default='/media/hdd/mimic-cxr-eye',
                              help='Path the EGD data')
    parser.add_argument('--generated-heatmaps-path', type=str, default=None,
                              help='Path to pre-generated heatmaps. If None, generate heatmaps at runtime')

    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')

    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda_available else {}

    if args.data_idx_path is not None:
        with open(args.data_idx_path, 'rb') as f:
            train_idx = pickle.load(f)
    else:
        train_idx = None

    # Load the data
    num_labels = 10
    if args.dataset == 'mnist':
        train_set = datasets.MNIST(args.data_path, train=True, download=True,
                                                               transform=transforms.Compose([
                                                                   transforms.ToTensor(),
                                                                   transforms.Normalize((0.1307,), (0.3081,))
                                                               ]))
        num_labels = 10
        num_features = 784
    elif args.dataset == 'femnist':
        train_set, _ = create_leaf_fldataset('femnist', args.data_path, split='train', transforms=transforms.Compose([
            transforms.Normalize((0.9641,), (0.1592,))
        ]))

        num_labels = 62
        num_features = 784
    elif args.dataset == 'compas':
        train_set = CompasDataset(args.data_path, train=True)

        num_labels = 2
        num_features = 466
    elif args.dataset == 'adult':
        train_set = AdultDataset(args.data_path, train=True)

        num_labels = 2
        num_features = 205
    elif args.dataset == 'texas':
        train_set = TexasDataset(args.data_path, train=True)

        num_labels = 100
        num_features = 252
    elif args.dataset == 'leaf':
        train_set, _ = create_leaf_fldataset('synthetic', args.data_path, split='train')

        num_labels = 12
        num_features = 80
    elif args.dataset == 'egd':
        train_loader, _ = load_data(args.gaze_data_path, args.cxr_data_path, args.generated_heatmaps_path,
                                              test_split=0.2, batch_size=1,
                                              num_workers=1, train_idx_path=args.data_idx_path,
                                              test_idx_path=args.data_idx_path)

        num_labels = 3
        num_features = 267456
    elif args.dataset == 'nature':
        train_set = datasets.INaturalist(args.data_path, version='2021_train_mini', target_type='kingdom',
                                         transform=transforms.Compose([
                                             transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                         ]))

        num_labels = 10
        num_features = 3 * 224 * 224
    else:
        raise NotImplementedError('Dataset {} not implemented!'.format(args.dataset))

    if args.dataset != 'egd':
        if args.data_idx_path is not None:
            train_sampler = load_test_train_split(args.data_idx_path)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, sampler=train_sampler,
                                                       **kwargs)
        else:
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, **kwargs)


    # Load the model
    if args.model_type == 'small-cnn':
        model = SmallCNN(num_classes=num_labels)
        model = model.to(device)

        candidate_layers = ['conv2']
        target_layer = 'conv2'

        if args.expl_type == 'c-gradcam':
            guided_gcam = GuidedGradCam(model, model.conv2)
    elif args.model_type == 'small-net':
        model = SmallNet(num_classes=num_labels)
        model = model.to(device)

        candidate_layers = ['conv1']
        target_layer = 'conv1'

        if args.expl_type == 'c-gradcam':
            guided_gcam = GuidedGradCam(model, model.conv1)
    elif args.model_type == 'large-cnn':
        model = Net(num_classes=num_labels)
        model = model.to(device)

        candidate_layers = ['conv2']
        target_layer = 'conv2'

        if args.expl_type == 'c-gradcam':
            guided_gcam = GuidedGradCam(model, model.conv2)
    elif args.model_type == 'mlp':
        model = MLP(num_features=num_features, num_classes=num_labels)
        model = model.to(device)

        candidate_layers = ['fc3']
        target_layer = 'fc3'

        if args.expl_type == 'c-gradcam':
            guided_gcam = GuidedGradCam(model, model.fc3)
    elif args.model_type == 'densenet':
        model = load_model(args.model_type, num_labels)

        candidate_layers = ['fc3']
        target_layer = 'fc3'

        if args.expl_type == 'c-gradcam':
            guided_gcam = GuidedGradCam(model, model.features.denseblock4.denselayer16)
    else:
        raise NotImplementedError("Model architecture {} is not supported!".format(args.model_type))

    np.random.seed(42)
    torch.manual_seed(42)

    if args.dp:
        errors = ModuleValidator.validate(model, strict=False)

        if len(errors) != 0:
            # Let Opacus try and fix our model
            print('WARNING: Model {} is not able to be used with differential privacy (most likely due to '
                  'normalisation. Attempting to fix.')

            model = ModuleValidator.fix(model)

    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()
    model.eval()

    if args.expl_type == 'shap':
        shap = GradientShap(model)

    gcam = GradCam(model=model, candidate_layers=candidate_layers)

    # Each entry is a tuple, (explanation, target)
    all_expl = []
    all_targets = []
    all_outputs = []
    j = -1

    with torch.no_grad():
        for data in tqdm(train_loader):
            if args.dataset == 'egd':
                images, _, labels = data
                _, labels = torch.max(labels, 2)
            else:
                images, labels = data

            images = images.cuda()
            labels = labels.cuda()

            if args.expl_type == 'gradcam':
                prob_activation = nn.Sigmoid()

                output = model(images)
                _, expl_out, one_hot = compute_gradCAM(output, labels, gcam, False, prob_activation, target_layer)
            elif args.expl_type == 'shap':
                # As we're using images it should be fine to use the all zero tensor as a baseline
                baseline = torch.zeros_like(images)
                baseline = baseline.cuda()

                expl_out = shap.attribute(images, baselines=baseline, target=labels)
            elif args.expl_type == 'c-gradcam':
                expl_out = guided_gcam.attribute(images, target=labels)
            else:
                raise NotImplementedError('Explanation technique {} not supported!'.format(args.expl_type))

            expl_out = expl_out.cpu()
            labels = labels.cpu()
            if args.save_output:
                with torch.no_grad():
                    outputs = model(images)

                outputs = outputs.cpu()

                for i in range(len(outputs)):
                    all_outputs.append(outputs[i])

            for i in range(len(expl_out)):
                all_expl.append(expl_out[i])
                all_targets.append(labels[i])

    with open(os.path.join(args.output_dir, '{}.pt'.format(args.expl_type)), 'wb') as f:
        torch.save(torch.stack(all_expl), f)

    with open(os.path.join(args.output_dir, 'targets.pt'), 'wb') as f:
        torch.save(torch.stack(all_targets), f)

    if args.save_output:
        with open(os.path.join(args.output_dir, 'outputs.pt'), 'wb') as f:
            torch.save(torch.stack(all_outputs), f)

    print(len(torch.stack(all_expl)))
    print('----------- {} saved to {}'.format(args.expl_type.upper(), os.path.join(args.output_dir,
                                                                                   '{}.pt'.format(args.expl_type))))

if __name__ == '__main__':
    main()
