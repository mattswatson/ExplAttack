import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler
import torch.nn.functional as F

import numpy as np
from collections import OrderedDict

from captum.attr import GradientShap, GuidedGradCam

import syft as sy

# These are for our own baselines we've created
from torchvision import datasets, transforms

from models import SmallCNN


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
            one_hot[i][torch.max(labels, 1)[1][i]] = 1.0

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
    parser.add_argument('--model-type', default='small-cnn', choices=['small-cnn'])

    parser.add_argument('--expl-type', choices=['gradcam', 'shap', 'c-gradcam'], default='gradcam',
                        help='Explanation method to use')
    parser.add_argument('--save-output', action='store_true', help='Also save the output of the model on each input')

    parser.add_argument('--dataset', choices=['mnist'], default='mnist')
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size to use')

    parser.add_argument('--num-clients', type=int, default=2)

    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')

    parser.add_argument('--seed', type=int, default=1)


    args = parser.parse_args()

    hook = sy.TorchHook(torch)
    torch.manual_seed(args.seed)

    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda_available else {}

    # Create virtual workers
    clients = [sy.VirtualWorker(hook, id='client{}'.format(i)) for i in range(args.num_clients)]

    # Load the model
    if args.model_type == 'small-cnn':
        model = SmallCNN()
        model = model.to(device)

        candidate_layers = ['conv2']
        target_layer = 'conv2'

        if args.expl_type == 'c-gradcam':
            guided_gcam = GuidedGradCam(model, model.conv2)
    else:
        raise NotImplementedError("Model architecture {} is not supported!".format(args.model_type))

    np.random.seed(42)
    torch.manual_seed(42)

    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()
    model.eval()

    if args.expl_type == 'shap':
        shap = GradientShap(model)

    gcam = GradCam(model=model, candidate_layers=candidate_layers)

    # Load the data and split it between the two virtual workers
    if args.dataset == 'mnist':
        federated_train_loader = sy.FederatedDataLoader(
            datasets.MNIST(args.data_path, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])).federate(clients),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    else:
        raise NotImplementedError('Dataset {} not implemented!'.format(args.dataset))

    # Dictionary entry for each client, then it's just a list of explanations/targets
    all_expl = {'client{}'.format(i): [] for i in range(args.num_clients)}
    all_targets = {'client{}'.format(i): [] for i in range(args.num_clients)}
    all_outputs = {'client{}'.format(i): [] for i in range(args.num_clients)}
    for images, labels in tqdm(federated_train_loader):
        location_id = images.location.id

        # Get images/labels from client
        images = images.get()
        labels = labels.get()

        #print(images)
        #print(labels)

        images = images.to(device)
        labels = labels.to(device)

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

        if args.save_output:
            with torch.no_grad():
                outputs = model(images)
            outputs = outputs.cpu()

            for i in range(len(outputs)):
                all_outputs[location_id].append(outputs[i])

        for i in range(len(expl_out)):
            all_expl[location_id].append(expl_out[i])
            all_targets[location_id].append(labels[i])

    for client in all_expl:
        with open(os.path.join(args.output_dir, '{}-{}.pt'.format(args.expl_type, client)), 'wb') as f:
            torch.save(torch.stack(all_expl[client]), f)

        with open(os.path.join(args.output_dir, 'targets-{}.pt'.format(client)), 'wb') as f:
            torch.save(torch.stack(all_targets[client]), f)

        if args.save_output:
            with open(os.path.join(args.output_dir, 'outputs-{}.pt'.format(client)), 'wb') as f:
                torch.save(torch.stack(all_outputs[client]), f)

    print('----------- {} saved to {}'.format(args.expl_type.upper(), os.path.join(args.output_dir,
                                                                                   '{}.pt'.format(args.expl_type))))

if __name__ == '__main__':
    main()
