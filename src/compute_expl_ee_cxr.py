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
from train_target_model import load_test_train_split, Net, SmallNet
from FEMNISTDataset import create_leaf_fldataset
from explanation_ensemble_cxr import Ensemble, get_models
from compute_expl import _BaseWrapper, GradCam, compute_gradCAM
from TabularDatasets import CompasDataset, AdultDataset, TexasDataset
from cxr import load_model, load_data


def main():
    parser = argparse.ArgumentParser(description='Compute and save GradCAM or SHAP attributions for a given'
                                                 'explanation ensemble model')

    parser.add_argument('--model-path', type=str, help='Path to model checkpoint to load', required=True)
    parser.add_argument('--model-type', default='densenet', choices=['densenet'])

    parser.add_argument('--data-idx-path', type=str, default=None,
                        help='Path to indices of data to use for training/test split')

    parser.add_argument('--cxr-data-path', type=str, default='/media/hdd/mimic-cxr-jpg',
                        help='Path to JPG CXR data')
    parser.add_argument('--gaze-data-path', type=str, default='/media/hdd/mimic-cxr-eye',
                        help='Path the EGD data')
    parser.add_argument('--generated-heatmaps-path', type=str, default=None,
                        help='Path to pre-generated heatmaps. If None, generate heatmaps at runtime')

    parser.add_argument('--expl-type', choices=['gradcam', 'shap', 'c-gradcam'], default='gradcam',
                        help='Explanation method to use')
    parser.add_argument('--save-output', action='store_true', help='Also save the output of the model on each input')
    parser.add_argument('--expl-per-submodel', action='store_true',
                        help='Calculate separate expalanations for each submodels in the ensemble')

    parser.add_argument('--dataset', choices=['egd'], default='egd')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size to use')

    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')

    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    args.model = args.model_type
    args.dropout = 0 # We run in eval mode so this value doesn't matter, is needed for loading the model though
    args.verbose = False

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
    if args.dataset == 'egd':
        train_loader, _ = load_data(args.gaze_data_path, args.cxr_data_path, args.generated_heatmaps_path,
                                    test_split=0.2, batch_size=1,
                                    num_workers=1, train_idx_path=args.data_idx_path,
                                    test_idx_path=args.data_idx_path)

        num_labels = 3
        num_features = 267456


    # Load the model
    models = get_models(args, train=True, as_ensemble=False, model_file=args.model_path)
    model = Ensemble(models)

    np.random.seed(42)
    torch.manual_seed(42)

    model = model.cuda()
    model.eval()

    if args.model_type == 'densenet':
        candidate_layers = ['features.denseblock4.denselayer16']
        target_layer = 'features.denseblock4.denselayer16'

        if args.expl_type == 'c-gradcam':
            guided_gcam = GuidedGradCam(model, model.models[0].features.denseblock4.denselayer16)
    else:
        raise NotImplementedError("Model architecture {} is not supported!".format(args.model_type))

    if args.expl_type == 'shap':
        if args.expl_per_submodel:
            shap = [GradientShap(m) for m in model.models]
        else:
            shap = GradientShap(model)
    else:
        # The final GradCAM output of the Ensemble is the weighted average of GradCAM for each of the sub-models
        # Using the same weighting as the prediction
        gcams = []
        for i in range(len(models)):
            gcams.append(GradCam(model=models[i], candidate_layers=candidate_layers))

    # Each entry is a tuple, (explanation, target)
    all_expl = []
    all_targets = []
    all_outputs = []
    j = -1

    with torch.no_grad():
        for images, _, labels in tqdm(train_loader):
            images = images.cuda()
            labels = labels.cuda()
            _, labels = torch.max(labels, 2)

            if args.expl_type == 'gradcam':
                prob_activation = nn.Sigmoid()

                output = model(images)

                expl_outs = []
                for i in range(len(models)):
                    _, expl_out, one_hot = compute_gradCAM(output, labels, gcams[i], False, prob_activation, target_layer)
                    expl_outs.append(expl_out)

                if args.expl_per_submodel:
                    expl_out = expl_outs
                else:
                    expl_out = torch.mean(torch.stack(expl_outs), dim=0)
            elif args.expl_type == 'shap':
                # As we're using images it should be fine to use the all zero tensor as a baseline
                baseline = torch.zeros_like(images)
                baseline = baseline.cuda()

                if args.expl_per_submodel:
                    expl_outs = []
                    for i in range(len(model.models)):
                        expl = shap[i].attribute(images, baselines=baseline, target=labels)
                        expl_outs.append(expl)

                    expl_out = expl_outs
                else:
                    expl_out = shap.attribute(images, baselines=baseline, target=labels)
            else:
                raise NotImplementedError('Explanation technique {} not supported!'.format(args.expl_type))
            if args.expl_per_submodel:
                expl_out = torch.stack(expl_out)

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
