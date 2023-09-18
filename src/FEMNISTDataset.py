import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler

import pickle
import json
import numpy as np
import os
from collections import defaultdict

import string


def hex_label_to_char(label):
    """
    FEMNIST data uses hexadecimal labels:
    - 0 through 9 for classes representing respective numbers
    - 10 through 35 for classes representing respective uppercase letters
    - 36 through 61 for classes representing respective lowercase letters
    This function returns the letter from the label
    """

    alphabet = string.ascii_lowercase

    if label <= 9:
        return str(label)

    if label <= 35:
        return alphabet[label - 10]

    alphabet = alphabet.upper()
    return alphabet[label - 36]


def batch_data(data, batch_size, seed):
    """
    data is a dict := {'x': torch tensor, 'y': torch tensor}
    returns x, y, which are both numpy array of length: batch_size
    """
    data_x = data['x'].detach().numpy()
    data_y = data['y'].detach().numpy()
    np.random.seed(seed)
    np.random.shuffle(data_x)
    np.random.seed(seed)
    np.random.shuffle(data_y)

    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        yield batched_x, batched_y


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    train_clients, train_groups, train_data = read_dir(train_data_dir)

    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


def create_clients(users, groups, train_data, test_data, model):
    if len(groups) == 0:
        groups = [[] for _ in users]

    clients = [[g, train_data[u], test_data[u]] for u, g in zip(users, groups)]

    return clients


def setup_clients(dataset, model=None, use_val_set=False, basepath='data'):
    eval_set = "test" if not use_val_set else "val"

    train_data_dir = os.path.join(basepath, dataset, "data", "train")
    test_date_dir = os.path.join(basepath, dataset, "data", eval_set)

    data = read_data(train_data_dir, test_date_dir)
    users, groups, train_data, test_data = data
    clients = create_clients(users, groups, train_data, test_data, model)
    return clients


class ScalerDataset(Dataset):
    def __init__(self, x_data, y_data, reshape, normalise):
        if reshape is not None:
            self.x_data = x_data.reshape(*reshape)
        else:
            self.x_data = x_data

        self.y_data = y_data
        self.normalise = normalise

        if self.normalise:
            self.scaler = self.get_scaler()
        else:
            self.scaler = None

        self.y_data = torch.stack(self.y_data)

    def get_scaler(self, idx=None):
        """
        Returns a StandardScaler object, fit to the data
        :param idx: If None, fit Scaler to the whole dataset. If iterable, fit scaler to just indices in the iterable
        :return:
        """
        scaler = StandardScaler()

        if idx is not None:
            scaler = scaler.fit(self.x_daya[idx])
        else:
            scaler = scaler.fit(self.x_data)

        return scaler

    def apply_scaler(self, scaler, idx=None):
        """
        Applies a StandardScaler fit_transform to the dataset (or subset if idx is not None
        :param scaler:
        :return:
        """

        if idx is not None:
            self.x_data[idx] = torch.tensor(scaler.fit_transform(self.x_data[idx]))
        else:
            self.x_data = torch.tensor(scaler.fit_transform(self.x_data))


class FEMNIST(ScalerDataset):
    """
    Class representing one client of FEMNIST data. This should be passed to a FederatedDataset
    """

    def __init__(self, client, split='all', sy_client=None, normalise=False, reshape=(-1, 1, 28, 28)):
        """

        :param client: client list of [group, train_data, test_data]
        :param split: split to use. One of ['train', 'test', 'all']
        :param normalise: If true, run a standardscaler on the data
        :param reshape: If not none, a tuple that is the shape to resize the data to
        """

        if split == 'all':
            # Let's just make the train/test data one whole data
            x_data = torch.tensor(client[1]['x'] + client[2]['x'])

            y_data = []

            # Make the labels tensors with correct data type
            for i in range(len(client[1]['y'])):
                y_data.append(torch.tensor(float(client[1]['y'][i])).type(torch.LongTensor))

            for i in range(len(client[2]['y'])):
                y_data.append(torch.tensor(float(client[2]['y'][i])).type(torch.LongTensor))
        elif split == 'train':
            # Let's just make the train/test data one whole data
            x_data = torch.tensor(client[1]['x'])

            y_data = []

            # Make the labels tensors with correct data type
            for i in range(len(client[1]['y'])):
                y_data.append(torch.tensor(float(client[1]['y'][i])).type(torch.LongTensor))
        elif split == 'test':
            # Let's just make the train/test data one whole data
            x_data = torch.tensor(client[2]['x'])

            y_data = []

            # Make the labels tensors with correct data type
            for i in range(len(client[2]['y'])):
                y_data.append(torch.tensor(float(client[2]['y'][i])).type(torch.LongTensor))
        else:
            raise AttributeError('Split must be one of train, test, all - not {}'.format(split))

        super(FEMNIST, self).__init__(x_data, y_data, reshape, normalise)

        if reshape is not None:
            # Reshape the flat input to be an image
            pass

        if normalise:
            self.apply_scaler(self.scaler)

        if sy_client is not None:
            self.x_data.send(sy_client)
            self.y_data.send(sy_client)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, item):
        data = self.x_data[item]
        label = self.y_data[item]

        return data, label


class FEMNISTCollated(ScalerDataset):
    def __init__(self, clients_datasets, transform=None, load_all_to_gpu=False, normalise=False):
        """
        A dataset that has collated data from across clients
        :param clients_datasets: List of FEMNIST datasets
        :param split: split to use. One of ['train', 'test', 'all']
        """

        x_data = []
        y_data = []
        self.clients = None
        self.normalise = normalise

        for client in clients_datasets:
            for i in range(len(client)):
                x_data.append(client[i][0])
                y_data.append(client[i][1])

        x_data = torch.stack(x_data)

        super(FEMNISTCollated, self).__init__(x_data, y_data, None, normalise)

        self.transfrom = transform

        if normalise:
            self.apply_scaler(self.scaler)

        if load_all_to_gpu:
            self.x_data = self.x_data.cuda()

        self.x_data = self.x_data.float()
        self.y_data = self.y_data.long()

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, item):
        data = self.x_data[item]
        label = self.y_data[item]

        if self.transfrom is not None:
            data = self.transfrom(data)

        return data, label


def create_leaf_fldataset(dataset, basepath, split='all', transforms=None):
    """
    Creates a FederatedDataset object for the FEMNIST dataset
    basepath: str, path to the Leaf data directory
    split: split to use. One of ['train', 'test', 'all']
    :param dataset:
    """

    clients = setup_clients(dataset, basepath=basepath)

    if dataset == 'femnist':
        # FEMNIST class already has the correct defaults in __init__ signature
        kwargs = {}
        normalise = False
    elif dataset == 'synthetic':
        kwargs = {'normalise': False, 'reshape': None}
        normalise = True
    else:
        raise NotImplementedError('LEAF dataset {} is not supported!'.format(dataset))

    all_datasets = [FEMNIST(c, split=split, **kwargs) for c in clients]

    # Get number of samples per client
    num_samples_per_client = [len(c[1]['x']) for c in clients]

    return FEMNISTCollated(all_datasets, transform=transforms, load_all_to_gpu=False, normalise=normalise), \
           num_samples_per_client
