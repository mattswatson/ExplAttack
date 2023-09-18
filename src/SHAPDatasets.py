import torch
from torch.utils.data import Dataset

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


class SHAPDataset(Dataset):
    def __init__(self, csv_path, return_label=None, remove_target=True, remove_output=True):
        """
        :param csv_path: string to CSV/Pickle file containing SHAP data
        :param return_label: If not None, return only samples with the given label
        """
        super(SHAPDataset, self).__init__()

        try:
            self.df = pd.read_csv(csv_path)
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            self.df = pd.read_pickle(csv_path)
        except:
            raise TypeError('{} must be a .pkl or .csv file!'.format(csv_path))

        if remove_target:
            try:
                self.df = self.df.drop([c for c in self.df.columns if 'target' in c], axis=1)
            except KeyError:
                pass

        if remove_output:
            try:
                self.df = self.df.drop([c for c in self.df.columns if 'output' in c], axis=1)
            except KeyError:
                pass

        if return_label is not None:
            if return_label not in np.unique(np.array(self.df['in_training'], dtype=np.int32)):
                raise ValueError('{} is not a valid label!'.format(return_label))

            self.df = self.df[self.df['in_training'] == return_label]

        self.targets = np.array(self.df['in_training'], dtype=np.int32)

        del self.df['in_training']

        try:
            del self.df['Unnamed: 0']
        except Exception:
            pass

        # Make sure we actually do have samples that weren't used in training
        if return_label is None and len(np.unique(self.targets)) != 2:
            raise ValueError('Only one class present in dataset {}'.format(csv_path))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        sample = self.df.iloc[item]
        target = self.targets[item]

        return torch.tensor(np.array(sample)).type(torch.FloatTensor), torch.tensor([target]).type(torch.FloatTensor)

    def get_scaler(self, idx=None):
        """
        Returns a StandardScaler object, fit to the data
        :param idx: If None, fit Scaler to the whole dataset. If iterable, fit scaler to just indices in the iterable
        :return:
        """
        scaler = StandardScaler()

        if idx is not None:
            scaler = scaler.fit(self.df.iloc[idx])
        else:
            scaler = scaler.fit(self.df)

        return scaler

    def apply_scaler(self, scaler, idx=None):
        """
        Applies a StandardScaler fit_transform to the dataset (or subset if idx is not None
        :param scaler:
        :return:
        """

        if idx is not None:
            self.df.iloc[idx] = pd.DataFrame(scaler.fit_transform(self.df.iloc[idx]))
        else:
            self.df = pd.DataFrame(scaler.fit_transform(self.df))
