import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler

import pickle
import json
import numpy as np
import os
from collections import defaultdict
import pandas as pd

import string


class CompasDataset(Dataset):
    def __init__(self, data_path, normalise=True, train=True):
        self.df = pd.read_csv(os.path.join(data_path, 'data.csv'))

        # These are the columns we want to keep
        columns = ['sex', 'age', 'race', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count',
                   'priors_count', 'c_days_from_compas', 'c_charge_degree', 'c_charge_desc', 'is_recid',
                   'r_charge_degree',
                   'r_days_from_arrest', 'is_violent_recid']
        self.df = self.df[columns]

        # for categorical columns, replace NA with mode. For scalar columns, replace with mean
        scalar_columns = ['age', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count',
                          'priors_count', 'c_days_from_compas', 'r_days_from_arrest']
        for c in scalar_columns:
            self.df[c] = self.df[c].fillna(self.df[c].mean())

        for c in self.df.columns:
            if c not in scalar_columns:
                self.df[c] = self.df[c].fillna(self.df[c].mode())

        # Convert categorical columns to one-hot encodings (leave those that are binary already)
        for c in ['sex', 'race', 'c_charge_degree', 'c_charge_desc', 'r_charge_degree']:
            one_hot = pd.get_dummies(self.df[c], prefix=c)

            del self.df[c]
            self.df = self.df.join(one_hot)

        if train:
            idx = list(range(len(self.df)))
            train_idx = idx[:int(0.7 * len(idx))]
            self.df = self.df.loc[train_idx]

        self.targets = self.df['is_recid']
        del self.df['is_recid']

        if normalise:
            self.scaler = StandardScaler().fit(self.df)
            self.df = self.scaler.transform(self.df)

    def __getitem__(self, item):
        data = torch.tensor(self.df[item]).long()
        label = torch.tensor(self.targets[item]).long()

        return data, label

    def __len__(self):
        return len(self.df)

    def apply_scaler(self, scaler):
        self.df = scaler.transform(self.df)


class AdultDataset(Dataset):
    def __init__(self, data_path, normalise=True, train=True):
        df_train = pd.read_csv(os.path.join(data_path, 'adult.data'))
        df_test = pd.read_csv(os.path.join(data_path, 'adult.test'))

        # Convert categorical columns to one-hot encodings (leave those that are binary already)
        for c in ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                  'hours-per-week', 'native-country']:
            one_hot = pd.get_dummies(df_test[c], prefix=c)
            one_hot_train = pd.get_dummies(df_train[c], prefix=c)

            del df_test[c]
            df_test = df_test.join(one_hot)

            del df_train[c]
            df_train = df_train.join(one_hot_train)

        # Some one-hot columns are not present in the test data - add them here
        for c in df_train.columns:
            if c not in df_test.columns:
                df_test[c] = 0

        for c in df_test.columns:
            if c not in df_train.columns:
                df_train[c] = 0

        if train:
            self.df = df_train
            del df_test
        else:
            self.df = df_test
            del df_train

        # Convert earnings column to binary
        self.df['earnings'] = self.df['earnings'].map({' <=50K': 0, ' >50K': 1})
        print(self.df['earnings'].value_counts())

        self.targets = self.df['earnings']
        del self.df['earnings']

        if normalise:
            self.scaler = StandardScaler().fit(self.df)
            self.df = self.scaler.transform(self.df)

    def __getitem__(self, item):
        data = torch.tensor(self.df[item]).long()
        label = torch.tensor(self.targets[item]).long()

        return data, label

    def __len__(self):
        return len(self.df)

    def apply_scaler(self, scaler):
        self.df = scaler.transform(self.df)


class TexasDataset(Dataset):
    def __init__(self, data_path, normalise=True, train=True):
        self.df = pd.read_csv(os.path.join(data_path, 'data.csv'), delimiter='\t')

        cols_to_read = ['PROVIDER_NAME', 'FAC_TEACHING_IND', 'FAC_PSYCH_IND', 'FAC_REHAB_IND', 'FAC_ACUTE_CARE_IND',
                        'FAC_SNF_IND', 'FAC_LONG_TERM_AC_IND', 'FAC_OTHER_LTC_IND', 'FAC_PEDS_IND',
                        'ENCOUNTER_INDICATOR', 'SEX_CODE', 'TYPE_OF_ADMISSION', 'SOURCE_OF_ADMISSION', 'PAT_STATE',
                        'PAT_COUNTRY', 'PUBLIC_HEALTH_REGION', 'ADMIT_WEEKDAY', 'LENGTH_OF_STAY', 'PAT_AGE',
                        'PAT_STATUS', 'RACE', 'ETHNICITY', 'FIRST_PAYMENT_SRC', 'SECONDARY_PAYMENT_SRC', 'TYPE_OF_BILL',
                        'PRIVATE_AMOUNT', 'SEMI_PRIVATE_AMOUNT', 'WARD_AMOUNT', 'ICE_AMOUNT', 'CCU_AMOUNT',
                        'OTHER_AMOUNT', 'PHARM_AMOUNT', 'MEDSURG_AMOUNT', 'DME_AMOUNT', 'USED_DME_AMOUNT', 'PT_AMOUNT',
                        'OT_AMOUNT', 'SPEECH_AMOUNT', 'IT_AMOUNT', 'BLOOD_AMOUNT', 'BLOOD_ADM_AMOUNT', 'OR_AMOUNT',
                        'LITH_AMOUNT', 'CARD_AMOUNT', 'ANES_AMOUNT', 'LAB_AMOUNT', 'RAD_AMOUNT', 'MRI_AMOUNT',
                        'OP_AMOUNT', 'ER_AMOUNT', 'AMBULANCE_AMOUNT', 'PRO_FEE_AMOUNT', 'ORGAN_AMOUNT', 'ESRD_AMOUNT',
                        'CLINIC_AMOUNT', 'TOTAL_CHARGES', 'TOTAL_NON_COV_CHARGES', 'TOTAL_CHARGES_ACCOMM',
                        'TOTAL_NON_COV_CHARGES_ACCOMM', 'TOTAL_CHARGES_ANCIL', 'TOTAL_NON_COV_CHARGES_ANCIL',
                        'PRINC_SURG_PROC_CODE', 'PRINC_SURG_PROC_DAY', 'HCFA_MDC', 'APR_MDC', 'RISK_MORTALITY',
                        'ILLNESS_SEVERITY', 'CERT_STATUS']
        categorical_columns = ['PROVIDER_NAME', 'FAC_TEACHING_IND', 'FAC_PSYCH_IND', 'FAC_REHAB_IND',
                               'FAC_ACUTE_CARE_IND', 'FAC_SNF_IND', 'FAC_LONG_TERM_AC_IND', 'FAC_OTHER_LTC_IND',
                               'FAC_PEDS_IND', 'SEX_CODE', 'TYPE_OF_ADMISSION', 'SOURCE_OF_ADMISSION', 'PAT_STATE',
                               'PAT_COUNTRY', 'PUBLIC_HEALTH_REGION', 'ADMIT_WEEKDAY', 'PAT_STATUS', 'RACE',
                               'ETHNICITY', 'FIRST_PAYMENT_SRC', 'SECONDARY_PAYMENT_SRC', 'TYPE_OF_BILL',
                               'PRINC_SURG_PROC_CODE', 'HCFA_MDC', 'APR_MDC', 'CERT_STATUS']

        self.df = self.df[cols_to_read]

        # Get the top 100 most common procedures
        procedure_counts = self.df['PRINC_SURG_PROC_CODE'].value_counts()
        procedures_to_use = procedure_counts.sort_values(ascending=False).iloc[:100]
        procedures_list = list(procedures_to_use.keys())

        self.df = self.df[self.df['PRINC_SURG_PROC_CODE'].isin(procedures_list)]

        # Convert procedures to a class index
        self.df['label'] = self.df['PRINC_SURG_PROC_CODE'].map(lambda x: procedures_list.index(x))

        self.targets = torch.tensor(self.df['label'].values)
        del self.df['label']

        # Drop a couple of now unneeded columns
        del self.df['PRINC_SURG_PROC_CODE']
        del self.df['PROVIDER_NAME']

        # Replace NANs
        columns = self.df.columns
        numeric_columns = self.df._get_numeric_data().columns
        categorical_attributes = list(set(columns) - set(numeric_columns))
        self.df[categorical_attributes] = self.df[categorical_attributes].fillna('NA')
        self.df[numeric_columns] = self.df[numeric_columns].fillna(0)

        # Convert categorical to one hot
        for c in categorical_attributes:
            one_hot = pd.get_dummies(self.df[c], prefix=c)

            del self.df[c]
            self.df = self.df.join(one_hot)

        if train:
            idx = list(range(len(self.df)))
            train_idx = idx[:int(0.7 * len(idx))]
            self.df = self.df.iloc[train_idx]

        if normalise:
            self.scaler = StandardScaler().fit(self.df)
            self.df = self.scaler.transform(self.df)

    def __getitem__(self, item):
        data = torch.tensor(self.df[item]).long()
        label = torch.tensor(self.targets[item]).long()

        return data, label

    def __len__(self):
        return len(self.df)

    def apply_scaler(self, scaler):
        self.df = scaler.transform(self.df)