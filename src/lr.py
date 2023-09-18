import pandas as pd
import numpy as np
import os
from argparse import ArgumentParser
from joblib import dump, load
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from utils import membership_advantage

def main():
    parser = ArgumentParser(description='Train a LR classifier on SHAP values from FL models')
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--save-path', type=str, default=None)

    parser.add_argument('--cv', type=int, default=None, help='Number of CV folds to run')
    parser.add_argument('--use-outputs', action='store_true', help='Use model output for classification as well as SHAP')
    parser.add_argument('--scale', action='store_true', help='Scale data')

    parser.add_argument('--pkl', action='store_true', help='Load pickle-saved DF instead of CSV')
    args = parser.parse_args()

    if args.pkl:
        df = pd.read_pickle(args.data_path)
    else:
        df = pd.read_csv(args.data_path)
    print('---------- Loaded data from {}'.format(args.data_path))

    if not args.use_outputs:
        try:
            df = df.drop([c for c in df.columns if 'output' in c], axis=1)
        except KeyError:
            pass

    data_train, data_test, label_train, label_test = train_test_split(df.loc[:, df.columns != 'in_training'],
                                                                      df['in_training'])

    if args.scale:
        scaler = StandardScaler()
        scaler = scaler.fit(data_train)
        data_train = scaler.transform(data_train)
        data_test = scaler.transform(data_test)

    # liblinear is meant to be best for binary classification
    if args.cv is not None:
        model = LogisticRegressionCV(max_iter=1000, cv=args.cv, solver='liblinear', verbose=1, class_weight='balanced')
    else:
        model = LogisticRegressionCV(max_iter=1000, solver='liblinear', verbose=1, class_weight='balanced')
    model.fit(data_train, label_train)
    print('---------- Model trained!')

    preds = model.predict(data_test)
    acc = np.sum(preds == label_test) / len(label_test)
    print('test accuracy: {:.4f}'.format(acc))

    # Convert our labels to the format output by the LR

    confusion_matrix = metrics.confusion_matrix(label_test, preds)
    m_adv = membership_advantage(confusion_matrix)
    print(confusion_matrix)
    print('\nMembership advantage: {:.4f}'.format(m_adv))

    if args.save_path is not None:
        dump(model, args.save_path)
        print('---------- Saved LR to {}'.format(args.save_path))


if __name__ == '__main__':
    main()
