# ExplAttack

Supporting repository for the paper M. Watson, B. Awwad Shiekh Hasan, N. Al Moubayed "Membership Inference Attacks and 
Defences Using Deep Learning Model Explanations", Under Review, 2023.

Code to reproduce all experiments included in the paper are included. Note that all model training and SHAP 
calculation code has CLI options for setting the random seed used. To accurately reproduce our
experiments, one must set the random seed to those reported in the paper. `requirements.txt` include all Python
dependencies required.

All models are saved as state dicts. All runnable scripts accept a number of CLI arguments which are explained through 
comments and the use of `python script_name.py -h`. All code is in the `src/` directory.

- `art_mia_attack.py` trains baseline membership inference attacks (MIA) (using implementations in the
  ([Adversarial Robustness Toolbox]https://github.com/Trusted-AI/adversarial-robustness-toolbox)) on all datasets
  (except MIMIC-CXR-EGD)
- `art_mia_attack_cxr.py` trains baseline membership inference attacks (MIA) (using implementations in the
  ([Adversarial Robustness Toolbox]https://github.com/Trusted-AI/adversarial-robustness-toolbox)) on MIMIC-CXR-EGD
- `compute_expl.py` computes SHAP values for baseline models on all datasets
- `compute_expl_across_clients.py` computes SHAP values for federated models
- `compute_expl_ee.py` computes SHAP values for Deep Explanation Ensemble (DEE) models
- `compute_expl_ee_cxr.py` computes SHAP values for DEEs on the MIMIC-CXR-EGD dataset
- `cxr.py` trains [baseline models](https://www.nature.com/articles/s41597-021-00863-5) on MIMIC-CXR-EGD
- `explanation_ensemble.py` trains DEEs on all datasets other than MIMIC-CXR-EGD
- `explanation_ensemble_cxr.py` trains DEEs on the MIMIC-CXR-EGD dataset
- `FEMNISTDataset.py` contains supporting code to use the [FEMNIST](https://leaf.cmu.edu) dataset
- `lr.py` trains a logistic regression model on SHAP values from a downstream model, as an MIA (ExplAttack LR)
- `mlp.py` trains an MLP on SHAP values from a downstream model, as an MIA (ExplAttack MLP)
- `models.py` contains supporting code for all downstream model architectures
- `SHAPDatasets.py` contains supporting code for using SHAP values during training
- `TabularDatasets.py` contains supporting code for all tabular datasets used
- `train_target_model.py` trains baseline models on a classifcation task. Can optionally use differential privacy
- `train_target_model_fl.py` trains baseline models using federated learning. Can optionally use differential privacy
- `utils.py` contains supporting code
- `XrayDataset.py` contains supporting code for using the MIMIC-CXR-EGD dataset

As a general rule, experiments should follow the same pattern:

- Train a downstream model on a classification task (e.g. using `train_target_model.py`)
- Calculate SHAP values for this model using `calculate_expl.py`
- Train baseline MIAs on this model using `art_mia_attack.py`
- Train some variety of ExplAttack on the SHAP values from the original model (using either `mlp.py` or `lr.py`)
- Results from ExplAttack can then be compared against results from the attacks included in ART (from step 3)

# MIMIC-CXR-EGD

We extensively use the [MIMIC-CXR-EGD](https://www.nature.com/articles/s41597-021-00863-5) dataset and test against some
of their baseline models using their code, which is found in the `egd_code` directory.