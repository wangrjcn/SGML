import SGML
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import random
import matplotlib.pyplot as plt

current_dir = os.getcwd().replace('\\', '/')
# %% ann

seed_value = 42
torch.manual_seed(seed_value)

solution_1 = SGML.create_solution_function(expression='(1 / (1.05 - 0.15 * 0.165 - 0.16 * 0.165 ** 2)) ** 3 * delta_a ** 3',
                                           variables=['delta_a'])
solution_2 = SGML.create_solution_function(expression='9 * 3.1415926 / 16 * (delta_a ** 3) * (R_a ** 0.25)',
                                           variables=['R_a', 'delta_a'])

my_ann, y_test, y_pre = SGML.ann(train_path=os.path.join(current_dir, 'dataset/nanod_train1.csv'),
                                 test_path=os.path.join(current_dir, 'dataset/nanod_test1.csv'),
                                 feature_names=['R_a', 'delta_a'],
                                 lable_names=['F_Eah'],
                                 # solution_functions=[solution_1, solution_2],
                                 solution_functions=None,
                                 model_loadpath=None,
                                 model_savepath=None,
                                 hidden_layers=[4, 4],
                                 activation_function=None,
                                 batch_size=None,
                                 criterion=None,
                                 optimizer=None,
                                 learning_rate=None,
                                 epochs=None)

MSE = np.sum((y_pre - y_test) ** 2) / len(y_test)
RMSE = MSE ** 0.5
print('RMSE=', RMSE)

# %% ridge

solution_1 = SGML.create_solution_function(expression='(1 / (1.05 - 0.15 * 0.165 - 0.16 * 0.165 ** 2)) ** 3 * delta_a ** 3',
                                           variables=['delta_a'])
solution_2 = SGML.create_solution_function(expression='9 * 3.1415926 / 16 * (delta_a ** 3) * (R_a ** 0.25)',
                                           variables=['R_a', 'delta_a'])

my_ridge, y_test, y_pre = SGML.ridge(train_path=os.path.join(current_dir, 'dataset/nanod_train1.csv'),
                                     test_path=os.path.join(current_dir, 'dataset/nanod_test1.csv'),
                                     feature_names=['R_a', 'delta_a'],
                                     lable_names=['F_Eah'],
                                     # solution_functions=[solution_1, solution_2],
                                     solution_functions=None,
                                     model_loadpath=None,
                                     model_savepath=None,
                                     alpha=0.01,
                                     fit_intercept=None,
                                     copy_X=None,
                                     max_iter=None,
                                     tol=None,
                                     solver=None,
                                     positive=None,
                                     random_state=None)

MSE = np.sum((y_pre - y_test) ** 2) / len(y_test)
RMSE = MSE ** 0.5
print('RMSE=', RMSE)

# %% bayesianridge

solution_1 = SGML.create_solution_function(expression='(1 / (1.05 - 0.15 * 0.165 - 0.16 * 0.165 ** 2)) ** 3 * delta_a ** 3',
                                           variables=['delta_a'])
solution_2 = SGML.create_solution_function(expression='9 * 3.1415926 / 16 * (delta_a ** 3) * (R_a ** 0.25)',
                                           variables=['R_a', 'delta_a'])

my_bayesianridge, y_test, y_pre = SGML.bayesianridge(train_path=os.path.join(current_dir, 'dataset/nanod_train1.csv'),
                                                     test_path=os.path.join(current_dir, 'dataset/nanod_test1.csv'),
                                                     feature_names=['R_a', 'delta_a'],
                                                     lable_names=['F_Eah'],
                                                     # solution_functions=[solution_1, solution_2],
                                                     solution_functions=None,
                                                     model_loadpath=None,
                                                     model_savepath=None,
                                                     max_iter=None,
                                                     tol=None,
                                                     alpha_1=None,
                                                     alpha_2=None,
                                                     lambda_1=None,
                                                     lambda_2=None,
                                                     alpha_init=None,
                                                     lambda_init=None,
                                                     compute_score=None,
                                                     fit_intercept=None,
                                                     copy_X=None,
                                                     verbose=None)

MSE = np.sum((y_pre - y_test) ** 2) / len(y_test)
RMSE = MSE ** 0.5
print('RMSE=', RMSE)

# %% adaboost

solution_1 = SGML.create_solution_function(expression='(1 / (1.05 - 0.15 * 0.165 - 0.16 * 0.165 ** 2)) ** 3 * delta_a ** 3',
                                           variables=['delta_a'])
solution_2 = SGML.create_solution_function(expression='9 * 3.1415926 / 16 * (delta_a ** 3) * (R_a ** 0.25)',
                                           variables=['R_a', 'delta_a'])

my_adaboost, y_test, y_pre = SGML.adaboost(train_path=os.path.join(current_dir, 'dataset/nanod_train1.csv'),
                                           test_path=os.path.join(current_dir, 'dataset/nanod_test1.csv'),
                                           feature_names=['R_a', 'delta_a'],
                                           lable_names=['F_Eah'],
                                           # solution_functions=[solution_1, solution_2],
                                           solution_functions=None,
                                           model_loadpath=None,
                                           model_savepath=None,
                                           estimator=None,
                                           n_estimators=100,
                                           learning_rate=0.001,
                                           loss=None,
                                           random_state=None)

MSE = np.sum((y_pre - y_test) ** 2) / len(y_test)
RMSE = MSE ** 0.5
print('RMSE=', RMSE)

# %% svr

solution_1 = SGML.create_solution_function(expression='(1 / (1.05 - 0.15 * 0.165 - 0.16 * 0.165 ** 2)) ** 3 * delta_a ** 3',
                                           variables=['delta_a'])
solution_2 = SGML.create_solution_function(expression='9 * 3.1415926 / 16 * (delta_a ** 3) * (R_a ** 0.25)',
                                           variables=['R_a', 'delta_a'])

my_svr, y_test, y_pre = SGML.svr(train_path=os.path.join(current_dir, 'dataset/nanod_train1.csv'),
                                 test_path=os.path.join(current_dir, 'dataset/nanod_test1.csv'),
                                 feature_names=['R_a', 'delta_a'],
                                 lable_names=['F_Eah'],
                                 # solution_functions=[solution_1, solution_2],
                                 solution_functions=None,
                                 model_loadpath=None,
                                 model_savepath=None,
                                 kernel='linear',
                                 degree=None,
                                 gamma=None,
                                 coef0=None,
                                 tol=None,
                                 C=100,
                                 epsilon=0.001,
                                 shrinking=None,
                                 cache_size=None,
                                 verbose=None,
                                 max_iter=None)

MSE = np.sum((y_pre - y_test) ** 2) / len(y_test)
RMSE = MSE ** 0.5
print('RMSE=', RMSE)
