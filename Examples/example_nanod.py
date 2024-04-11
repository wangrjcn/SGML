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
                                 solution_functions='default',
                                 model_loadpath='default',
                                 model_savepath='default',
                                 hidden_layers=[4, 4],
                                 activation_function='default',
                                 batch_size='default',
                                 criterion='default',
                                 optimizer='default',
                                 learning_rate='default',
                                 epochs='default')

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
                                     solution_functions='default',
                                     model_loadpath='default',
                                     model_savepath='default',
                                     alpha=0.01,
                                     fit_intercept='default',
                                     copy_X='default',
                                     max_iter='default',
                                     tol='default',
                                     solver='default',
                                     positive='default',
                                     random_state='default')

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
                                                     solution_functions='default',
                                                     model_loadpath='default',
                                                     model_savepath='default',
                                                     max_iter='default',
                                                     tol='default',
                                                     alpha_1='default',
                                                     alpha_2='default',
                                                     lambda_1='default',
                                                     lambda_2='default',
                                                     alpha_init='default',
                                                     lambda_init='default',
                                                     compute_score='default',
                                                     fit_intercept='default',
                                                     copy_X='default',
                                                     verbose='default')

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
                                           solution_functions='default',
                                           model_loadpath='default',
                                           model_savepath='default',
                                           estimator='default',
                                           n_estimators=100,
                                           learning_rate=0.001,
                                           loss='default',
                                           random_state='default')

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
                                 solution_functions='default',
                                 model_loadpath='default',
                                 model_savepath='default',
                                 kernel='linear',
                                 degree='default',
                                 gamma='default',
                                 coef0='default',
                                 tol='default',
                                 C=100,
                                 epsilon=0.001,
                                 shrinking='default',
                                 cache_size='default',
                                 verbose='default',
                                 max_iter='default')

MSE = np.sum((y_pre - y_test) ** 2) / len(y_test)
RMSE = MSE ** 0.5
print('RMSE=', RMSE)
