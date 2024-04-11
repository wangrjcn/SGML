import SGML
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt

current_dir = os.getcwd().replace('\\', '/')
# %% ann

solution_1 = SGML.create_solution_function(expression='F * (-1 / 6 * x ** 3 + 1 / 2 * 0.3 * x ** 2) / (2100 * 180)',
                                           variables=['F', 'x'])

my_ann, y_test, y_pre = SGML.ann(train_path=os.path.join(current_dir, 'dataset/bending_train1.csv'),
                                 test_path=os.path.join(current_dir, 'dataset/bending_test1.csv'),
                                 feature_names=['F', 'x'],
                                 lable_names=['y'],
                                 solution_functions=[solution_1],
                                 model_loadpath='default',
                                 model_savepath='default',
                                 hidden_layers='default',
                                 activation_function='default',
                                 batch_size='default',
                                 criterion='default',
                                 optimizer='default',
                                 learning_rate='default',
                                 epochs='default')

# %% ridge

solution_1 = SGML.create_solution_function(expression='F * (-1 / 6 * x ** 3 + 1 / 2 * 0.3 * x ** 2) / (2100 * 180)',
                                           variables=['F', 'x'])

my_ridge, y_test, y_pre = SGML.ridge(train_path=os.path.join(current_dir, 'dataset/bending_train1.csv'),
                                     test_path=os.path.join(current_dir, 'dataset/bending_test1.csv'),
                                     feature_names=['F', 'x'],
                                     lable_names=['y'],
                                     solution_functions=[solution_1],
                                     model_loadpath='default',
                                     model_savepath='default',
                                     alpha=0.1,
                                     fit_intercept='default',
                                     copy_X='default',
                                     max_iter='default',
                                     tol='default',
                                     solver='default',
                                     positive='default',
                                     random_state='default')

# %% bayesianridge

solution_1 = SGML.create_solution_function(expression='F * (-1 / 6 * x ** 3 + 1 / 2 * 0.3 * x ** 2) / (2100 * 180)',
                                           variables=['F', 'x'])

my_bayesian, y_test, y_pre = SGML.bayesianridge(train_path=os.path.join(current_dir, 'dataset/bending_train1.csv'),
                                                test_path=os.path.join(current_dir, 'dataset/bending_test1.csv'),
                                                feature_names=['F', 'x'],
                                                lable_names=['y'],
                                                solution_functions=[solution_1],
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

# %% adaboost

solution_1 = SGML.create_solution_function(expression='F * (-1 / 6 * x ** 3 + 1 / 2 * 0.3 * x ** 2) / (2100 * 180)',
                                           variables=['F', 'x'])

my_adaboost, y_test, y_pre = SGML.adaboost(train_path=os.path.join(current_dir, 'dataset/bending_train1.csv'),
                                           test_path=os.path.join(current_dir, 'dataset/bending_test1.csv'),
                                           feature_names=['F', 'x'],
                                           lable_names=['y'],
                                           solution_functions=[solution_1],
                                           model_loadpath='default',
                                           model_savepath='default',
                                           estimator='default',
                                           n_estimators='default',
                                           learning_rate='default',
                                           loss='default',
                                           random_state='default')

# %% svr

solution_1 = SGML.create_solution_function(expression='F * (-1 / 6 * x ** 3 + 1 / 2 * 0.3 * x ** 2) / (2100 * 180)',
                                           variables=['F', 'x'])

my_model, y_test, y_pre = SGML.svr(train_path=os.path.join(current_dir, 'dataset/bending_train1.csv'),
                                   test_path=os.path.join(current_dir, 'dataset/bending_test1.csv'),
                                   feature_names=['F', 'x'],
                                   lable_names=['y'],
                                   solution_functions=[solution_1],
                                   model_loadpath='default',
                                   model_savepath='default',
                                   kernel='default',
                                   degree='default',
                                   gamma='default',
                                   coef0='default',
                                   tol='default',
                                   C=100,
                                   epsilon='default',
                                   shrinking='default',
                                   cache_size='default',
                                   verbose='default',
                                   max_iter='default')
