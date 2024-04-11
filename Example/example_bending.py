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
                                 solution_functions=None,
                                 model_loadpath=None,
                                 model_savepath=None,
                                 hidden_layers=[8, 8],
                                 activation_function=None,
                                 batch_size=None,
                                 criterion=None,
                                 optimizer=None,
                                 learning_rate=None,
                                 epochs=None)

# %% ridge

solution_1 = SGML.create_solution_function(expression='F * (-1 / 6 * x ** 3 + 1 / 2 * 0.3 * x ** 2) / (2100 * 180)',
                                           variables=['F', 'x'])

my_ridge, y_test, y_pre = SGML.ridge(train_path=os.path.join(current_dir, 'dataset/bending_train1.csv'),
                                     test_path=os.path.join(current_dir, 'dataset/bending_test1.csv'),
                                     feature_names=['F', 'x'],
                                     lable_names=['y'],
                                     solution_functions=[solution_1],
                                     model_loadpath=None,
                                     model_savepath=None,
                                     alpha=0.1,
                                     fit_intercept=None,
                                     copy_X=None,
                                     max_iter=None,
                                     tol=None,
                                     solver=None,
                                     positive=None,
                                     random_state=None)

# %% bayesianridge

solution_1 = SGML.create_solution_function(expression='F * (-1 / 6 * x ** 3 + 1 / 2 * 0.3 * x ** 2) / (2100 * 180)',
                                           variables=['F', 'x'])

my_bayesian, y_test, y_pre = SGML.bayesianridge(train_path=os.path.join(current_dir, 'dataset/bending_train1.csv'),
                                                test_path=os.path.join(current_dir, 'dataset/bending_test1.csv'),
                                                feature_names=['F', 'x'],
                                                lable_names=['y'],
                                                solution_functions=[solution_1],
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

# %% adaboost

solution_1 = SGML.create_solution_function(expression='F * (-1 / 6 * x ** 3 + 1 / 2 * 0.3 * x ** 2) / (2100 * 180)',
                                           variables=['F', 'x'])

my_adaboost, y_test, y_pre = SGML.adaboost(train_path=os.path.join(current_dir, 'dataset/bending_train1.csv'),
                                           test_path=os.path.join(current_dir, 'dataset/bending_test1.csv'),
                                           feature_names=['F', 'x'],
                                           lable_names=['y'],
                                           solution_functions=[solution_1],
                                           model_loadpath=None,
                                           model_savepath=None,
                                           estimator=None,
                                           n_estimators=None,
                                           learning_rate=None,
                                           loss=None,
                                           random_state=None)

# %% svr

solution_1 = SGML.create_solution_function(expression='F * (-1 / 6 * x ** 3 + 1 / 2 * 0.3 * x ** 2) / (2100 * 180)',
                                           variables=['F', 'x'])

my_model, y_test, y_pre = SGML.svr(train_path=os.path.join(current_dir, 'dataset/bending_train1.csv'),
                                   test_path=os.path.join(current_dir, 'dataset/bending_test1.csv'),
                                   feature_names=['F', 'x'],
                                   lable_names=['y'],
                                   solution_functions=[solution_1],
                                   model_loadpath=None,
                                   model_savepath=None,
                                   kernel=None,
                                   degree=None,
                                   gamma=None,
                                   coef0=None,
                                   tol=None,
                                   C=100,
                                   epsilon=None,
                                   shrinking=None,
                                   cache_size=None,
                                   verbose=None,
                                   max_iter=None)
