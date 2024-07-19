import SGML
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import random
import matplotlib.pyplot as plt

current_dir = os.getcwd().replace('\\', '/')
model_path = os.path.join(current_dir, 'trained_model/ann_nanod.pth')
# %% ann

seed_value = 42
torch.manual_seed(seed_value)

solution_1 = SGML.create_solution_function(expression='(1 / (1.05 - 0.15 * 0.165 - 0.16 * 0.165 ** 2)) ** 3 * delta_a ** 3',
                                           variables=['delta_a'])
solution_2 = SGML.create_solution_function(expression='9 * 3.1415926 / 16 * (delta_a ** 3) * (R_a ** 0.25)',
                                           variables=['R_a', 'delta_a'])

my_ann = SGML.ann(train_path=os.path.join(current_dir, 'dataset/nanod_train1.csv'),
                  test_path=os.path.join(current_dir, 'dataset/nanod_test1.csv'),
                  feature_names=['R_a', 'delta_a'],
                  label_names=['F_Eah'],
                  solution_functions=[solution_1, solution_2],
                  # solution_functions='default',
                  model_loadpath='default',
                  model_savepath='default',
                  hidden_layers='default',
                  activation_function='default',
                  batch_size='default',
                  criterion='default',
                  optimizer='default',
                  learning_rate='default',
                  epochs='default')

my_ann.train()

y_test = my_ann.test()
y_pre = my_ann.predict()

my_ann.plot_results(y_test, y_pre)

