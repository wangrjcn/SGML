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

# %% svr

solution_1 = SGML.create_solution_function(expression='(1 / (1.05 - 0.15 * 0.165 - 0.16 * 0.165 ** 2)) ** 3 * delta_a ** 3',
                                           variables=['delta_a'])
solution_2 = SGML.create_solution_function(expression='9 * 3.1415926 / 16 * (delta_a ** 3) * (R_a ** 0.25)',
                                           variables=['R_a', 'delta_a'])

my_svr = SGML.svr(train_path=os.path.join(current_dir, 'dataset/nanod_train1.csv'),
                  test_path=os.path.join(current_dir, 'dataset/nanod_test1.csv'),
                  feature_names=['R_a', 'delta_a'],
                  label_names=['F_Eah'],
                  solution_functions=[solution_1, solution_2],
                  C=100,
                  epsilon=0.001)

my_svr.train()

y_test = my_svr.test()
y_pre = my_svr.predict()

my_svr.plot_results(y_test, y_pre)
