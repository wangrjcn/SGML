import SGML
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt

current_dir = os.getcwd().replace('\\', '/')
model_path = os.path.join(current_dir, 'trained_model/ann_bending.pth')

# %% svr

solution_1 = SGML.create_solution_function(expression='F * (-1 / 6 * x ** 3 + 1 / 2 * 0.3 * x ** 2) / (2100 * 180)',
                                           variables=['F', 'x'])

my_svr = SGML.svr(train_path=os.path.join(current_dir, 'dataset/bending_train1.csv'),
                  test_path=os.path.join(current_dir, 'dataset/bending_test1.csv'),
                  feature_names=['F', 'x'],
                  label_names=['y'],
                  solution_functions=[solution_1],
                  C=100,
                  epsilon=0.001)

my_svr.train()

y_test = my_svr.test()
y_pre = my_svr.predict()

my_svr.plot_results(y_test, y_pre)

