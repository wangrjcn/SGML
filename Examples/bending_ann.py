import SGML
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt

current_dir = os.getcwd().replace('\\', '/')
model_path = os.path.join(current_dir, 'trained_model/ann_bending.pth')
# %% ann

solution_1 = SGML.create_solution_function(expression='F * (-1 / 6 * x ** 3 + 1 / 2 * 0.3 * x ** 2) / (2100 * 180)',
                                           variables=['F', 'x'])

my_ann = SGML.ann(train_path=os.path.join(current_dir, 'dataset/bending_train1.csv'),
                  test_path=os.path.join(current_dir, 'dataset/bending_test1.csv'),
                  feature_names=['F', 'x'],
                  label_names=['y'],
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

my_ann.train()

y_test = my_ann.test()
y_pre = my_ann.predict()

my_ann.plot_results(y_test, y_pre)

