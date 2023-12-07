# Solution-guided Machine Learning (SGML)

SGML...

# Preparation

1. 安装`pytorch`
2.


# API
## 1. Based Artificial Neural Networks (ANN)

*def*   **SGML.ann(**

*`train_path`* = *str*, 

<sup> ↪ Load path for the training set. </sup>

*`test_path`* = *str*, 

<sup> ↪ Load path for the testing set. </sup>

*`feature_names`* = *list*, 

<sup> ↪ List containing feature names, such as *`['x1', 'x2', ...]`*. </sup>

*`lable_names`* = *list*, 

<sup> ↪ List containing label names, such as *`['y']`*. </sup>

*`solution_functions`* = *list*, 

<sup> ↪ List containing label names, such as *`[solution1, solution2, ...]`*. *`default=None`* </sup>

*`model_loadpath`* = *str*, 

<sup> ↪ Load path for the existing model. *`default=None`* </sup>

*`model_savepath`* = *str*, 

<sup> ↪ Save path for the model. *`default=None`* </sup>

*`hidden_layers`* = *list*, 

<sup> ↪ The hidden layer architecture, denoted as *`[4, 8, 2]`*, signifies the presence of three hidden layers with node counts of 4, 8, and 2, respectively. *`default=[8, 8]`* </sup>

*`activation_function`*  = *class*, 

<sup> ↪ The activation function—refer to the [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) for details. *`default=torch.nn.PReLU()`* </sup>

*`batch_size`* = *int*, 

<sup> ↪ The number of training samples used by the model during each parameter update. *`default=Total number of samples`* </sup>

*`criterion`* = *class*, 

<sup> ↪ The loss function—refer to the [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) for details. *`default=torch.nn.MSELoss()`* </sup>

*`optimizer`* = *class*, 

<sup> ↪ The optimizer—refer to the [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) for details.*`default=torch.optim.Adam()`* </sup>

*`learning_rate`*  = *float*, 

<sup> ↪ *`default=0.01`* </sup>

*`epochs`* = *int*, 

<sup> ↪ *`default=5000`* </sup>

**)**

*return* *class* <sub>(the trained model)</sub>, *ndarray* <sub>(labels of testing set)</sub>, *ndarray* <sub>(predicted results)</sub>





