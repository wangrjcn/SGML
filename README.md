# Solution-guided Machine Learning (SGML)

SGML...

## Preparation

1. 安装`pytorch`
2.

## API

💡 The solution function will be returned with the data type of a *function* for the given solution. Users can assign it a name for subsequent guidance in machine learning. 

💡 All models are presently implemented as functions, each returning the trained model along with both the labels and predictions for the testing set. 

💡 When the parameters are set to *`None`* or left unspecified, the default values for those parameters will be applied. 

### 1. Solution Function

*def* **SGML.create_solution_function(**

*`expression`* = *str*, 

<sup> Solution expression, such as *`'a**3+2*b+1'`*. </sup> 
                                           
*`variables`* = *list*, 

<sup> List of variables included in the solution，such as *`['a', 'b']`*. </sup>

**)**

*return* *function*

### 2. Artificial Neural Network-Based Model

*def* **SGML.ann(**

*`train_path`* = *str*, 

<sup> The file path for loading the training set. </sup>

*`test_path`* = *str*, 

<sup> The file path for loading the testing set. </sup>

*`feature_names`* = *list*, 

<sup> List containing feature names, such as *`['x1', 'x2', ...]`*. </sup>

*`lable_names`* = *list*, 

<sup> List containing label names, such as *`['y']`*. </sup>

*`solution_functions`* = *list*, 

<sup> List containing solution function names, such as *`[solution1, solution2, ...]`*. *`default=None`* </sup>

*`model_loadpath`* = *str*, 

<sup> The file path for the existing model. *`default=None`* </sup>

*`model_savepath`* = *str*, 

<sup> Path to save the model. *`default=None`* </sup>

*`hidden_layers`* = *list*, 

<sup> The hidden layer architecture, denoted as *`[4, 8, 2]`*, signifies the presence of three hidden layers with node counts of 4, 8, and 2, respectively. *`default=[8, 8]`* </sup>

*`activation_function`*  = *class*, 

<sup> The activation function—refer to the [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) for details. *`default=torch.nn.PReLU()`* </sup>

*`batch_size`* = *int*, 

<sup> The number of training samples used by the model during each parameter update. *`default=Total number of samples`* </sup>

*`criterion`* = *class*, 

<sup> The loss function—refer to the [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) for details. *`default=torch.nn.MSELoss()`* </sup>

*`optimizer`* = *class*, 

<sup> The optimizer—refer to the [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) for details.*`default=torch.optim.Adam()`* </sup>

*`learning_rate`*  = *float*, 

<sup> *`default=0.01`* </sup>

*`epochs`* = *int*

<sup> *`default=5000`* </sup>

**)**

*return* *class*, *ndarray*, *ndarray*

### 3. Support Vector Regression-Based Model






