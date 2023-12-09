# 1 Solution-guided Machine Learning (SGML)

SGML...

<br>
<br>
<br>

# 2 Preparation

1. 安装`pytorch`
2.

<br>
<br>
<br>

# 3 API

> [!IMPORTANT]
> - The solution function will be returned with the data type of a *function* for the given solution. Users can assign it a name for subsequent guidance in machine learning.
>
> - All models are presently implemented as functions, each returning the trained model along with both the labels and predictions for the testing set.
>
> - When the parameters are set to *`None`* or left unspecified, the default values for those parameters will be applied. 

<br>

## 3.1 Solution Function

*def* **SGML.create_solution_function(**

*`expression`* = *str*, 
                                           
*`variables`* = *list* 

**)**

*return* *function*

> [!TIP]
> *`expression`* : Solution expression, such as *`'a**3+2*b+1'`*.
>                                          
> *`variables`* : List of variables included in the solution，such as *`['a', 'b']`*.

<br>

### 3.2 Artificial Neural Network-Based Model

*def* **SGML.ann(**

*`train_path`* = *str*, 

*`test_path`* = *str*, 

*`feature_names`* = *list*, 

*`lable_names`* = *list*, 

*`solution_functions`* = *list*, 

*`model_loadpath`* = *str*, 

*`model_savepath`* = *str*, 

*`hidden_layers`* = *list*, 

*`activation_function`*  = *class*, 

*`batch_size`* = *int*, 

*`criterion`* = *class*, 

*`optimizer`* = *class*, 

*`learning_rate`*  = *float*, 

*`epochs`*  = *int*, 

**)**

*return* *class*, *ndarray*, *ndarray*

> [!TIP]
> *`train_path`* : The file path for loading the training set. 
>
>*`test_path`* : The file path for loading the testing set. 
>
>*`feature_names`* : List containing feature names, such as *`['x1', 'x2', ...]`*. 
>
>*`lable_names`* : List containing label names, such as *`['y']`*. 
>
>*`solution_functions`* : List containing solution function names, such as *`[solution1, solution2, ...]`*. *`default=None`* 
>
>*`model_loadpath`* : The file path for the existing model. *`default=None`* 
>
>*`model_savepath`* : Path to save the model. *`default=None`* 
>
>*`hidden_layers`* : The hidden layer architecture, denoted as *`[4, 8, 2]`*, signifies the presence of three hidden layers with node counts of 4, 8, and 2, respectively. *`default=[8, 8]`* 
>
>*`activation_function`* : The activation function—refer to [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) for details. *`default=torch.nn.PReLU()`* 
>
>*`batch_size`* : The number of training samples used by the model during each parameter update. *`default=Total number of samples`* 
>
>*`criterion`* : The loss function—refer to [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) for details. *`default=torch.nn.MSELoss()`* 
>
>*`optimizer`* : The optimizer—refer to [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) for details.*`default=torch.optim.Adam()`* 
>
>*`learning_rate`* : *`default=0.01`* 
>
>*`epochs`* : *`default=5000`* 

<br>

## 3.3 Support Vector Regression-Based Model

*def* **SGML.svr(**

*`train_path`* = *str*, 

*`test_path`* = *str*, 

*`feature_names`* = *list*, 

*`lable_names`* = *list*, 

*`solution_functions`* = *list*, 

*`model_loadpath`* = *str*, 

*`model_savepath`* = *str*, 

*`kernel`* = *str*, 

<sup> Refer to [sklearn.svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) for detailed information, and the same applies to the following parameters. *`default='linear'`* </sup>

*`degree`* = *int*, 

<sup> *`default=3`* </sup>

*`gamma`* = *str* or *float*, 

<sup> *`default='scale'`* </sup>

*`coef0`* = *float*, 

<sup> *`default=0.0`* </sup>

*`tol`* = *float*, 

<sup> *`default=1e-3`* </sup>

*`C`* = *float*, 

<sup> *`default=1.0`* </sup>

*`epsilon`* = *float*, 

<sup> *`default=0.1`* </sup>

*`shrinking`* = *bool*, 

<sup> *`default=True`* </sup>

*`cache_size`* = *float*, 

<sup> *`default=200`* </sup>

*`verbose`* = *bool*, 

<sup> *`default=False`* </sup>

*`max_iter`* = *int* 

<sup> *`default=-1`* </sup>

**)**

*return* *class*, *ndarray*, *ndarray*

> [!TIP]
> The API reference for the parameters *`train_path`*, *`test_path`*, *`feature_names`*, *`lable_names`*, *`solution_functions`*, *`model_loadpath`*, and *`model_savepath`* can be found in Section 3.2.
>
> *`kernel`* : Refer to [sklearn.svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) for detailed information, and the same applies to the following parameters. *`default='linear'`*
>
> *`degree`* : *`default=3`* 
>
>*`gamma`* : *`default='scale'`* 
>
>*`coef0`* : *`default=0.0`* 
>
>*`tol`* : *`default=1e-3`* 
>
>*`C`* : *`default=1.0`* 
>
>*`epsilon`* : *`default=0.1`* 
>
>*`shrinking`* : *`default=True`* 
>
>*`cache_size`* : *`default=200`* 
>
>*`verbose`* : *`default=False`* 
>
>*`max_iter`* : *`default=-1`*









