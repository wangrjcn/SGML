# SGML

SGML...

# Preparation

1. 安装`pytorch`
2.


# API
## 1.1 Artificial Neural Networks (ANN)

*def*   SGML.ann(

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

<sup> ↪ The hidden layer architecture, denoted as *`[4, 8, 2]`*, signifies the presence of three hidden layers with node counts of 4, 8, and 2, respectively. </sup>

*`activation_function`*  = *class*, 

<sup> ↪ None, </sup>

*`batch_size`* = *int*, 

<sup> ↪ None, </sup>

*`criterion`* = *class*, 

<sup> ↪ None, </sup>

*`optimizer`* = *class*, 

<sup> ↪ None, </sup>

*`learning_rate`*  = *float*, 

<sup> ↪ None, </sup>

*`epochs`* = *int*, 

<sup> ↪ 5000 </sup>

)

*return* the trained model, labels of testing set, predicted results





