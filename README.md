# SGML

SGML...

# Preparation

1. 安装`pytorch`
2.


# API
## 1.1 Artificial Neural Networks (ANN)

*def*   SGML.ann(

*`train_path`* = *str*, <sub> ➡ Load path for the training set. </sub>

*`test_path`* = *str*, <sub> ➡ Load path for the testing set. </sub>

*`feature_names`* = *list*, <sub> ➡ List containing feature names, such as *`['x1', 'x2', ...]`*. </sub>

*`lable_names`* = *list*, <sub> ➡ List containing label names, such as *`['y']`*. </sub>

*`solution_functions`* = *list*, <sub> ➡ List containing label names, such as *`[solution1, solution2, ...]`*. *`default=None`* </sub>

*`model_loadpath`* = *str*, <sub> ➡ Load path for the existing model. *`default=None`* </sub>

*`model_savepath`* = *str*, <sub> ➡ Save path for the model. *`default=None`* </sub>

*`hidden_layers`* = *list*, <sub> ➡ The hidden layer architecture, denoted as *`[4, 8, 2]`*, signifies the presence of three hidden layers with node counts of 4, 8, and 2, respectively. </sub>

*`activation_function`*  = *class*, <sub> ➡ None, </sub>

*`batch_size`* = *int*, <sub> ➡ None, </sub>

*`criterion`* = *class*, <sub> ➡ None, </sub>

*`optimizer`* = *class*, <sub> ➡ None, </sub>

*`learning_rate`*  = *float*, <sub> ➡ None, </sub>

*`epochs`* = *int*, <sub> ➡ 5000 </sub>

)

*return* the trained model, labels of testing set, predicted results





