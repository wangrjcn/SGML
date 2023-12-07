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

*`feature_names`* = *list*, <sub> ➡ List containing feature names. Such as *`['x1', 'x2', ...]`*. </sub>

*`lable_names`* = *list*, <sub> ➡ List containing label names. Such as *`['y']`*. </sub>

*`solution_functions`* = *list*, <sub> ➡ List containing label names. Such as *`[solution1, solution2, ...]`*. </sub>

*`model_loadpath`* = *str*, <sub> ➡ Load path for the existing model. </sub>

*`model_savepath`* = *str*, <sub> ➡ Save path for the model. </sub>

*`hidden_layers`* = *list*, <sub> ➡ [8, 8], </sub>

*`activation_function`*  = *class*, <sub> ➡ None, </sub>

*`batch_size`* = *int*, <sub> ➡ None, </sub>

*`criterion`* = *class*, <sub> ➡ None, </sub>

*`optimizer`* = *class*, <sub> ➡ None, </sub>

*`learning_rate`*  = *float*, <sub> ➡ None, </sub>

*`epochs`* = *int*, <sub> ➡ 5000 </sub>

)

*return* the trained model, labels of testing set, predicted results





