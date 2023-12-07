# SGML

SGML...

# Preparation

1. 安装`pytorch`
2.


# API
## 1.1 Artificial Neural Networks (ANN)

*def*   SGML.ann(

*`train_path`* = *str*, <sub> ➡ The path to the training set. </sub>

*`test_path`* = *str*, <sub> ➡ The path to the testing set. </sub>

*`feature_names`* = *list*, <sub> ➡ List containing feature names. Such as ['x<sub>1</sub>', 'x2', ...] </sub>

*`lable_names`* = *list*, <sub> ➡ ['y'], </sub>

*`solution_functions`* = *list*, <sub> ➡ [solution_1], </sub>

*`model_loadpath`* = *str*, <sub> ➡ None, </sub>

*`model_savepath`* = *str*, <sub> ➡ None, </sub>

*`hidden_layers`* = *list*, <sub> ➡ [8, 8], </sub>

*`activation_function`*  = *class*, <sub> ➡ None, </sub>

*`batch_size`* = *int*, <sub> ➡ None, </sub>

*`criterion`* = *class*, <sub> ➡ None, </sub>

*`optimizer`* = *class*, <sub> ➡ None, </sub>

*`learning_rate`*  = *float*, <sub> ➡ None, </sub>

*`epochs`* = *int*, <sub> ➡ 5000 </sub>

)

*return* the trained model, labels of testing set, predicted results





