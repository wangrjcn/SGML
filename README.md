# SGML

SGML...

# Preparation

1. 安装`pytorch`
2.


# API
## 1.1 Artificial Neural Networks (ANN)

*def*   SGML.ann(

*`train_path`* = *str*, ➡ os.path.join(current_dir, 'dataset/bending_train1.csv'),

*`test_path`* = *str*, ➡ os.path.join(current_dir, 'dataset/bending_test1.csv'),

*`feature_names`* = *list*, ➡ ['F', 'x'],

*`lable_names`* = *list*, ➡ ['y'],

*`solution_functions`* = *list*, ➡ [solution_1],

*`model_loadpath`* = *str*, ➡ None,

*`model_savepath`* = *str*, ➡ None,

*`hidden_layers`* = *list*, ➡ [8, 8],

*`activation_function`*  = *class*, ➡ None,

*`batch_size`* = *int*, ➡ None,

*`criterion`* = *class*, ➡ None,

*`optimizer`* = *class*, ➡ None,

*`learning_rate`*  = *float*, ➡ None,

*`epochs`* = *int*, ➡ 5000

)

*return* the trained model, labels of testing set, predicted results





