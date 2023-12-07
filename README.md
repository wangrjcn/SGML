# SGML

SGML...

# Preparation

1. 安装`pytorch`
2.


# API
## 1.1 Artificial Neural Networks (ANN)

*def*   SGML.ann(

*`train_path`* = *str*, #0969DA os.path.join(current_dir, 'dataset/bending_train1.csv'),

*`test_path`* = *str*, os.path.join(current_dir, 'dataset/bending_test1.csv'),

*`feature_names`* = *list*, ['F', 'x'],

*`lable_names`* = ['y'],

*`solution_functions`* = [solution_1],

*`model_loadpath`* = None,

*`model_savepath`* = None,

*`hidden_layers`* = [8, 8],

*`activation_function`*  =None,

*`batch_size`* = None,

*`criterion`* = None,

*`optimizer`* = None,

*`learning_rate`*  =None,

*`epochs`* = 5000

)

*return* the trained model, labels of testing set, predicted results





