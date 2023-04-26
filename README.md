# The Costs and Benefits of Goal-Directed Attention in Deep Convolutional Neural Networks
\[[Original paper](https://link.springer.com/article/10.1007/s42113-021-00098-y)\]

### Set up environment
`conda create --name myenv --file conda_env/attn_tf22_py37.txt`,
`pip install tensorflow-gpu==2.2.0` <br/>or `conda install tensorflow-gpu==2.2.0`

### Train models
* To learn corresponding attention weights, `python META_train --option attn_layer` <br/>
* To learn corresponding final layer weights, `python META_train --option last_layer`

### Evaluate models
* To evaluate trained models on tests and generate intermediate results, such as probabilities, `python META_eval --option eval --exp <num> --task <task code>`<br/>
* To plot the final results, `python META_eval --option plot --exp <num> --task <task code>`. <br/>
* `<num>` is a number among `1, 2` or `3`. `<task code>` is a string between `EXP` or `retrain` where `EXP` corresponds to models trained using attention layer and `retrain` corresponds to training the last layer without attention.

### Repo organisation
* Model definition, custom data generator, fitting function can be found in `top_down_attention/keras_custom/`
* Model training code can be found in `top_down_attention/TRAIN/`
* Model evaluation and results plotting code can be found in `top_down_attention/EVAL/`

### Attribution
```
@article{Luo2021TheNetworks,
    title = {{The costs and benefits of goal-directed attention in deep convolutional neural networks}},
    year = {2021},
    journal = {Computational Brain {\&} Behavior},
    author = {Luo, Xiaoliang and Roads, Brett D. and Love, Bradley C.},
    month = {2},
    pages = {1--18},
    url = {https://doi.org/10.1007/s42113-021-00098-y},
    doi = {10.1007/s42113-021-00098-y},
    issn = {23318422},
}
```
