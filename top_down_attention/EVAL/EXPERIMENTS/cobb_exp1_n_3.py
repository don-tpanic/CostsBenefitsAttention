import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pickle
import scipy.stats as stats
import pandas as pd
import ast
import time
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import preprocess_input

from keras_custom.generators.generator_wrapers import create_good_generator
from tensorflow.keras.applications.vgg16 import VGG16
from keras_custom.models.attn_models import AttentionModel_FilterWise


def per_category_probs(model, 
                      description, 
                      importance, 
                      exp_num,
                      target_wnid, 
                      nontarget_wnid, 
                      true_label,
                      random_seed, 
                      eval_data_dir,
                      probs_dir,
                      task):
    test_generator, test_steps = create_good_generator(
                                                    directory=eval_data_dir,
                                                    classes=target_wnid,
                                                    batch_size=16,
                                                    seed=42,
                                                    shuffle=False,
                                                    subset=None,
                                                    validation_split=0.0,
                                                    class_mode='sparse',
                                                    target_size=(224, 224),
                                                    preprocessing_function=preprocess_input,
                                                    horizontal_flip=False,
                                                    AlexNetAug=False,
                                                    focus_classes=None,
                                                    subsample_rate=1)

    test_generator_nontrg, test_steps_nontrg = create_good_generator(
                                                    directory=eval_data_dir,
                                                    classes=nontarget_wnid,
                                                    batch_size=16,
                                                    seed=42,
                                                    shuffle=False,
                                                    subset=None,
                                                    validation_split=0.0,
                                                    class_mode='sparse',
                                                    target_size=(224, 224),
                                                    preprocessing_function=preprocess_input,
                                                    horizontal_flip=False,
                                                    AlexNetAug=False,
                                                    focus_classes=None,
                                                    subsample_rate=1)
    if task == 'EXP':
        if importance != 1:
            attention_weights = f'attention_weights/block4_pool/attention={importance}/[hybrid]_{description}-imp{importance}-run1-float16.npy'
        else:
            print('CHECK: loading uniform attn ws..')
            attention_weights = 'attention_weights/block4_pool/basemodel/attention=1/[hybrid]_albatross-imp1-run1-float16.npy'
        model.get_layer('att_layer_1').set_weights([np.load(attention_weights)])
        print('loaded attenion weights.')

    elif task == 'retrain':
        with open(f'attention_weights/cobb_lastLayer/attention={importance}/[hybrid]_{description}-imp{importance}-run1.pkl', 'rb') as f:
            ws = pickle.load(f)
            model.get_layer('predictions').set_weights([ws[0], ws[1]])
        print('loaded retrained weights.')

    # (n, 1000), posotive examples.
    probs_pos = model.predict_generator(test_generator, test_steps, verbose=1, workers=3)
    np.save(f'exp_results/cobb/exp{exp_num}/{probs_dir}/{description}_imp{importance}_truePos.npy', probs_pos)
    num_pos = len(probs_pos)

    # negative examples.
    probs_neg = model.predict_generator(test_generator_nontrg, test_steps_nontrg, verbose=1, workers=3)
    # first we sample the same amount of images == num_pos
    np.random.seed(true_label + random_seed)
    sampled_indices = np.random.choice(np.arange(len(probs_neg)), size=num_pos, replace=False)
    probs_neg = probs_neg[sampled_indices, :]
    num_neg = len(probs_neg)
    assert num_neg == num_pos, "num of neg does not equal num of pos!"
    np.save(f'exp_results/cobb/exp{exp_num}/{probs_dir}/{description}_imp{importance}_falsePos.npy', probs_neg)
    K.clear_session()


def all_categories_probs(model, importance, exp_num, random_seed, eval_data_dir, probs_dir, task):
    df = pd.read_csv('groupings-csv/imagenetA_Imagenet.csv', usecols=['wnid', 'idx', 'description'])
    sorted_indices = np.argsort([i for i in df['wnid']])
    group_classes = np.array([i for i in df['wnid']])[sorted_indices]
    group_indices = np.array([int(i) for i in df['idx']])[sorted_indices]
    group_descriptions = np.array([i for i in df['description']])[sorted_indices]

    # check dir saving probs exists:
    probs_full_path = f'exp_results/cobb/exp{exp_num}/{probs_dir}'
    if not os.path.exists(probs_full_path):
        print(f'{probs_full_path} is now created.')
        os.mkdir(probs_full_path)
    else:
        print(f'{probs_full_path} already exists.')

    # use one attention model at a time
    for i in range(len(group_classes)):
        wnid = [group_classes[i]]  # wnid
        index = group_indices[i]  # network index aka true_label
        description = group_descriptions[i]  # class description

        nontarget_wnid = sorted([i for i in group_classes if i not in wnid])
        print('CHECK: num of non-targets = [%s]' % len(nontarget_wnid))
        print('wnid = [%s], index = [%s], description = [%s]' % (wnid, index, description))

        # check if this class has already had saved probs for neg,
        # if not we run a fresh start to get both pos and neg, otherwise we skip to the next class.
        if not os.path.exists(os.path.join(probs_full_path, f'{description}_imp{importance}_falsePos.npy')):
            per_category_probs(model=model,
                                description=description,
                                importance=importance,
                                exp_num=exp_num,
                                target_wnid=wnid,
                                nontarget_wnid=nontarget_wnid,
                                true_label=index,
                                random_seed=random_seed,
                                eval_data_dir=eval_data_dir,
                                probs_dir=probs_dir,
                                task=task)
        else:
            print(f'skip [{description}] because it exists.')
            continue


def execute(exp_num, task):
    # task = 'EXP'  # 'EXP'or 'retrain'
    random_seed = 42
    # -------------------------
    if task == 'EXP':
        probs_dir = f'probs{random_seed}'
        importances = [0.001, 0.5, 0, 1, 1/999000]
        model = AttentionModel_FilterWise(num_categories=list(range(1000)), attention_mode='BiVGG-FILTER', lr=0.0003)
    elif task == 'retrain':
        probs_dir = f'probs_retrain{random_seed}'
        importances = [0.001, 1]
        model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    model.compile(Adam(lr=0.0003), loss='sparse_categorical_crossentropy', metrics=['sparse_top_k_categorical_accuracy'])

    if exp_num == '1' or exp_num == '2':
        eval_data_dir = f'/mnt/fast-data17/datasets/ILSVRC/2012/clsloc/val_white/'
    elif exp_num == '3':
        eval_data_dir = '/home/oem/ken/projects/CostsBenefitsAttention_CBB/top_down_attention/data/imagenet-a'
    
    for importance in importances:
        all_categories_probs(model=model,
                             importance=importance,
                             exp_num=exp_num,
                             random_seed=random_seed,
                             eval_data_dir=eval_data_dir, 
                             probs_dir=probs_dir,
                             task=task)    
