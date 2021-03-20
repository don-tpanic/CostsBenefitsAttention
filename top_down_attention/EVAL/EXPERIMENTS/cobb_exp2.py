import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pickle
import scipy.stats as stats
import pandas as pd
import ast
import time
import itertools
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import preprocess_input

from keras_custom.generators.generator_wrapers import create_good_generator
from tensorflow.keras.applications.vgg16 import VGG16
from keras_custom.models.attn_models import AttentionModel_FilterWise


def blending(gen1, gen2, label=1, alpha1=0.5, alpha2=0.5):
    """
    given two gens, blend batch images using some alpha value and return.
    the labels of the new generator is either the labels of gen1 or gen2
    depending on the task.
    """
    while True:
        # remember batch_x_1 has two inputs [img, ones]
        batch_x_1, batch_y_1 = next(gen1)
        batch_x_2, batch_y_2 = next(gen2)

        # create actual alpha blending batch per batch on fly
        x1 = batch_x_1[0]
        x2 = batch_x_2[0]
        blended_batch_x = x1 * alpha1 + x2 * alpha2
        second_input = batch_x_1[1]
        if label == 1:
            batch_y = batch_y_1
        elif label == 2:
            batch_y = batch_y_2
        yield ([blended_batch_x, second_input], batch_y)


def all_categories_probs(model, importance, exp_num, random_seed, eval_data_dir, probs_dir, task):
    start_time = time.time()

    df = pd.read_csv('groupings-csv/imagenetA_Imagenet.csv', usecols=['wnid', 'idx', 'description'])
    sorted_indices = np.argsort([i for i in df['wnid']])
    group_wnids = np.array([i for i in df['wnid']])[sorted_indices]
    group_indices = np.array([int(i) for i in df['idx']])[sorted_indices]
    group_descriptions = np.array([i for i in df['description']])[sorted_indices]

    # check dir saving probs exists:
    probs_full_path = f'exp_results/cobb/exp{exp_num}/{probs_dir}'
    if not os.path.exists(probs_full_path):
        print(f'{probs_full_path} is now created.')
        os.mkdir(probs_full_path)
    else:
        print(f'{probs_full_path} already exists.')

    attention_wnids = group_wnids
    attention_indices = group_indices
    attention_desciptions = group_descriptions

    # all possible pairs of blended classes - we don't worry about match/mismatch here
    # combos are in sort order,
    # make sure wnid and index match because the boolean we use later are based on index due to limitation in np.equal
    wnid_pairs = np.array(list(itertools.combinations(group_wnids, 2)))
    index_pairs = np.array(list(itertools.combinations(group_indices, 2)))

    # one attn model at a time for both hit and fas cases.
    for a in range(len(attention_wnids)):

        attention_wnid = attention_wnids[a]
        attention_index = attention_indices[a]
        attention = attention_desciptions[a]

        # (19900, 2) boolean
        locs = np.equal(index_pairs, [attention_index])
        # binary list of match and mismatch indices
        loc_idx = np.sum(locs, axis=1)
        # grab wnid pairs based on loc_idx_bool
        match_loc_idx_bool = np.equal(loc_idx, [1])
        mismatch_loc_idx_bool = np.logical_not(match_loc_idx_bool)
        match_wnid_pairs = wnid_pairs[match_loc_idx_bool, :]
        mismatch_wnid_pairs = wnid_pairs[mismatch_loc_idx_bool, :]

        # check if this attn model has gone through hit or fas test.
        # if truePos is not done, we do both
        # if truePos is done, but falsePos isn't, we do only falsePos
        if task == 'EXP':
            if importance != 1:
                attention_weights = f'attention_weights/block4_pool/attention={importance}/[hybrid]_{attention}-imp{importance}-run1-float16.npy'
            else:
                print('CHECK: loading uniform attn ws..')
                attention_weights = 'attention_weights/block4_pool/basemodel/attention=1/[hybrid]_albatross-imp1-run1-float16.npy'
            model.get_layer('att_layer_1').set_weights([np.load(attention_weights)])
            print('loaded attenion weights.')

        elif task == 'retrain':
            with open(f'attention_weights/cobb_lastLayer/attention={importance}/[hybrid]_{attention}-imp{importance}-run1.pkl', 'rb') as f:
                ws = pickle.load(f)
                model.get_layer('predictions').set_weights([ws[0], ws[1]])
            print('loaded retrained weights.')

        if not os.path.exists(f'exp_results/cobb/exp{exp_num}/{probs_dir}/{attention}_imp{importance}_truePos.npy'):
            match_prob_allPairs_oneAttn(match_wnid_pairs=match_wnid_pairs,
                                        attention=attention,
                                        attention_index=attention_index,
                                        attention_wnid=attention_wnid,
                                        importance=importance,
                                        exp_num=exp_num,
                                        model=model,
                                        eval_data_dir=eval_data_dir,
                                        probs_dir=probs_dir,
                                        )
            hallu_prob_allPairs_oneAttn(match_wnid_pairs=match_wnid_pairs, 
                                        mismatch_wnid_pairs=mismatch_wnid_pairs,
                                        attention=attention,
                                        attention_index=attention_index,
                                        importance=importance,
                                        exp_num=exp_num,
                                        model=model,
                                        eval_data_dir=eval_data_dir,
                                        random_seed=random_seed,
                                        probs_dir=probs_dir,
                                        )
        else:
            print(f'-- skipping -- [{attention}] truePos')
            if not os.path.exists(f'exp_results/cobb/exp{exp_num}/{probs_dir}/{attention}_imp{importance}_falsePos.npy'):
                hallu_prob_allPairs_oneAttn(match_wnid_pairs=match_wnid_pairs, 
                                            mismatch_wnid_pairs=mismatch_wnid_pairs,
                                            attention=attention,
                                            attention_index=attention_index,
                                            importance=importance,
                                            exp_num=exp_num,
                                            model=model,
                                            eval_data_dir=eval_data_dir,
                                            random_seed=random_seed,
                                            probs_dir=probs_dir,
                                            )
            else:
                print(f'-- skipping -- [{attention}] falsePos')
                continue

            
def match_prob_allPairs_oneAttn(match_wnid_pairs,
                                attention,
                                attention_index,
                                attention_wnid,
                                exp_num,
                                importance,
                                model,
                                eval_data_dir,
                                probs_dir,
                                ):
    """
    Save one attn model's correponding probs of all match pairs.
    """
    per_attention_probs_pos = np.zeros((1,1000))
    print('per_attention_probs_pos.shape=', per_attention_probs_pos.shape)
    for i in range(match_wnid_pairs.shape[0]):
        wnid_1 = [match_wnid_pairs[i][0]]
        wnid_2 = [match_wnid_pairs[i][1]]

        if attention_wnid in wnid_1:
            num_pos, probs_pos = match_prob_onePair_oneAttn(
                                            attention_index=attention_index,
                                            importance=importance,
                                            model=model,
                                            wnid_1=wnid_1,
                                            wnid_2=wnid_2,
                                            label=1,
                                            eval_data_dir=eval_data_dir,
                                            )
        else:
            num_pos, probs_pos = match_prob_onePair_oneAttn(
                                            attention_index=attention_index,
                                            importance=importance,
                                            model=model,
                                            wnid_1=wnid_1,
                                            wnid_2=wnid_2,
                                            label=2,
                                            eval_data_dir=eval_data_dir,
                                            )
        # collect probs for one match pair at a time
        per_attention_probs_pos = np.vstack((per_attention_probs_pos, probs_pos))
    # save pos probs for an entire attention model across all match cases.
    per_attention_probs_pos = per_attention_probs_pos[1:, :] # HACK: remove first row.
    np.save(f'exp_results/cobb/exp{exp_num}/{probs_dir}/{attention}_imp{importance}_truePos.npy', per_attention_probs_pos)


def hallu_prob_allPairs_oneAttn(match_wnid_pairs, 
                                mismatch_wnid_pairs,
                                attention,
                                attention_index,
                                importance,
                                exp_num,
                                model,
                                eval_data_dir,
                                random_seed,
                                probs_dir,
                                ):
    """
    Save one attn model's correponding probs of all hallu pairs. 
    """
    per_attention_probs_neg = np.zeros((1, 1000))
    np.random.seed(attention_index + random_seed)
    # sample 199 from all 19701 cases
    mismatch_sampled_idx = np.random.choice(range(mismatch_wnid_pairs.shape[0]), size=match_wnid_pairs.shape[0], replace=False)
    mismatch_pairs_sampled = mismatch_wnid_pairs[mismatch_sampled_idx, :]
    for j in range(mismatch_pairs_sampled.shape[0]):
        mismatch_wnid_1 = [mismatch_pairs_sampled[j][0]]
        mismatch_wnid_2 = [mismatch_pairs_sampled[j][1]]

        probs_neg= hallu_prob_onePair_oneAttn(
                                importance=importance,
                                model=model,
                                wnid_1=mismatch_wnid_1,
                                wnid_2=mismatch_wnid_2,
                                attention_index=attention_index,
                                eval_data_dir=eval_data_dir,
                                )
        per_attention_probs_neg = np.vstack((per_attention_probs_neg, probs_neg))
    per_attention_probs_neg = per_attention_probs_neg[1:, :]
    np.save(f'exp_results/cobb/exp{exp_num}/{probs_dir}/{attention}_imp{importance}_falsePos.npy', per_attention_probs_neg)


def match_prob_onePair_oneAttn(attention_index, importance, model, wnid_1, wnid_2, label, eval_data_dir):
    """
    Compute prob for one match pair within one attn model.
    """
    batch_size = 16
    test_gen_1, test_steps = create_good_generator(directory=eval_data_dir,
                                                   classes=wnid_1,
                                                   batch_size=batch_size,
                                                   seed=42,
                                                   shuffle=True,
                                                   subset=None,
                                                   validation_split=0,
                                                   class_mode='sparse',
                                                   target_size=(224, 224),
                                                   preprocessing_function=preprocess_input,
                                                   horizontal_flip=False,
                                                   AlexNetAug=False,
                                                   focus_classes=wnid_1,
                                                   subsample_rate=1
                                                   )
    test_gen_2, _ = create_good_generator(directory=eval_data_dir,
                                          classes=wnid_2,
                                          batch_size=batch_size,
                                          seed=42,
                                          shuffle=True,
                                          subset=None,
                                          validation_split=0,
                                          class_mode='sparse',
                                          target_size=(224, 224),
                                          preprocessing_function=preprocess_input,
                                          horizontal_flip=False,
                                          AlexNetAug=False,
                                          focus_classes=wnid_2,
                                          subsample_rate=1
                                          )

    test_steps = 2
    blend_gen = blending(test_gen_1, test_gen_2, label=label)

    # (n, 1000)
    probs_pos = model.predict_generator(blend_gen, test_steps, verbose=1, workers=1)
    K.clear_session()
    return len(probs_pos), probs_pos
    

def hallu_prob_onePair_oneAttn(importance, model, wnid_1, wnid_2, attention_index, eval_data_dir):
    batch_size = 16
    test_gen_1, test_steps = create_good_generator(directory=eval_data_dir,
                                                   classes=wnid_1,
                                                   batch_size=batch_size,
                                                   seed=42,
                                                   shuffle=True,
                                                   subset=None,
                                                   validation_split=0.0,
                                                   class_mode='sparse',
                                                   target_size=(224, 224),
                                                   preprocessing_function=preprocess_input,
                                                   horizontal_flip=False,
                                                   AlexNetAug=False,
                                                   focus_classes=None,
                                                   subsample_rate=1
                                                   )
    test_gen_2, _ = create_good_generator(directory=eval_data_dir,
                                          classes=wnid_2,
                                          batch_size=batch_size,
                                          seed=42,
                                          shuffle=True,
                                          subset=None,
                                          validation_split=0.0,
                                          class_mode='sparse',
                                          target_size=(224, 224),
                                          preprocessing_function=preprocess_input,
                                          horizontal_flip=False,
                                          AlexNetAug=False,
                                          focus_classes=None,
                                          subsample_rate=1
                                          )

    # blend images generators
    # due to focus on attention, no need for two blenders
    test_steps = 2
    blend_gen = blending(test_gen_1, test_gen_2, label=1)

    # (n, 1000)
    probs_neg = model.predict_generator(blend_gen, test_steps, verbose=2, workers=1)
    K.clear_session()
    return probs_neg


def execute(exp_num, task):
    # task = 'EXP'  # 'EXP'or 'retrain'
    random_seed = 42
    # exp_num = 2
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
