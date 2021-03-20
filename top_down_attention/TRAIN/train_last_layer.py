import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import time

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

from keras_custom.generators.generator_wrapers import create_good_generator
from keras_custom.engine import training
from tensorflow.keras.applications.vgg16 import VGG16

from keras_custom import callbacks
from TRAIN.utils.data_utils import data_directory
from TRAIN.utils.saving_utils import save_fc2_weights


def train_model(wnid, index, model, 
               attention_mode, 
               description, 
               run, patience, lr, 
               importance, 
               subsample_rate, 
               imagenet_train, 
               batch_size, 
               epochs, float):
    """
    inputs:
    -------
        wnid: a list of wnids
        index: a list of fine tuning indices (consistent with network indices)
        model: a pre-defined model
        attention_mode: e.g. BiVGG-FILTER
        description: either a name of a grouping or individual category
        run: in case where there are multiple runs of the same model training
    """
    ############################################################################
    # callbacks
    relative_improve_as_percent = 0.1
    NAME = f'{description}_run{run}'
    tensorboard = TensorBoard(log_dir=f'log/cobb/{importance}/{NAME}')
    earlystopping = callbacks.RelativeEarlyStopping(monitor='val_loss',
                                              min_perc_delta=relative_improve_as_percent/100,
                                              patience=patience,
                                              verbose=2,
                                              mode='min',
                                              restore_best_weights=True
                                              )

    fine_tune_categories = index
    # ==========================================================================
    if importance == 0:

        classes = list(wnid)

        focus_classes = None
        subsample_rate = 1
        train_class_weight = None
        val_class_weight = None
        validation_freq = 1

        # TODO: figure this out
        # validation_freq = [500] + list(range(502, 1002, 2))
    # ==========================================================================
    if importance != 0:
        '''
        e.g.
            Given subsample_rate == 0.1,
            if importance == 1:
                train: target weight = 1
                        non-trg weight = 10
                val: target weight = 1
                        non-trg weight = 1
            if importance == 1/999:
                train: target weight = 1
                        non-trg weight = (1/999) * 10
                val: target weight = 1
                        non-trg weight = 1/999
            if importance == 0:
                it's more efficicent to seprate 0,
                otherwise we still loading in 10% non-target train although
                we weight them by 0, it's not as fast as not loading them
                at all in the first place.
        '''
        focus_classes = wnid
        classes = None

        # WARNING:  hard coded skip may not apply to other models
        validation_freq = [6] + list(range(8, 502, 2))
        # validation_freq = 1

        train_class_weight = {}
        val_class_weight = {}
        for i in range(1000):
            if i in fine_tune_categories:
                train_class_weight[i] = 1
                val_class_weight[i] = 1
            else:
                '''
                e.g. if set importance = 1/999, we sample 10%, so the actual weighting on non-target should be 10/999
                        for training set, as in:
                            importance = actual_weighting * subsample_rate
                    (OR)    actual_weighting = importance / subsample_rate
                '''
                train_class_weight[i] = importance / subsample_rate
                val_class_weight[i] = importance
        print('CHECK: train_class_weight non-target = %s' % (importance / subsample_rate))
        print('CHECK: val_class_weight non-target = %s' % (importance))
        print('\n\n')
    # ==========================================================================
    # --------------------------------------------------------------------------
    train_generator, train_steps = create_good_generator(
                                                         directory=imagenet_train,
                                                         classes=classes,
                                                         batch_size=batch_size,
                                                         seed=42,
                                                         shuffle=True,
                                                         subset='training',
                                                         validation_split=0.1,
                                                         class_mode='sparse',
                                                         target_size=(224, 224),
                                                         preprocessing_function=preprocess_input,
                                                         horizontal_flip=True,
                                                         AlexNetAug=False,
                                                         focus_classes=focus_classes,  #NOTE
                                                         subsample_rate=subsample_rate)

    val_generator, val_steps = create_good_generator(directory=imagenet_train,  # 10% train as val
                                                     classes=classes,
                                                     batch_size=batch_size,
                                                     seed=42,
                                                     shuffle=True,
                                                     subset='validation',
                                                     validation_split=0.1,
                                                     class_mode='sparse',
                                                     target_size=(224, 224),
                                                     preprocessing_function=preprocess_input,
                                                     horizontal_flip=False,
                                                     AlexNetAug=False,
                                                     focus_classes=focus_classes,
                                                     subsample_rate=1)
    print('training.......')
    if validation_freq == 1:
        print('CHECK: validation_freq is 1')
    else:
        print('CHECK: validation_freq length = {length}'.format(length=len(validation_freq)))
    model.fit_generator_custom(
                                train_generator,
                                steps_per_epoch=train_steps,
                                epochs=epochs,
                                validation_data=val_generator,
                                validation_steps=val_steps,
                                verbose=1,
                                callbacks=[tensorboard, earlystopping],
                                max_queue_size=40,
                                workers=3,
                                use_multiprocessing=False,
                                train_class_weight=train_class_weight,
                                val_class_weight=val_class_weight,
                                validation_freq=validation_freq)

    return model


def main(attention_mode, patience, lr, importance, subsample_rate, imagenet_train, batch_size, epochs, float):

    groupings = ['imagenetA']
    num_categories = list(range(1000))  # to control network output dim(1000)
    for group in groupings:
        print('group = %s ***************' % group)

        df = pd.read_csv('groupings-csv/%s_Imagenet.csv' % group, usecols=['wnid', 'idx', 'description'])
        sorted_indices = np.argsort([i for i in df['wnid']])[150:]

        group_wnids = np.array([i for i in df['wnid']])[sorted_indices]
        group_indices = np.array([int(i) for i in df['idx']])[sorted_indices]
        group_descriptions = np.array([i for i in df['description']])[sorted_indices]

        # get individual category from each group
        for i in range(len(group_wnids)):
            start_time = time.time()

            auto_index = sorted_indices[i]
            wnid = [group_wnids[i]]
            index = [group_indices[i]]
            description = group_descriptions[i]
            
            print(f'index={index}, wnid={wnid}, description={description}')
            print('============================================================')
            for run in [1]:

                opt = Adam(lr=0.0003)
                opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

                ###
                model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
                for layer in model.layers[:-1]:
                    # only FC2 trainable.
                    layer.trainable = False
                model = training.Model_custom(inputs=model.input, outputs=model.output)
                model.compile(
                  opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['acc', 'sparse_top_k_categorical_accuracy'],
                  )
                model.summary()

                # training begins.
                model = train_model(wnid, index, model, attention_mode, description, run, patience, lr, importance, subsample_rate, imagenet_train, batch_size, epochs, float)
                save_fc2_weights(model, description, importance, run)
                K.clear_session()

                end_time = (time.time() - start_time) / (3600.)
                print('use time = {end_time} hrs'.format(end_time=end_time))


def execute():
    float = 16
    patience = 1  # if using validation_freq, orig: 1
    lr = 0.0003   # orig: 0.0003
    attention_mode = 'BiVGG-FILTER'
    imagenet_train = data_directory()
    batch_size = 16
    epochs = 500
    #####################
    importance = 1/999
    subsample_rate = 0.1
    #####################
    main(attention_mode=attention_mode,
         subsample_rate=subsample_rate,
         patience=patience,
         lr=lr,
         importance=importance,
         imagenet_train=imagenet_train,
         batch_size=batch_size,
         epochs=epochs,
         float=float)