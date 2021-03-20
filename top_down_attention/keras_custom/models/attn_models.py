import os
import numpy as np
import h5py

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Layer, Flatten, Reshape, \
    Dense, Dropout, Input, Multiply, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.python.ops import gen_math_ops
from tensorflow.keras import backend as K

from keras_custom.engine import training
from keras_custom import constraints
from keras_custom.layers.attention_model_layers \
    import FineTuneOutput, ExpandAttention, AttLayer_Filter_at_branch


def AttentionModel_FilterWise(num_categories, attention_mode, lr, opt=Adam(lr=0.0003)):
    WHERE_IS_ATTENTION = 'block4_pool'

    model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    layers = [l for l in model.layers]
    weights_and_biases_on_1k = model.layers[-1].get_weights()

    index = None
    for i, layer in enumerate(layers):
        if layer.name == WHERE_IS_ATTENTION:
            index = i
            break
    x = layers[index].output

    num_c = int(x.shape[3])  ## for later use
    HWC = int(x.shape[1] * x.shape[2] * x.shape[3])
    x_mother = Flatten(name='flatten_mother')(x)

    # above the same
    # --------------------------------------------------------------------------

    '''
    element-wise multiply shall happen here. alone with the other branch
    of the network with fixed (ones) input
    '''
    # the other branch:
    input_batch = Input(shape=(512,))

    # a layer takes a bunch of ones and multiply them by attention weights
    inShape = (None, 512)
    outdim = 512
    x_branch = AttLayer_Filter_at_branch(outdim, input_shape=inShape, name='att_layer_1')(input_batch)

    # some deterministic function expands layer output to match flattened layer
    # representation from the mother model (i.e. x = Flatten()(x))
    transformation = np.tile(np.identity(512, dtype='float32'), [1, 196])

    # --------------------------------------------------------------------------
    # x_expanded is actually weights to be multiplied by x
    inShape = (None, 512)
    outdim = 100352

    # HACK: mother branch output has a flattened shape, whereas here the attention branch the shape is in 512
    # HACK: which means, in order to merge the branches, one has to expand the attention branch output to the flattened shape
    # HACK: AND, most importantly, one has to make sure each of 512 value gets multiplied onto the same filter of the flattened output from the preceding layer of attention
    x_expanded = ExpandAttention(outdim, input_shape=inShape, name='expand_attention')(x_branch)

    branch_model = Model(inputs=input_batch, outputs=x_expanded)

    # --------------------------------------------------------------------------
    x_combined = Multiply()([x_mother, x_expanded])

    if WHERE_IS_ATTENTION == 'block4_pool':
        x = Reshape((14, 14, 512))(x_combined)
    for layer in layers[index+1:-1]:
        x = layer(x)  # x->(?, 4096)
    ############################################################################
    input_shape = x.shape
    outdim = len(num_categories)

    x = FineTuneOutput(outdim, input_shape=input_shape, name='fine_tune_output_1')(x)
    ############################################################################
    model = training.Model_custom(inputs=[model.input, branch_model.input], outputs=x)
    model.compile(
                  opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['acc'],
                  )

    ws = weights_and_biases_on_1k[0][:, num_categories]
    bs = weights_and_biases_on_1k[1][num_categories]
    sub_ws_bs = list([ws, bs])

    model.get_layer('fine_tune_output_1').set_weights(sub_ws_bs)

    # change weights on dot_product  layer  as transformation
    model.get_layer('expand_attention').set_weights([transformation])
    # plot_model(model, to_file='filter_wise_attention.png')
    # model.summary()
    return model


if __name__ == '__main__':
    pass
