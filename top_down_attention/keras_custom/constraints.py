import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]= "3"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python.keras.constraints import Constraint 
from tensorflow.keras import backend as K
from tensorflow.python.util.tf_export import keras_export


class Clip(Constraint):
    """
    custom constraint: limites w into [0, 1] or [epsilon, +inf]
    """
    def __call__(self, w):
        value = K.cast(K.greater_equal(w, K.epsilon()), K.floatx())
        w = w * value
        return w