import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python.keras.regularizers import Regularizer 
from tensorflow.keras import backend as K
from tensorflow.python.util.tf_export import keras_export


class L1_C(Regularizer):
    """
    custom l1 where now |w| becomes |w-c| so that model weights will be
    forced to be around C instead of 0
    """
    def __init__(self, l1=0., c=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.c = K.cast_to_floatx(c)

    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(x - self.c)) ###

        return regularization

    def get_config(self):
        return {'l1': float(self.l1), 'c': float(self.c)}

def l1_c(l=0.01, c=1):
    return L1_C(l1=l, c=c)