import os
import warnings
import numpy as np
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.util.tf_export import keras_export


# custom stopping criteria: relative improvement in percentage
class RelativeEarlyStopping(Callback):
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 min_perc_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=True):
        super(RelativeEarlyStopping, self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.min_perc_delta = min_perc_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf


    def on_epoch_end(self, epoch, logs=None):

        current = self.get_monitor_value(logs)
        if current is None:
            return

        print('\n\n')
        relative_change = (abs(self.best - current) / self.best) * 100
        print('----------------------------')
        print('relative change = {%.3f} percent' % relative_change)
        print('----------------------------')
        print('best = %s' % self.best)
        print('current = %s' % current)
        print('----------------------------')
        print('\n\n')


        if self.monitor_op(current, self.best * (1 - self.min_perc_delta)):  # only this line matters

            print('### still improves over criterion ###')

            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:

            print('### wait += 1 ###')

            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value