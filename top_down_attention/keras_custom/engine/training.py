import os
import warnings
import numpy as np
import pandas as pd

from tensorflow.keras import backend as K
from tensorflow.keras import Model

from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import monitoring

from tensorflow.python.keras.engine import training_utils  #, training_v2_utils, 
from keras_custom.engine import training_generator

# _keras_api_gauge = monitoring.BoolGauge('/tensorflow/api/keras',
#                                         'keras api usage', 'method')

class Model_custom(Model):
    def fit_generator_custom(self,
                              generator,
                              steps_per_epoch=None,
                              epochs=1,
                              verbose=1,
                              callbacks=None,
                              validation_data=None,
                              validation_steps=None,
                              validation_freq=1,
                              train_class_weight=None,
                              val_class_weight=None,
                              max_queue_size=10,
                              workers=1,
                              use_multiprocessing=False,
                              shuffle=True,
                              initial_epoch=0):
      # TODO: failed to pass some monitoring checks.
      # _keras_api_gauge.get_cell('fit_generator_custom').set(True)
      return training_generator.fit_generator(
                              self,
                              generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              verbose=verbose,
                              callbacks=callbacks,
                              validation_data=validation_data,
                              validation_steps=validation_steps,
                              validation_freq=validation_freq,
                              train_class_weight=train_class_weight,
                              val_class_weight=val_class_weight,
                              max_queue_size=max_queue_size,
                              workers=workers,
                              use_multiprocessing=use_multiprocessing,
                              shuffle=shuffle,
                              initial_epoch=initial_epoch,
                              steps_name='steps_per_epoch')

    def train_on_batch_custom(self, 
                              x, 
                              y=None,
                              sample_weight=None,
                              class_weight=None,
                              reset_metrics=True,
                              return_dict=False):
        
        self._assert_compile_was_called()
        self._check_call_args('train_on_batch')
        with self.distribute_strategy.scope(), \
         training_utils.RespectCompiledTrainableState(self):
         iterator = data_adapter.single_batch_iterator(
                            self.distribute_strategy, 
                            x,
                            y, 
                            sample_weight,
                            class_weight)
         train_function = self.make_train_function()
         logs = train_function(iterator)

        if reset_metrics:
          self.reset_metrics()
        logs = tf_utils.to_numpy_or_python_type(logs)
        if return_dict:
          return logs
        else:
          results = [logs.get(name, None) for name in self.metrics_names]
          if len(results) == 1:
            return results[0]
          return results

    def test_on_batch_custom(self, 
                             x, 
                             y=None, 
                             class_weight=None, 
                             sample_weight=None, 
                             reset_metrics=True,
                             return_dict=False):
        self._assert_compile_was_called()
        self._check_call_args('test_on_batch')
        with self.distribute_strategy.scope():
          iterator = data_adapter.single_batch_iterator(
                            self.distribute_strategy, 
                            x,
                            y, 
                            sample_weight,
                            class_weight)
          test_function = self.make_test_function()
          logs = test_function(iterator)
        
        if reset_metrics:
          self.reset_metrics()
        logs = tf_utils.to_numpy_or_python_type(logs)
        if return_dict:
            return logs
        else:
          results = [logs.get(name, None) for name in self.metrics_names]
        if len(results) == 1:
          return results[0]
        return results