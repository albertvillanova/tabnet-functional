# Source code:
# https://github.com/google-research/google-research/blob/master/tabnet/tabnet_model.py
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .layers import GLU

# TODO: refactorize as config
BATCH_MOMENTUM = 0.7
BATCH_SIZE = 32
FEATURE_DIM = 4
NUM_DECISION_STEPS = 6
NUM_FEATURES = 54
OUTPUT_DIM = 2
VIRTUAL_BATCH_SIZE = 4


# Input features
# TODO: fix shape with embeddings
inputs = keras.Input(shape=(54,))
# Normalize
features = inputs
features = keras.layers.BatchNormalization(momentum=BATCH_MOMENTUM)(features)

# Initialize decision-step dependent variables
output_aggregated = tf.zeros([BATCH_SIZE, OUTPUT_DIM])
masked_features = features
mask_values = tf.zeros([BATCH_SIZE, NUM_FEATURES])
aggregated_mask_values = tf.zeros([BATCH_SIZE, NUM_FEATURES])
complementary_aggregated_mask_values = tf.ones(
    [BATCH_SIZE, NUM_FEATURES])
total_entropy = 0
# TODO
# if is_training:
#     v_b = self.virtual_batch_size
# else:
#     v_b = 1

# Shared layers
transform_f1 = keras.layers.Dense(FEATURE_DIM * 2, use_bias=False,
                                  name='transform_f1')  # TODO: remove name
transform_f2 = keras.layers.Dense(FEATURE_DIM * 2, use_bias=False,
                                  name='transform_f2')  # TODO: remove name

for ni in range(NUM_DECISION_STEPS):
    # Feature transformer with two shared and two decision step dependent
    # blocks
    # TODO: remove
    # reuse_flag = (ni > 0)
    # transform_1
    x = transform_f1(masked_features)
    x = keras.layers.BatchNormalization(
        momentum=BATCH_MOMENTUM, virtual_batch_size=VIRTUAL_BATCH_SIZE)(x)
    x = GLU()(x)
    # transform_2
    x1 = x
    x = transform_f2(x)
    x = keras.layers.BatchNormalization(
        momentum=BATCH_MOMENTUM, virtual_batch_size=VIRTUAL_BATCH_SIZE)(x)
    x = GLU()(x)
    x = (x + x1) * np.sqrt(0.5)
    # transform_3
    x2 = x
    x = keras.layers.Dense(FEATURE_DIM * 2, use_bias=False,
                           name=f'transform_f3_{ni}')(x)
    x = keras.layers.BatchNormalization(
        momentum=BATCH_MOMENTUM, virtual_batch_size=VIRTUAL_BATCH_SIZE)(x)
    x = GLU()(x)
    x = (x + x2) * np.sqrt(0.5)
    # transform_4
    x3 = x
    x = keras.layers.Dense(FEATURE_DIM * 2, use_bias=False,
                           name=f'transform_f4_{ni}')(x)
    x = keras.layers.BatchNormalization(
        momentum=BATCH_MOMENTUM, virtual_batch_size=VIRTUAL_BATCH_SIZE)(x)
    x = GLU()(x)
    x = (x + x3) * np.sqrt(0.5)

    #
    if ni > 0:
        pass
