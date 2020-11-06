# Source code:
# https://github.com/google-research/google-research/blob/master/tabnet/tabnet_model.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

from .layers import GLU

# TODO: refactorize as config
BATCH_MOMENTUM = 0.7
BATCH_SIZE = 32
EPSILON = 0.00001
FEATURE_DIM = 4
NUM_CLASSES = 7
NUM_DECISION_STEPS = 6
NUM_FEATURES = 54
OUTPUT_DIM = 2
RELAXATION_FACTOR = 1.5
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

# Input: masked_features
# Output: output_aggregated, total_entropy
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
        decision_out = tf.nn.relu(x[:, :OUTPUT_DIM])
        # Decision aggregation
        output_aggregated += decision_out
        # Aggregated masks are used for visualization of the
        # feature importance attributes
        scale_agg = tf.math.reduce_sum(decision_out, axis=1, keepdims=True) / (
                    NUM_DECISION_STEPS - 1)
        aggregated_mask_values += mask_values * scale_agg

    #
    features_for_coef = (x[:, OUTPUT_DIM:])

    #
    if ni < NUM_DECISION_STEPS - 1:
        # Determine the feature masks via linear and nonlinear
        # transformations, taking into account of aggregated feature use
        mask_values = keras.layers.Dense(
            NUM_FEATURES, use_bias=False,
            name=f'transform_coef_{ni}')(features_for_coef)
        mask_values = keras.layers.BatchNormalization(
            momentum=BATCH_MOMENTUM,
            virtual_batch_size=VIRTUAL_BATCH_SIZE)(mask_values)
        mask_values *= complementary_aggregated_mask_values
        mask_values = tfa.layers.Sparsemax()(mask_values)

        # Relaxation factor controls the amount of reuse of features between
        # different decision blocks and updated with the values of
        # coefficients.
        complementary_aggregated_mask_values *= (
                RELAXATION_FACTOR - mask_values)

        # Entropy is used to penalize the amount of sparsity in feature
        # selection.
        total_entropy += tf.math.reduce_mean(tf.math.reduce_sum(
            -mask_values * tf.math.log(mask_values + EPSILON), axis=1)) / (
                                     NUM_DECISION_STEPS - 1)

        # Feature selection
        masked_features = tf.math.multiply(mask_values, features)

        # Visualization of the feature selection mask at decision step ni
        tf.summary.image(
            f"Mask for step {ni}",
            tf.expand_dims(tf.expand_dims(mask_values, 0), 3),
            max_outputs=1)

# Visualization of the aggregated feature importances
tf.summary.image(
    "Aggregated mask",
    tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3),
    max_outputs=1)

outputs = (output_aggregated, total_entropy)

encoder = keras.Model(inputs=inputs, outputs=outputs, name='encoder')

# Classifier
logits = keras.layers.Dense(NUM_CLASSES, use_bias=False)(output_aggregated)
# predictions = tf.nn.softmax(logits)
# return logits, predictions
classifier_outputs = (logits, total_entropy)
classifier = keras.Model(inputs=inputs, outputs=classifier_outputs,
                         name='classifier')

# Regressor
predictions = keras.layers.Dense(1)(output_aggregated)
regressor_outputs = (predictions, total_entropy)
# return predictions
regressor = keras.Model(inputs=inputs, outputs=regressor_outputs,
                        name='regressor')
