'''
What is this for?

Both the training and ensemble stages need these functions.
So, initially, I duplicated them in both.

I've decided it would be cleaner to pull them out and refer to them with

from train_ensemble_utilities import *

However, the problem arises of getting data from the notebook environment into
this environment, so I've ended up duplicating a number of variables, which isn't
great, either
'''
import tensorflow as tf
from datetime import datetime

target_w                  = 320
target_h                  = 320
input_shape               = (320, 320, 3)
target_size               = (320, 320)
dataset_size              = 112120
local_image_directory     = 'nih_xrays_320'
num_labels                = 14
label_text                = ['Infiltration', 'Effusion', 'Atelectasis', 'Nodule', 'Mass', 'Pneumothorax', 'Consolidation', 'Pleural_Thickening', 'Cardiomegaly', 'Emphysema', 'Edema', 'Fibrosis', 'Pneumonia', 'Hernia']
y_pred_threshold_gt       = 0.5

# Constants
INPUT_SHAPE = (320, 320, 3)
IMG_SIZE = (320, 320)
BATCH_SIZE = 16
NUM_CLASSES = num_labels
EPOCHS = 250
IMG_DIR = 'nih_xrays_320'

TRAIN_SAMPLES = dataset_size


class DynamicWeightedBCE(tf.keras.losses.Loss):
    """
    Dynamic weighting based on batch statistics. Adapts weights during training.

    Example:
        # For dataset with varying class distributions
        loss_fn = DynamicWeightedBCE()
        model.compile(optimizer='adam', loss=loss_fn)
    """

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

        # Calculate class ratios per batch
        pos_ratio = tf.reduce_mean(y_true, axis=0)
        pos_weights = 1.0 / (pos_ratio + epsilon)
        neg_weights = 1.0 / (1.0 - pos_ratio + epsilon)

        return -tf.reduce_mean(
            y_true * pos_weights * tf.math.log(y_pred) +
            (1 - y_true) * neg_weights * tf.math.log1p(-y_pred)
        )


class DynamicWeightedBCE_wgt_smooth(tf.keras.losses.Loss):
    def __init__(self, smoothing_factor=0.1):
        super().__init__()
        self.smoothing_factor = smoothing_factor

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

        # Calculate class ratios with smoothing
        pos_ratio = tf.reduce_mean(y_true, axis=0)
        smoothed_ratio = pos_ratio * (1 - self.smoothing_factor) + 0.5 * self.smoothing_factor

        pos_weights = 1.0 / (smoothed_ratio + epsilon)
        neg_weights = 1.0 / (1.0 - smoothed_ratio + epsilon)

        return -tf.reduce_mean(
            y_true * pos_weights * tf.math.log(y_pred) +
            (1 - y_true) * neg_weights * tf.math.log1p(-y_pred)
        )


class DynamicWeightedBCE_combo(tf.keras.losses.Loss):
    """
    A custom TensorFlow loss function combining dynamic class weights with binary cross-entropy, temperature scaling, and label smoothing.

    This loss function adapts to class imbalance by maintaining moving averages of class ratios and computing dynamic weights.
    It includes temperature scaling to adjust prediction confidence and label smoothing to reduce overfitting.

    Args:
       temperature (float): Controls prediction sharpness. Higher values produce softer predictions. Default: 1.0
            must be positive (> 0), with:
                Values < 1: Sharper predictions
                Values > 1: Softer predictions
                1.0: No scaling effect

       smoothing_factor (float): Label smoothing strength. Default: 0.1
           must be in the range [0,1]
               0: No smoothing (original labels)
               1: Full smoothing (all labels become 0.5)
               Values between 0-1: Linear interpolation between original labels and 0.5

       momentum (float): Moving average momentum for class ratio tracking. Default: 0.0
           must be in range [0, 1], with:
                0: No moving average (only uses current batch)
                1: Keeps initial ratio forever
                Values between 0-1: Exponential moving average with higher values giving more weight to history

    Attributes:
       moving_pos_ratio: Moving average of positive class ratios
       initialized: Tracks if moving averages are initialized

    Example:
       loss_fn = DynamicWeightedBCE_combo(temperature=1.0, smoothing_factor=0.094, momentum=0.2)
       model.compile(loss=loss_fn)
    """

    def __init__(self, temperature=1.0, smoothing_factor=0.1, momentum=0.0):
        super().__init__()
        self.temperature = temperature
        self.smoothing_factor = smoothing_factor
        self.momentum = momentum
        # Initialize variables
        self.moving_pos_ratio = None
        self.initialized = tf.Variable(False)

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

        # Temperature scaling
        y_pred = y_pred ** (1 / self.temperature)

        # Calculate current batch ratios
        current_pos_ratio = tf.reduce_mean(y_true, axis=0)

        # Initialize moving average if needed
        if self.moving_pos_ratio is None:
            self.moving_pos_ratio = tf.Variable(
                tf.zeros_like(current_pos_ratio),
                trainable=False
            )

        # Update moving average using tf.assign
        def update_moving_average():
            return self.moving_pos_ratio.assign(
                self.momentum * self.moving_pos_ratio +
                (1 - self.momentum) * current_pos_ratio
            )

        # Initialize or update moving average
        self.moving_pos_ratio.assign(
            tf.cond(
                self.initialized,
                update_moving_average,
                lambda: self.moving_pos_ratio.assign(current_pos_ratio)
            )
        )

        # Mark as initialized
        self.initialized.assign(True)

        # Apply label smoothing
        smoothed_ratio = self.moving_pos_ratio * (1 - self.smoothing_factor) + 0.5 * self.smoothing_factor

        pos_weights = 1.0 / (smoothed_ratio + epsilon)
        neg_weights = 1.0 / (1.0 - smoothed_ratio + epsilon)

        return -tf.reduce_mean(
            y_true * pos_weights * tf.math.log(y_pred) +
            (1 - y_true) * neg_weights * tf.math.log1p(-y_pred)
        )


import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os
from tqdm import tqdm
from pprint import pprint
import sklearn.metrics




# 1. Create smaller datasets
def create_subset_data(df, n_samples, random_state=42):
    """Create a smaller subset of the data"""
    # return df.sample(n=n_samples, random_state=random_state)
    return df.head(n_samples)


# 2. Data loading and preprocessing
def load_and_preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    try:
        img = tf.image.decode_png(img, channels=3)
    except:
        raise

    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, [320, 320])
    img = tf.keras.applications.densenet.preprocess_input(img)
    return img


def create_dataset(df, img_dir, batch_size=BATCH_SIZE, shuffle_buffer=1000, cache=False):
    """Create a TensorFlow dataset from DataFrame"""
    # Create full image paths
    image_paths = [os.path.join(img_dir, img_name) for img_name in df['Image']]

    # Create labels array with explicit type
    label_columns = label_text
    labels = df[label_columns].values.astype(np.float32)

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    # Map loading and preprocessing function
    dataset = dataset.map(
        lambda x, y: (load_and_preprocess_image(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Shuffle and batch
    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)

    # add caching
    if cache:
        dataset = dataset.batch(batch_size).cache().prefetch(
            tf.data.AUTOTUNE)  # in-memory cache blows up the g5.xlarge kernel
    else:
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)  # this one worked

    return dataset


# 3. Create model
def create_model(num_labels=NUM_CLASSES, input_shape=INPUT_SHAPE, dropout_rate=0.3):
    # def create_model_with_attention(num_labels=NUM_CLASSES, input_shape=INPUT_SHAPE, dropout_rate=0.3):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Base DenseNet121
    base_model = tf.keras.applications.DenseNet121(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )

    # All layers are trainable by default
    x = base_model.output

    # Add Convolutional Block Attention Module (CBAM)

    # Channel Attention
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling2D()(x)

    avg_pool = tf.keras.layers.Reshape((1, 1, 1024))(avg_pool)
    max_pool = tf.keras.layers.Reshape((1, 1, 1024))(max_pool)

    shared_dense_1 = tf.keras.layers.Dense(512, activation='relu')
    shared_dense_2 = tf.keras.layers.Dense(1024)

    avg_pool = shared_dense_1(avg_pool)
    max_pool = shared_dense_1(max_pool)
    avg_pool = shared_dense_2(avg_pool)
    max_pool = shared_dense_2(max_pool)

    channel_attention = tf.keras.layers.Add()([avg_pool, max_pool])
    channel_attention = tf.keras.layers.Activation('sigmoid')(channel_attention)

    # Apply channel attention
    x = tf.keras.layers.Multiply()([x, channel_attention])

    # Spatial Attention
    avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=-1, keepdims=True))(x)
    max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=-1, keepdims=True))(x)
    spatial_attention = tf.keras.layers.Concatenate()([avg_pool, max_pool])

    spatial_attention = tf.keras.layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(
        spatial_attention)

    # Apply spatial attention
    x = tf.keras.layers.Multiply()([x, spatial_attention])

    # Global pooling and classification layers
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(num_labels, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


# 4. Create callbacks
def create_callbacks():
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        min_delta=0.00001,
        verbose=2,
        restore_best_weights=True
    )

    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        factor=0.5,
        patience=2,
        min_delta=0.0001,
        verbose=2,
        min_lr=1e-6
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_checkpoints/weights.{epoch:04d}-{val_loss:.5f}.keras",
        save_weights_only=False,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1)

    class CSVTrainingLogger(tf.keras.callbacks.Callback):
        def __init__(self, filename='model_statistics/training_history.csv'):
            super().__init__()
            self.filename = filename

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # Define headers
            self.headers = ['epoch', 'timestamp', 'loss', 'val_loss', 'auc', 'val_auc',
                            'recall', 'val_recall', 'specificity', 'val_specificity',
                            'f1', 'val_f1', 'hamming_loss', 'val_hamming_loss', 'ppv', 'val_ppv',
                            'npv', 'val_npv']

            # Check if file exists and is empty
            file_exists = os.path.exists(filename)
            file_empty = False
            if file_exists:
                file_empty = os.path.getsize(filename) == 0

            # Write headers if file doesn't exist or is empty
            if not file_exists or file_empty:
                with open(filename, 'w', newline='') as f:
                    f.write(','.join(self.headers) + '\n')

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Create list of values in same order as headers
            values = [
                str(epoch),
                timestamp,
                str(logs.get('loss', '')),
                str(logs.get('val_loss', '')),
                str(logs.get('auc', '')),
                str(logs.get('val_auc', '')),
                str(logs.get('recall', '')),
                str(logs.get('val_recall', '')),
                str(logs.get('specificity', '')),
                str(logs.get('val_specificity', '')),
                str(logs.get('f1', '')),
                str(logs.get('val_f1', '')),
                str(logs.get('hamming_loss', '')),
                str(logs.get('val_hamming_loss', '')),
                str(logs.get('ppv', '')),
                str(logs.get('val_ppv', '')),
                str(logs.get('npv', '')),
                str(logs.get('val_npv', ''))
            ]

            try:
                # Append metrics to CSV using 'a' mode
                with open(self.filename, 'a', newline='') as f:
                    f.write(','.join(values) + '\n')
            except Exception as e:
                print(f"Error writing to CSV file: {e}")
                raise

    # Create logger instance
    csv_logger = CSVTrainingLogger()

    callback_list = [
        early_stopping,
        checkpoint_callback,
        lr_schedule,
        csv_logger]

    return callback_list, early_stopping


# 4. Create metrics
def setup_strategy():
    """Set up distributed training strategy based on available GPUs"""
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()
    return strategy



class NegativePredictiveValue(tf.keras.metrics.Metric):
    def __init__(self, name='npv', threshold=0.5, **kwargs):
        super(NegativePredictiveValue, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        # Initialize accumulator variables
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to binary based on threshold
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        # Calculate true negatives and false negatives
        neg_pred = tf.cast(y_pred < self.threshold, tf.float32)
        neg_true = tf.cast(y_true < self.threshold, tf.float32)

        true_negatives = tf.reduce_sum(neg_pred * neg_true)
        false_negatives = tf.reduce_sum((1 - neg_pred) * neg_true)

        # Update accumulator variables
        self.true_negatives.assign_add(true_negatives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        # Calculate NPV: TN / (TN + FN)
        denominator = self.true_negatives + self.false_negatives
        return tf.math.divide_no_nan(self.true_negatives, denominator)

    def reset_state(self):
        # Reset accumulator variables
        self.true_negatives.assign(0.0)
        self.false_negatives.assign(0.0)


npv = NegativePredictiveValue()


class Specificity(tf.keras.metrics.Metric):
    def __init__(self, name="specificity", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tn = self.add_weight(name="tn", initializer="zeros")
        self.pn = self.add_weight(name="pn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.less(y_pred, y_pred_threshold_gt), tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        neg_pred = tf.equal(y_pred, 0)
        neg_true = tf.equal(y_true, 0)

        tn = tf.reduce_sum(tf.cast(tf.logical_and(neg_pred, neg_true), tf.float32))
        pn = tf.reduce_sum(tf.cast(neg_true, tf.float32))

        self.tn.assign_add(tn)
        self.pn.assign_add(pn)

    def result(self):
        return self.tn / (self.pn + tf.keras.backend.epsilon())

    def reset_state(self):
        self.tn.assign(0.)
        self.pn.assign(0.)


class HammingLoss(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name="hamming_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.greater(y_pred, self.threshold), tf.float32)

        loss = tf.reduce_mean(tf.abs(y_true - y_pred), axis=1)
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)

        self.total.assign_add(tf.reduce_sum(loss))
        self.count.assign_add(batch_size)

    def result(self):
        return self.total / (self.count + tf.keras.backend.epsilon())

    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)


def create_metrics():
    return [
        tf.keras.metrics.Recall(name='recall'),
        Specificity(name="specificity"),
        tf.keras.metrics.AUC(multi_label=True, num_labels=num_labels, name='auc'),
        tf.keras.metrics.Precision(name='ppv'),
        npv,
        tf.keras.metrics.F1Score(threshold=y_pred_threshold_gt, average='micro', name='f1'),
        HammingLoss(threshold=y_pred_threshold_gt, name="hamming_loss")
    ]







