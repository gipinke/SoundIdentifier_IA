# Third party libs
import pandas as pd
import librosa
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# File imports
from pathlib import Path
from constants import constants
from sklearn import metrics


def get_database(filepath):
    database_pd = pd.read_csv(filepath)
    database_df = pd.DataFrame(database_pd, columns=["slice_file_name", "fold", "class"])

    # Comment for 10 Classes database
    database_df = database_df.loc[database_df['class'].isin(['car_horn', 'dog_bark', 'gun_shot', 'siren'])]

    database_df = database_df.rename(columns={"slice_file_name": "filename", "class": "class_name"})

    return database_df


def get_audio_path(filename, fold):
    audio_path = f"{Path.home()}/Desktop/UrbanSound8K/audio/fold{fold}/{filename}"
    return audio_path


def check_file_data():
    if (os.path.exists(constants.features_filepath) is False or
            os.path.exists(constants.classes_filepath) is False):
        return False

    if (os.path.getsize(constants.features_filepath) == 0 and
            os.path.getsize(constants.classes_filepath) == 0):
        return False

    return True


def check_model_file():
    if os.path.exists(constants.model_filepath) is False:
        return False

    if os.path.getsize(constants.model_filepath) == 0:
        return False

    return True


def get_mel_spectrogram(filepath, mfcc_padding, number_mels):
    try:
        # Load audio file
        loaded_audio, audio_sampling_rate = librosa.load(filepath)

        # Normalize audio data between -1 and 1
        normalized_audio = librosa.util.normalize(loaded_audio)

        # Generate mel scaled filterbanks
        mel = librosa.feature.melspectrogram(
            normalized_audio,
            sr=audio_sampling_rate,
            n_mels=number_mels
        )

        # Convert sound intensity to log amplitude:
        mel_db = librosa.amplitude_to_db(abs(mel))

        # Normalize between -1 and 1
        normalized_mel = librosa.util.normalize(mel_db)

        # Should we require padding
        shape = normalized_mel.shape[1]
        if mfcc_padding > 0 & shape < mfcc_padding:
            diff = mfcc_padding - shape
            left = diff // 2
            right = diff - left
            normalized_mel = np.pad(
                normalized_mel,
                pad_width=((0, 0), (left, right)),
                mode='constant'
            )

    except Exception as e:
        print(f"Error in extract mel features: {e}")
        return None

    return normalized_mel


def extract_mfcc_features(filepath, mfcc_padding, number_mfcc):
    try:
        # Get audio and sampling rate
        loaded_audio, audio_sampling_rate = librosa.load(filepath)

        # Normalize audio data
        normalized_audio = librosa.util.normalize(loaded_audio)

        # Get MFCC sequence and normalize it
        mfcc_sequence = librosa.feature.mfcc(
            y=normalized_audio,
            sr=audio_sampling_rate,
            n_mfcc=number_mfcc
        )
        normalized_mfcc_sequence = librosa.util.normalize(mfcc_sequence)

        # Check if has padding
        shape = normalized_mfcc_sequence[1]
        if mfcc_padding > 0 and shape < mfcc_padding:
            diff = mfcc_padding - shape
            left = diff // 2
            right = diff - left
            normalized_mfcc_sequence = np.pad(
                normalized_mfcc_sequence,
                pad_width=((0, 0), (left, right)),
                mode='constant'
            )

    except Exception as e:
        print(f"Error in extract mfcc features: {e}")
        return None

    return normalized_mfcc_sequence


def add_padding(audio_features, max_number_of_frames):
    audio_features_padded = []
    for index in range(len(audio_features)):
        audio_feature = audio_features[index]
        size = len(audio_feature[0])

        # Add padding if required
        if size < max_number_of_frames:
            diff = max_number_of_frames - size
            left = diff // 2
            right = diff - left
            audio_feature_np = np.pad(
                audio_feature,
                pad_width=((0, 0), (left, right)),
                mode='constant'
            )

        audio_features_padded.append(audio_feature_np)

    return audio_features_padded


# def create_model(num_labels, spatial_dropout_layer_1=0, spatial_dropout_layer_2=0, l2_rate=0):
#     model = tf.keras.Sequential()
#
#     # Add first convolutional layer (input layer) with 32 filters
#     model.add(tf.keras.layers.Conv2D(
#         filters=32,
#         kernel_size=constants.kernel_size,
#         kernel_regularizer=tf.keras.regularizers.l2(l2_rate),
#         input_shape=(
#             constants.num_rows,
#             constants.num_columns,
#             constants.num_channels
#         )
#     ))
#     model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
#     model.add(tf.keras.layers.BatchNormalization())
#     model.add(tf.keras.layers.SpatialDropout2D(spatial_dropout_layer_1))
#
#     # Add second convolutional layer with 32 filters
#     model.add(tf.keras.layers.Conv2D(
#         filters=32,
#         kernel_size=constants.kernel_size,
#         kernel_regularizer=tf.keras.regularizers.l2(l2_rate)
#     ))
#     model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
#     model.add(tf.keras.layers.BatchNormalization())
#     model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#     model.add(tf.keras.layers.SpatialDropout2D(spatial_dropout_layer_1))
#
#     # Add third convolutional layer with 64 filters
#     model.add(tf.keras.layers.Conv2D(
#         filters=64,
#         kernel_size=constants.kernel_size,
#         kernel_regularizer=tf.keras.regularizers.l2(l2_rate)
#     ))
#     model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
#     model.add(tf.keras.layers.BatchNormalization())
#     model.add(tf.keras.layers.SpatialDropout2D(spatial_dropout_layer_2))
#
#     # Add fourth convolutional layer with 64 filters
#     model.add(tf.keras.layers.Conv2D(
#         filters=64,
#         kernel_size=constants.kernel_size,
#         kernel_regularizer=tf.keras.regularizers.l2(l2_rate)
#     ))
#     model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
#     model.add(tf.keras.layers.BatchNormalization())
#
#     # Reduces each h×w feature map to a single number by taking the average of all h,w values.
#     model.add(tf.keras.layers.GlobalAveragePooling2D())
#
#     # Softmax output
#     model.add(tf.keras.layers.Dense(num_labels, activation='softmax'))
#     return model


# def create_model(num_classes, spatial_dropout_layer_1=0, spatial_dropout_layer_2=0, l2_rate=0):
#     base_model = tf.keras.applications.resnet.ResNet50(
#         weights=None,
#         include_top=None,
#         pooling=(2, 2),
#         input_shape=(
#             constants.num_rows,
#             constants.num_columns,
#             constants.num_channels
#         )
#     )
#
#     x = base_model.output
#     x = tf.keras.layers.GlobalAveragePooling2D()(x)
#     x = tf.keras.layers.Dropout(0.5)(x)
#     predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
#     return tf.keras.Model(inputs=base_model.input, outputs=predictions)


def create_model(num_classes, spatial_dropout_layer_1=0, spatial_dropout_layer_2=0, l2_rate=0):
    base_model = tf.keras.applications.EfficientNetB0(
        weights=None,
        include_top=None,
        input_shape=(
            constants.num_rows,
            constants.num_columns,
            constants.num_channels
        )
    )

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=base_model.input, outputs=predictions)


def evaluate_model(model, x_train, y_train, x_test, y_test):
    train_score = model.evaluate(x_train, y_train, verbose=0)
    test_score = model.evaluate(x_test, y_test, verbose=0)
    return train_score, test_score


def model_evaluation_report(model, x_train, y_train, x_test, y_test, calc_normal=True):
    dash = '-' * 38

    # Compute scores
    train_score, test_score = evaluate_model(model, x_train, y_train, x_test, y_test)

    # Pint Train vs Test report
    print('{:<10s}{:>14s}{:>14s}'.format("", "LOSS", "ACCURACY"))
    print(dash)
    print('{:<10s}{:>14.4f}{:>14.4f}'.format("Training:", train_score[0], 100 * train_score[1]))
    print('{:<10s}{:>14.4f}{:>14.4f}'.format("Test:", test_score[0], 100 * test_score[1]))

    # Calculate and report normalized error difference?
    if calc_normal:
        max_err = max(train_score[0], test_score[0])
        error_diff = max_err - min(train_score[0], test_score[0])
        normal_diff = error_diff * 100 / max_err
        print('{:<10s}{:>13.2f}{:>1s}'.format("Normal diff ", normal_diff, ""))


def plot_train_history(history, x_ticks_vertical=False):
    history = history.history

    # min loss / max accs
    min_loss = min(history['loss'])
    min_val_loss = min(history['val_loss'])
    max_accuracy = max(history['accuracy'])
    max_val_accuracy = max(history['val_accuracy'])

    # x pos for loss / acc min/max
    min_loss_x = history['loss'].index(min_loss)
    min_val_loss_x = history['val_loss'].index(min_val_loss)
    max_accuracy_x = history['accuracy'].index(max_accuracy)
    max_val_accuracy_x = history['val_accuracy'].index(max_val_accuracy)

    # summarize history for loss, display min
    plt.figure(figsize=(16, 8))
    plt.plot(history['loss'], color="#1f77b4", alpha=0.7)
    plt.plot(history['val_loss'], color="#ff7f0e", linestyle="--")
    plt.plot(min_loss_x, min_loss, marker='o', markersize=3, color="#1f77b4", alpha=0.7, label='Inline label')
    plt.plot(min_val_loss_x, min_val_loss, marker='o', markersize=3, color="#ff7f0e", alpha=0.7, label='Inline label')
    plt.title('Model loss', fontsize=20)
    plt.ylabel('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.legend(['Train',
                'Test',
                ('%.3f' % min_loss),
                ('%.3f' % min_val_loss)],
               loc='upper right',
               fancybox=True,
               framealpha=0.9,
               shadow=True,
               borderpad=1)

    if x_ticks_vertical:
        plt.xticks(np.arange(0, len(history['loss']), 5.0), rotation='vertical')
    else:
        plt.xticks(np.arange(0, len(history['loss']), 5.0))

    plt.show()

    # summarize history for accuracy, display max
    plt.figure(figsize=(16, 6))
    plt.plot(history['accuracy'], alpha=0.7)
    plt.plot(history['val_accuracy'], linestyle="--")
    plt.plot(max_accuracy_x, max_accuracy, marker='o', markersize=3, color="#1f77b4", alpha=0.7)
    plt.plot(max_val_accuracy_x, max_val_accuracy, marker='o', markersize=3, color="orange", alpha=0.7)
    plt.title('Model accuracy', fontsize=20)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.legend(['Train',
                'Test',
                ('%.2f' % max_accuracy),
                ('%.2f' % max_val_accuracy)],
               loc='upper left',
               fancybox=True,
               framealpha=0.9,
               shadow=True,
               borderpad=1)
    plt.figure(num=1, figsize=(10, 6))

    if x_ticks_vertical:
        plt.xticks(np.arange(0, len(history['accuracy']), 5.0), rotation='vertical')
    else:
        plt.xticks(np.arange(0, len(history['accuracy']), 5.0))

    plt.show()


def compute_confusion_matrix(y_true, y_pred, normalize=False):
    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm


# Plots a confussion matrix
def plot_confusion_matrix(cm, classes, normalized=False, title=None, cmap=plt.cm.Blues, size=(10, 10)):
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes, yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label'
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalized else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.show()


# Expects a NumPy array with probabilities and a confusion matrix data, retuns accuracy per class
def acc_per_class(np_probs_array):
    accs = []
    for idx in range(0, np_probs_array.shape[0]):
        correct = np_probs_array[idx][idx].astype(int)
        total = np_probs_array[idx].sum().astype(int)
        acc = (correct / total) * 100
        accs.append(acc)
    return accs
