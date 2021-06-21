# Imports
import keras.applications.resnet
import numpy as np
import pandas as pd
import tensorflow as tf

# "From" imports
from keras import backend as keras_backend
from utils import utils
from constants import constants
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Ensure "channel last" data format on Keras
keras_backend.set_image_data_format('channels_last')

# database = utils.get_database(constants.csv_path)
database = pd.read_csv("4_Classes_Audio.csv")
print("Database table:\n", database)
print("\nTabela com número de incidencias por classe:\n", database['class_name'].value_counts())

# max_classes_number = database['class_name'].value_counts().min()
#
# count1 = 0
# count2 = 0
# count3 = 0
# count4 = 0
# new_database = []
#
# for database_index, database_value in database.iterrows():
#     if database_value['class_name'] == 'dog_bark' and count1 < max_classes_number:
#         new_database.append([database_value['filename'], database_value['class_name'], database_value['fold']])
#         count1 += 1
#     if database_value['class_name'] == 'car_horn' and count2 < max_classes_number:
#         new_database.append([database_value['filename'], database_value['class_name'], database_value['fold']])
#         count2 += 1
#     if database_value['class_name'] == 'siren' and count3 < max_classes_number:
#         new_database.append([database_value['filename'], database_value['class_name'], database_value['fold']])
#         count3 += 1
#     if database_value['class_name'] == 'gun_shot' and count4 < max_classes_number:
#         new_database.append([database_value['filename'], database_value['class_name'], database_value['fold']])
#         count4 += 1
#
# print(new_database)
# a = np.asarray(new_database)
# pd.DataFrame(a).to_csv("4_Classes_Audio.csv")

if not utils.check_file_data():
    # Declare variables used in feature extraction
    audio_features = []
    audio_classes = []
    extracted_features_and_classes = []
    max_number_of_frames = 0

    for database_index, database_value in database.iterrows():
        audio_file_path = utils.get_audio_path(database_value['filename'], database_value['fold'])
        class_name = database_value['class_name']

        # Get MFCCs without padding
        # audio_feature = utils.extract_mfcc_features(
        #     audio_file_path,
        #     constants.padding_mels_or_mfcc,
        #     constants.number_mels_or_mfcc
        # )

        # Extract Log-Mel Spectrograms without padding
        audio_feature = utils.get_mel_spectrogram(
            audio_file_path,
            constants.padding_mels_or_mfcc,
            constants.number_mels_or_mfcc
        )

        # Save MFCC frame count
        number_of_frames = audio_feature.shape[1]

        # Add values into features and classes list
        audio_features.append(audio_feature)
        audio_classes.append(class_name)

        # Update maximum number of frames
        if number_of_frames > max_number_of_frames:
            max_number_of_frames = number_of_frames

    padded_audio_features = utils.add_padding(audio_features, max_number_of_frames)

    print("Raw features length: {}".format(len(audio_features)))
    print("Padded features length: {}".format(len(padded_audio_features)))
    print("Feature labels length: {}".format(len(audio_classes)))

    features = np.array(padded_audio_features)
    classes = np.array(audio_classes)

    np.save(constants.features_filepath, features)
    np.save(constants.classes_filepath, classes)

# Load extraction file data
features = np.load(constants.features_filepath)
classes = np.load(constants.classes_filepath)
print("\nSuccess on load data")

features_train, features_test, classes_train, classes_test = train_test_split(
    features,
    classes,
    test_size=0.2,
    random_state=0
)

# Transform classes data into binary array
label_encoder = LabelEncoder()
classes_train_encoded = tf.keras.utils.to_categorical(label_encoder.fit_transform(classes_train))
classes_test_encoded = tf.keras.utils.to_categorical(label_encoder.fit_transform(classes_test))

# Reshape to fit the network input (channel last)
features_train = features_train.reshape(
    features_train.shape[0],
    constants.num_rows,
    constants.num_columns,
    constants.num_channels
)
features_test = features_test.reshape(
    features_test.shape[0],
    constants.num_rows,
    constants.num_columns,
    constants.num_channels
)

if not utils.check_model_file():
    # Total number of labels to predict (equal to the network output nodes)
    num_labels = classes_train_encoded.shape[1]
    model = utils.create_model(
        num_labels,
        constants.spatial_dropout_layer_1,
        constants.spatial_dropout_layer_2,
        constants.l2_rate
    )

    adam = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.99, beta_2=0.999)
    # adam = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=adam
    )

    # Display model architecture summary
    model.summary()

    # Save checkpoints
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=constants.model_filepath,
        verbose=1,
        save_best_only=True
    )

    start_datetime = datetime.now()
    history = model.fit(
        features_train,
        classes_train_encoded,
        batch_size=constants.batch_size,
        epochs=constants.epochs,
        validation_split=1/5.,
        # validation_data=(features_test, classes_test_encoded),
        callbacks=[checkpointer],
        verbose=1
    )

    duration = datetime.now() - start_datetime
    print("Training completed in time: ", duration)
    utils.plot_train_history(history, x_ticks_vertical=True)

# Prediction with model
model = tf.keras.models.load_model(constants.model_filepath)
utils.model_evaluation_report(model, features_train, classes_train_encoded, features_test, classes_test_encoded)

# Predict probabilities for test set
classes_probs = model.predict(features_test, verbose=0)

# Get predicted labels
classes_labels_probs = np.argmax(classes_probs, axis=1)
classes_trues = np.argmax(classes_test_encoded, axis=1)

# Sets decimal precision (for printing output only)
np.set_printoptions(precision=2)

# Compute confusion matrix data
cm = confusion_matrix(classes_trues, classes_labels_probs)

utils.plot_confusion_matrix(
    cm,
    constants.audio_class_names,
    normalized=False,
    title="Confusion Matrix",
    size=(8, 8)
)

accuracies = utils.acc_per_class(cm)

print(pd.DataFrame({
    'CLASS': constants.audio_class_names,
    'ACCURACY': accuracies
}).sort_values(by="ACCURACY", ascending=False))

# for test_audio in constants.test_audios_files:
count = 0
error_count = 0
for test_audio in features_test:
    # teste_audio = utils.get_mel_spectrogram(
    #     f"C:/Users/giova/Downloads/{test_audio}",
    #     constants.padding_mels_or_mfcc,
    #     constants.number_mels_or_mfcc
    # )
    #
    # if teste_audio[1].size < features.shape[2]:
    #     diff = features.shape[2] - teste_audio[1].size
    #     left = diff // 2
    #     right = diff - left
    #     teste_audio = np.pad(
    #         teste_audio,
    #         pad_width=((0, 0), (left, right)),
    #         mode='constant'
    #     )

    teste_audio_reshaped = test_audio.reshape(
        1,
        constants.num_rows,
        constants.num_columns,
        constants.num_channels
    )

    predicted_label = model.predict(teste_audio_reshaped)
    print('\n')
    print(predicted_label)
    # if np.argwhere(predicted_label >= 0.7):
    prediction_class = label_encoder.inverse_transform(np.argmax(predicted_label, axis=-1))
    print('Audio played:', classes_test[count])
    print('Predicted Label:', prediction_class[0])
    if classes_test[count] != prediction_class[0]:
        error_count += 1
    count += 1
print(f"{error_count}/{count}")
