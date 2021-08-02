# Imports
import numpy as np
import pandas as pd
import tensorflow as tf
import wave

# "From" imports
from keras import backend as keras_backend
from utils import NNET, api
from constants import constants
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

label_encoder = LabelEncoder()


def model_creation():
    # Ensure "channel last" data format on Keras
    keras_backend.set_image_data_format('channels_last')

    database = NNET.get_database(constants.csv_path)
    # database = pd.read_csv("4_Classes_Audio.csv")
    print("Database table:\n", database)
    print("\nTabela com número de incidencias por classe:\n", database['class_name'].value_counts())

    if not NNET.check_file_data():
        # Declare variables used in feature extraction
        audio_features = []
        audio_classes = []
        extracted_features_and_classes = []
        max_number_of_frames = 0

        for database_index, database_value in tqdm(database.iterrows()):
            audio_file_path = NNET.get_audio_path(database_value['filename'], database_value['fold'])
            class_name = database_value['class_name']

            # Extract Log-Mel Spectrograms without padding
            audio_feature = NNET.get_mel_spectrogram(
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

        padded_audio_features = NNET.add_padding(audio_features, max_number_of_frames)

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
        random_state=5
    )

    print(features_train.shape)
    print(features_test.shape)
    print(classes_train.shape)
    print(classes_test.shape)

    # Check distribuition of classes
    arr = np.array(classes_train)
    print(arr.size)
    print(f"Dog Bark array size: {(arr == 'dog_bark').sum()}")
    print(f"Gun Shot array size: {(arr == 'gun_shot').sum()}")
    print(f"Car Horn array size: {(arr == 'car_horn').sum()}")
    print(f"Siren array size: {(arr == 'siren').sum()}")

    # Transform classes data into binary array
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

    print(features_train.shape)
    print(features_test.shape)

    if not NNET.check_model_file():
        # Total number of labels to predict (equal to the network output nodes)
        num_labels = classes_train_encoded.shape[1]
        model = NNET.create_model(
            num_labels,
            constants.spatial_dropout_layer_1,
            constants.spatial_dropout_layer_2,
            constants.l2_rate
        )

        adam = tf.keras.optimizers.Adam(lr=0.0001)
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
        NNET.plot_train_history(history, x_ticks_vertical=True)

    # Prediction with model
    model = tf.keras.models.load_model(constants.model_filepath)
    NNET.model_evaluation_report(model, features_train, classes_train_encoded, features_test, classes_test_encoded)

    # Predict probabilities for test set
    classes_probs = model.predict(features_test, verbose=0)

    # Get predicted labels
    classes_labels_probs = np.argmax(classes_probs, axis=1)
    classes_trues = np.argmax(classes_test_encoded, axis=1)

    # Sets decimal precision (for printing output only)
    np.set_printoptions(precision=2)

    # Compute confusion matrix data
    cm = confusion_matrix(classes_trues, classes_labels_probs)

    NNET.plot_confusion_matrix(
        cm,
        constants.audio_class_names,
        normalized=False,
        title="Confusion Matrix",
        size=(8, 8)
    )

    true_pos = np.diag(cm)
    false_pos = np.sum(cm, axis=0) - true_pos
    false_neg = np.sum(cm, axis=1) - true_pos
    true_neg = np.sum(cm) - (true_pos + false_neg + false_pos)

    print("\nMetrics:")
    print(f"True Positives: {true_pos}")
    print(f"True Negatives: {true_neg}")
    print(f"False Positives: {false_pos}")
    print(f"False Negatives: {false_neg}")
    print("\n")

    # accuracies = NNET.acc_per_class(cm)
    accuracies = NNET.accuracy_per_class(true_pos, true_neg, false_pos, false_neg)
    precisions = NNET.precision_per_class(true_pos, false_pos)
    recalls = NNET.recall_per_class(true_pos, false_neg)

    print(pd.DataFrame({
        'CLASS': constants.audio_class_names,
        'ACCURACY': accuracies,
        'PRECISION': precisions,
        'RECALL': recalls,
    }).sort_values(by="ACCURACY", ascending=False))

    # Example of classification with created model
    # for test_audio in constants.test_audios_files:
    #     count = 0
    #     error_count = 0
    #     for test_audio in features_test:
    #         teste_audio_reshaped = test_audio.reshape(
    #             1,
    #             constants.num_rows,
    #             constants.num_columns,
    #             constants.num_channels
    #         )
    #
    #         audio_prediction = model.predict(teste_audio_reshaped)
    #         predict_max_value = np.max(audio_prediction)
    #         prediction_class = label_encoder.inverse_transform(np.argmax(audio_prediction, axis=-1))
    #
    #         if predict_max_value >= 0.8:
    #             if classes_test[count] != prediction_class[0]:
    #                 print('Audio played:', classes_test[count])
    #                 print('Predicted Label:', prediction_class[0])
    #                 print('Predicted Acc:', predict_max_value)
    #                 error_count += 1
    #         else:
    #             print('Audio played:', classes_test[count])
    #             print('Predicted Label:', prediction_class[0])
    #             print('Predicted Acc:', predict_max_value)
    #             error_count += 1
    #
    #         count += 1
    #     print(f"{error_count}/{count}")


def load_model():
    model = tf.keras.models.load_model(constants.model_filepath)
    return model


def classification(
        audio,
        audio_data,
        model,
        user_firebase_token,
        isCarHornEnable,
        isGunShotEnable,
        isDogBarkEnable,
        isSirenEnable
):
    waveFile = wave.open("teste.wav", 'wb')
    waveFile.setnchannels(constants.CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(constants.FORMAT))
    waveFile.setframerate(constants.RATE)
    waveFile.writeframes(b''.join(audio_data))
    waveFile.close()

    # Extract Log-Mel Spectrogram without padding
    audio_feature = NNET.get_mel_spectrogram(
        "teste.wav",
        constants.padding_mels_or_mfcc,
        constants.number_mels_or_mfcc
    )

    size = len(audio_feature[0])

    # Add padding if required
    if size < constants.num_columns:
        diff = constants.num_columns - size
        left = diff // 2
        right = diff - left
        audio_feature = np.pad(
            audio_feature,
            pad_width=((0, 0), (left, right)),
            mode='constant'
        )

    audio_feature_reshaped = audio_feature.reshape(
        1,
        constants.num_rows,
        constants.num_columns,
        constants.num_channels
    )

    audio_prediction = model.predict(audio_feature_reshaped)
    predict_max_value = np.max(audio_prediction)
    prediction_class = label_encoder.inverse_transform(np.argmax(audio_prediction, axis=-1))

    print('[PREDICTED LABEL]:', prediction_class[0])
    print('[PREDICTED ACC]:', predict_max_value)
    if predict_max_value >= 0.8:
        notification_text = ""
        notification_id = -1
        if prediction_class[0] == 'dog_bark' and isDogBarkEnable == "true":
            notification_text = "Foi detectado um latido perto de você"
            notification_id = constants.dog_bark
        elif prediction_class[0] == 'gun_shot' and isGunShotEnable == "true":
            notification_text = "Foi detectado um tiroteio perto de você"
            notification_id = constants.gun_shot
        elif prediction_class[0] == 'car_horn' and isCarHornEnable == "true":
            notification_text = "Foi detectado uma buzina perto de você"
            notification_id = constants.car_horn
        elif prediction_class[0] == 'siren' and isSirenEnable == "true":
            notification_text = "Foi detectado uma sirene perto de você"
            notification_id = constants.siren

        if notification_id != -1 and notification_text != 0:
            api.send_classification(user_firebase_token, notification_text, notification_id)
