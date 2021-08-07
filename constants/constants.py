import pyaudio

# Server specs
port = 5001

# Audio specs
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

# Firebase Message
serverToken = 'AAAAkn3SXl0:APA91bHnUPAT7Kgwcc4XOWIyfk92wkiu1G4cfWRpI6K-2Efpx8rescBdbXGnA8mNbXrZQ1kc5OZMlS9KsxRmMT-W6ffT1b6M-pT9HK_EiTiPbydeUP_I1_9i7PjS9OMcNEzgCSG789U1'
serverUrl = "https://fcm.googleapis.com/fcm/send"

# Define paths used
csv_path = "metadata/4_Classes_Audio.csv"
features_filepath = "data/features_mel_spec.npy"
classes_filepath = "data/classes_mel_spec.npy"
model_filepath = "model/simple-train-nb3.hdf5"

# Pre processing phase
number_mels_or_mfcc = 40
padding_mels_or_mfcc = 0

# Define audio structure
num_rows = 40
num_columns = 173  # 173 for 4 Classes and 174 for 10 Classes
num_channels = 1

# Regularization rates
spatial_dropout_layer_1 = 0.07
spatial_dropout_layer_2 = 0.14
l2_rate = 0.0005

# Model variables
epochs = 150
batch_size = 8
kernel_size = (5, 5)

# Array containing audio class_names for 4 classes
audio_class_names = [
    'Car Horn',
    'Dog bark',
    'Gun Shot',
    'Siren',
]

car_horn = 0
dog_bark = 1
gun_shot = 2
siren = 3

# Test Audios file name
test_audios_files = ["gun-shot.wav", "siren.wav", "dog-bark.wav", "car-horn.wav"]
