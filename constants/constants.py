# Define paths used
csv_path = "metadata/UrbanSound8K.csv"
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
epochs = 20
batch_size = 2
kernel_size = (5, 5)

# Array containing audio class_names for 10 classes
# audio_class_names = [
#     'Air Conditioner',
#     'Car Horn',
#     'Children Playing',
#     'Dog bark',
#     'Drilling',
#     'Engine Idling',
#     'Gun Shot',
#     'Jackhammer',
#     'Siren',
#     'Street Music'
# ]

# Array containing audio class_names for 4 classes
audio_class_names = [
    'Car Horn',
    'Dog bark',
    'Gun Shot',
    'Siren',
]

# Test Audios file name
test_audios_files = ["Gun-Shot.wav", "Siren.wav", "Dog-Bark.wav", "Car-Horn.wav"]
