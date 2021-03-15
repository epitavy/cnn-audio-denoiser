from data_processing.dataset import Dataset
import random
import warnings
import os

VALIDATION_SPLIT = 0.8

warnings.filterwarnings(action='ignore')
dataset_path = '../DroneBot_Audio_Files/dataset'
data_files = os.listdir(dataset_path)

# Check with double underscore '__' prevent data from brikair to be used, keep it for test data
clean_filenames = [os.path.join(dataset_path, f) for f in data_files if f.startswith('clean') and not f.startswith('clean__')]
noise_filenames = [os.path.join(dataset_path, f) for f in data_files if f.startswith('noise') and not f.startswith('noise__')]

# Make clean files match with their noise file
clean_filenames.sort()
noise_filenames.sort()

def shuffle(a, b):
    assert len(a) == len(b)
    indexes = list(range(len(a)))
    random.shuffle(indexes)

    sa, sb = [], []
    for i in indexes:
        sa.append(a[i])
        sb.append(b[i])

    return sa, sb

clean_filenames, noise_filenames = shuffle(clean_filenames, noise_filenames)

windowLength = 256
config = {'windowLength': windowLength,
          'overlap': round(0.25 * windowLength),
          'fs': 16000,
          'numSegments': 8,
          'audio_max_duration': 0.8}

split = int(len(clean_filenames) * VALIDATION_SPLIT)
val_dataset = Dataset(clean_filenames[split:], noise_filenames[split:], **config)
val_dataset.create_tf_record(prefix='val', subset_size=2000)

train_dataset = Dataset(clean_filenames[:split], noise_filenames[:split], **config)
train_dataset.create_tf_record(prefix='train', subset_size=4000)
