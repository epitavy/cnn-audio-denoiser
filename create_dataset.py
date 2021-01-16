from data_processing.dataset import Dataset
import random
import warnings
import os

warnings.filterwarnings(action='ignore')
dataset_path = '../DroneBot_Audio_Files/dataset'
data_files = os.listdir(dataset_path)
clean_filenames = [os.path.join(dataset_path, f) for f in data_files if f.startswith('clean')]
noise_filenames = [os.path.join(dataset_path, f) for f in data_files if f.startswith('noise')]


clean_filenames.sort()
noise_filenames.sort()

clean_to_clean_ratio = 10

out_filenames = clean_filenames# + clean_filenames[::clean_to_clean_ratio]
in_filenames = noise_filenames# + clean_filenames[::clean_to_clean_ratio]

def shuffle(a, b):
    assert len(a) == len(b)
    indexes = list(range(len(a)))
    random.shuffle(indexes)

    sa, sb = [], []
    for i in indexes:
        sa.append(a[i])
        sb.append(b[i])

    return sa, sb

in_filenames, out_filenames = shuffle(in_filenames, out_filenames)

windowLength = 256
config = {'windowLength': windowLength,
          'overlap': round(0.25 * windowLength),
          'fs': 16000,
          'audio_max_duration': 0.8}

N_TEST_FILE = 20

# Quick fix to remove augmented data from validation set
val_out_filenames = [f for f in out_filenames[-N_TEST_FILE:] if len(f.rsplit('/')[-1]) <= 12]
val_in_filenames = [f for f in in_filenames[-N_TEST_FILE:] if len(f.rsplit('/')[-1]) <= 12]

val_dataset = Dataset(val_out_filenames, val_in_filenames[-N_TEST_FILE:], **config)
val_dataset.create_tf_record(prefix='val', subset_size=2000, parallel=False)

train_dataset = Dataset(out_filenames[:-N_TEST_FILE], in_filenames[:-N_TEST_FILE], **config)
train_dataset.create_tf_record(prefix='train', subset_size=4000, parallel=False)
"""
## Create Test Set
clean_test_filenames = mcv.get_test_filenames()

noise_test_filenames = us8K.get_test_filenames()

test_dataset = Dataset(clean_test_filenames, noise_test_filenames, **config)
tet_dataset.create_tf_record(prefix='test', subset_size=1000, parallel=False)
"""
