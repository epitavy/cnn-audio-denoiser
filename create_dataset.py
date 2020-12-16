from data_processing.dataset import Dataset
import warnings
import os

warnings.filterwarnings(action='ignore')
dataset_path = '../DroneBot_Audio_Files/dataset'
data_files = os.listdir(dataset_path)
clean_train_filenames = [os.path.join(dataset_path, f) for f in data_files if f.startswith('out_train')]
clean_val_filenames = [os.path.join(dataset_path, f) for f in data_files if f.startswith('out_val')]
noise_train_filenames = [os.path.join(dataset_path, f) for f in data_files if f.startswith('in_train')]
noise_val_filenames = [os.path.join(dataset_path, f) for f in data_files if f.startswith('in_val')]

clean_train_filenames.sort()
noise_train_filenames.sort()
clean_val_filenames.sort()
noise_val_filenames.sort()

windowLength = 256
config = {'windowLength': windowLength,
          'overlap': round(0.25 * windowLength),
          'fs': 16000,
          'audio_max_duration': 0.8}

val_dataset = Dataset(clean_val_filenames, noise_val_filenames, **config)
val_dataset.create_tf_record(prefix='val', subset_size=2000, parallel=False)

train_dataset = Dataset(clean_train_filenames, noise_train_filenames, **config)
train_dataset.create_tf_record(prefix='train', subset_size=4000, parallel=False)
"""
## Create Test Set
clean_test_filenames = mcv.get_test_filenames()

noise_test_filenames = us8K.get_test_filenames()

test_dataset = Dataset(clean_test_filenames, noise_test_filenames, **config)
tet_dataset.create_tf_record(prefix='test', subset_size=1000, parallel=False)
"""
