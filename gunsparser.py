import librosa
import pandas as pd
import os
import time
import numpy as np
start = time.clock()

print('Doing work, hold tight')

# Define file locations
file_location = os.path.join('/Users', 'olalansman', 'schoolprojects', 'data_science', 'GunShotRecognizer', 'urban-sound-classification', 'train')
labels = pd.read_csv(os.path.join(file_location, "train.csv"))
files = os.listdir(os.path.join(file_location, 'Train'))

# Initialize list for gathering of rows before creating the dataframe to speed up the loop
rows_list = []

for file, label in zip(files, labels['Class']):
    data, sample_rate = librosa.load(os.path.join(file_location, 'Train', file), res_type='kaiser_fast')

    # Extract features and transpose resulting matrix for fixed size features layer for ANN
    feat = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)

    # Save dataframe rows as dictionaries for row list
    cols = {}
    cols.update({'feature': feat, 'label': label})
    rows_list.append(cols)

df = pd.DataFrame(rows_list)
df.to_pickle('dataset.pkl')


print('Done')
end = time.clock()
print("Took %.1f minutes" % ((end - start) / 60))
