from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from skimage import io
from skimage.transform import resize

import numpy as np

import random
import glob

# ------------------------------ Image Preprocessing ------------------------------ # 
nCategorySamples = 4000
positiveSamples = glob.glob('../data/positives/*')[0:nCategorySamples]
negativeSamples = glob.glob('../data/negatives/*')[0:nCategorySamples]

print('Positives:', len(positiveSamples))
print('Negatives:', len(negativeSamples))

nImageRows = 106
nImageCols = 106
nChannels = 3

negativeSamples = random.sample(negativeSamples, len(positiveSamples))

X_train = []
Y_train = []

for i in range(len(positiveSamples)):
    X_train.append(resize(io.imread(positiveSamples[i]), (nImageRows, nImageCols)))
    Y_train.append(1)
    if i % 1000 == 0:
        print('Reading positive image number ', i)

for i in range(len(negativeSamples)):
    X_train.append(resize(io.imread(negativeSamples[i]), (nImageRows, nImageCols)))
    Y_train.append(0)
    if i % 1000 == 0:
        print('Reading negative image number ', i)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.30, random_state=42)

mean = np.array([0.5, 0.5, 0.5])
std = np.array([1, 1, 1])

X_train = X_train.astype('float')
X_test = X_test.astype('float')

for i in range(3):
    X_train[:,:,:,i] = (X_train[:,:,:,i] - mean[i]) / std[i]
    X_test[:,:,:,i] = (X_test[:,:,:,i] - mean[i]) / std[i]

num_classes = 2

Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.07,
    shear_range=0.7,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest')

datagen.fit(X_train)

NUM_IMAGES = 20
for i, batch in enumerate(datagen.flow(X_train, Y_train, batch_size=1,
                          save_to_dir='../preview', save_prefix='drunk', save_format='jpeg')):
    if i > NUM_IMAGES:
        break  # otherwise the generator would loop indefinitely