
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras import optimizers
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from skimage import io
from skimage.transform import resize

import numpy as np
import tensorflow as tf
import random
import glob
# import coremltools

class DrunkCNN:
    batch_size = 32
    num_classes = 2
    epochs = 80
    log_filepath = './deepKerasLog'

    def __init__(self, input_shape):
        # input_shape = (nImageRows, nImageCols, nChannels)
        self.model = Sequential()

        self.model.add(Conv2D(8, kernel_size=(3,3), activation='relu', strides=(1,1), padding='same', input_shape=input_shape))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

        self.model.add(Conv2D(8, kernel_size=(3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

        # self.model.add(Conv2D(8, kernel_size=(3,3), padding='same', activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

        self.model.add(Flatten())
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(2, activation='softmax'))

        # Try out adam instead?
        sgd = optimizers.SGD(lr=.001, momentum=0.9, decay=0.000005, nesterov=False)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        self.model.summary()
    
    def fit(self):
        tensorBoardCallback = TensorBoard(log_dir=self.log_filepath, histogram_freq=0)
        callbacks = [tensorBoardCallback]

        self.model.fit(self.X_train, self.Y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(self.X_test, self.Y_test))

        score = self.model.evaluate(self.X_test, self.Y_test, verbose=0)

        print('Test loss: ', score[0])
        print('Test accuracy: ', score[1])


    def fit_generator(self):
        '''
        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=0,
            width_shift_range=0.05,
            height_shift_range=0.07,
            zoom_range=0.05,
            horizontal_flip=True)
        datagen.fit(X_train)


        model.fit_generator(datagen.flow(X_train, Y_train,batch_size=batch_size),
            steps_per_epoch=1*num_iterations,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(X_test, Y_test))
        '''

        pass
    

    def coreml(self):
        '''
        coreml_model = coremltools.converters.keras.convert(self.model, 
                                                            input_names=['image'], 
                                                            output_names=['output'], 
                                                            class_labels=['Negative', 'Drunk'],
                                                            image_input_names='image', 
                                                            image_scale=1/255.0, 
                                                            red_bias=-0.5,
                                                            green_bias=-0.5, 
                                                            blue_bias=-0.5)

        coreml_model.save('../ios/models/DrunkKerasModel.mlmodel')
        # return coreml_model
        '''
        pass


    def preprocess_images(self, nImageRows, nImageCols, nChannels):
        random_seed = 1

        tf.set_random_seed(random_seed)
        np.random.seed(random_seed)

        nCategorySamples = 4000
        positiveSamples = glob.glob('workspace/SoBr/data/resize_frontal_face/yes/*')[0:nCategorySamples]
        negativeSamples = glob.glob('workspace/SoBr/data/resize_frontal_face/no/*')[0:nCategorySamples]

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

        # mean  = X_train.mean(axis=0).mean(axis=0).mean(axis=0)
        # std   = X_train.std(axis=0).mean(axis=0).mean(axis=0)
        mean = np.array([0.5,0.5,0.5])
        std = np.array([1,1,1])

        X_train = X_train.astype('float')
        X_test = X_test.astype('float')

        for i in range(3):
            X_train[:,:,:,i] = (X_train[:,:,:,i] - mean[i]) / std[i]
            X_test[:,:,:,i] = (X_test[:,:,:,i] - mean[i]) / std[i]

        # unused 
        # num_iterations = int(len(X_train) / self.batch_size) + 1

        Y_train = to_categorical(Y_train, self.num_classes)
        Y_test = to_categorical(Y_test, self.num_classes)

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test


if __name__ == '__main__':
    nImageRows = 106
    nImageCols = 106
    nChannels = 3

    model = DrunkCNN((nImageRows, nImageCols, nChannels))
    model.preprocess_images(nImageRows, nImageCols, nChannels)
    # model.fit_generator()
    model.fit()
    # model.coreml()
