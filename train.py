from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import SeparableConv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras import callbacks
import matplotlib
matplotlib.use('TkAgg')
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adagrad
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
from KernelConv import *

class Model:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        channelDim = -1

        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
            channelDim = 1


        model.add(KernelConv2D(32,3,padding='same',input_shape=inputShape))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(KernelConv2D(64,3,padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(KernelConv2D(64, 3, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(KernelConv2D(128,3, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(KernelConv2D(128, 3, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(KernelConv2D(128,3, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation('softmax'))
        return model

if __name__ == '__main__':
    # obtain path to data and store in list
    refined_data_path = 'data/400x'  # for training, testing, validataion data
    train_path = os.path.sep.join([refined_data_path, 'train'])
    test_path = os.path.sep.join([refined_data_path, 'test'])
    val_path = os.path.sep.join([refined_data_path, 'val'])

    train_list = list(paths.list_images(train_path))
    val_list = list(paths.list_images(val_path))
    test_list = list(paths.list_images(test_path))

    # number of images
    number_of_train = len(train_list)
    number_of_val = len(val_list)
    number_of_test = len(test_list)

    # accounting for skew in labeled data
    trainLabels = [int(im.split(os.path.sep)[-2]) for im in train_list]
    trainLabels = np_utils.to_categorical(trainLabels)  # convert array of label data to one-hot vector
    classTotal = trainLabels.sum(axis=0)
    classWeight = classTotal.max()/classTotal

    numEpochs = 3
    lrRate = 1e-2
    lrRateDecay = lrRate/numEpochs
    batchSize = 32

    # initializing training data augmentation object
    trainAug = ImageDataGenerator(rotation_range=20,
                                  rescale=1/255.0,
                                  zoom_range=0.05,
                                  height_shift_range=0.1,
                                  width_shift_range=0.1,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  shear_range=0.5,
                                  fill_mode='nearest')

    # initializing validation data augmentation object
    valAug = ImageDataGenerator(rescale=1/255.0)

    # initializing training, validation and testing generator
    trainGen = trainAug.flow_from_directory(train_path,
                                            class_mode='categorical',
                                            target_size=(48,48),
                                            shuffle=True,
                                            batch_size=batchSize,
                                            color_mode='rgb')

    valGen = valAug.flow_from_directory(val_path,
                                        class_mode='categorical',
                                        target_size=(48,48),
                                        shuffle=False,
                                        batch_size=batchSize,
                                        color_mode='rgb')

    testGen = valAug.flow_from_directory(test_path,
                                         class_mode='categorical',
                                         target_size=(48,48),
                                         shuffle=False,
                                         batch_size=batchSize,
                                         color_mode='rgb')

    model = Model.build(width=48, height=48, depth=3, classes=2)
    opt = Adagrad(lr=lrRate, decay=lrRateDecay)
    # compile the model
    # compiling with binary_crossentropy loss function as 2 classes of data
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    log_dir = './tf-log/'
    tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
    cbks = [tb_cb]
    # to store the model after every epoch
    #callbacks = [ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5')]


    # fit the model
    history = model.fit_generator(trainGen,
                                  validation_data=valGen,
                                  steps_per_epoch=number_of_train // batchSize,
                                  validation_steps=number_of_val // batchSize,
                                  class_weight=classWeight,
                                  epochs=numEpochs,
                                  callbacks=cbks)
    print('Training complete')
    print('Evaluating network')
    testGen.reset()
    # make prediction on test data
    predIdx = model.predict_generator(testGen,
                                      steps=(number_of_test//batchSize) + 1)

    # grab the highest prediction indices in each sample
    predIdx = np.argmax(predIdx, axis=1)
    # print the classificaiton report
    print(classification_report(testGen.classes,
                                predIdx,
                                target_names=testGen.class_indices.keys()))

    # compute confusion matrix
    confusionMatrix = confusion_matrix(testGen.classes, predIdx)
    total = sum(sum(confusionMatrix))
    # compute accuracy, sensitivty, specificity
    # sensitivity measures the proportion of true positives also predicted as positives
    # Similarly, specificity measures the proportion of true negatives
    accuracy = (confusionMatrix[0,0] + confusionMatrix[1,1]) / total
    sensitivity = confusionMatrix[0,0] / (confusionMatrix[0,0] + confusionMatrix[0,1])
    specificity = confusionMatrix[1,1] / (confusionMatrix[1,0] + confusionMatrix[1,1])

    print(confusionMatrix)
    print('accuracy: {:4f}'.format(accuracy))
    print('sensitivity: {:4f}'.format(sensitivity))
    print('specificity: {:4f}'.format(specificity))
    N = numEpochs
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0,N), history.history['loss'], label='train_loss')
    plt.plot(np.arange(0, N), history.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, N), history.history['acc'], label='train_acc')
    plt.plot(np.arange(0, N), history.history['val_acc'], label='val_acc')

    plt.title('Training loss and accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.legend(loc='upper right')
    #plt.savefig('plot400demox.png')

    #model_json = model.to_json()
    #with open('model400x.json', 'w') as json_file:
        #json_file.write(model_json)

    #model.save_weights('model400x.h5')
    print('model json saved. model weights saved')