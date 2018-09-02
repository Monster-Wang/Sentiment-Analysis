# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
import scipy
import h5py
import math
import sys

from PIL import Image
from numpy import *
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint,Callback

#set the image size 256*256
img_rows, img_cols = 256, 256

#set the train dataset path and the output file path
pathdir="C:/Users/WT/Desktop/ML_Train/"

path_neg = pathdir+"images/neg"
path_pos = pathdir+"images/pos"
ModelSavePath = pathdir+"Image_emotion_model.h5"
IFP =pathdir+"images/pos/p (1).jpg"


neg_num=688
pos_num=777
total_num = pos_num+neg_num# the number of all images

batch_size = 24# Size of each training and gradient update block
iepoch = 50 # iterations

Img_class = 2 #the classes of images: positive or negtive
#Build a one-dimensional array with a two-dimensional array as the element to store the matrix
Imglist=[[]] * total_num

#Function data preprocess
#load image from the directory and changeing the images format from RGB to HSV
#Input: images save path
#Output: a list of image data of format HSV
def ImagePreprocess(imgpath):
    pp = 0
    listing = os.listdir(imgpath)
    ret_list=[[]]*size(listing)
    for file in listing:  # positive图片
        imgl = [[0 for row in range(img_rows)] for col in range(img_cols)]
        im = Image.open(imgpath + '//' + file)
        img = im.resize((img_rows, img_cols))  # 调整图片尺寸
        img_array = img.load()

        for i in range(img_rows):
            for j in range(img_cols):
                r, g, b = img_array[i, j]
                h, s, v = RGBtoHSV(r, g, b)
                if h > 300 and h <= 360 or h > 0 and h <= 25:
                    h = 0
                elif h > 25 and h <= 41:
                    h = 1
                elif h > 41 and h <= 75:
                    h = 2
                elif h > 75 and h <= 156:
                    h = 3
                elif h > 156 and h <= 201:
                    h = 4
                elif h > 201 and h <= 272:
                    h = 5
                elif h > 272 and h <= 285:
                    h = 6
                elif h > 285 and h <= 330:
                    h = 7
                if s > 0.1 and s < 0.65:
                    s = 0
                elif s >= 0.65 and s <= 1:
                    s = 1
                v = 0
                imgl[i][j] = 2 * h + s + v
        ret_list[pp]=imgl
        pp += 1
        print("Load image ", pp)
    return ret_list

#format change function
def HSVtoRGB(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b

def RGBtoHSV(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mi = min(r, g, b)
    df = mx-mi
    if mx == mi:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v

def DatasetDivide(data_list):
    imnbr = len(data_list)
    print("Image dataset number：", imnbr)

    immatrix = array([array(data_list[i]).flatten() for i in range(total_num)], 'f')
    print(immatrix)
    print(immatrix.shape)

    # Initialize all to 1
    label = np.ones((total_num,), dtype=int)
    label[0:neg_num] = 0
    label[neg_num:] = 1
    # Random sorting of training data sets
    data, Label = shuffle(immatrix, label, random_state=2)
    train_data = [data, Label]

    (X, y) = (train_data[0], train_data[1])

    # Partition 90% as training set and 10% as test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=4)

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    #  type conversion
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalize the data to 0-1 because the maximum image data is 15.
    X_train /= 15
    X_test /= 15

    # Y_train have Img_class=2，keras need dataaet form: binary class matrices,
    # change form ，use the function by keras directly
    Y_train = np_utils.to_categorical(y_train, Img_class)
    Y_test = np_utils.to_categorical(y_test, Img_class)

    return X_train,X_test,Y_train,Y_test

def ImagePredictPrerocess(imgpath):

    im = Image.open(imgpath)
    imgl = [[0 for row in range(img_rows)] for col in range(img_cols)]
    img = im.resize((img_rows, img_cols))  # set the image size 256*256
    img_array = img.load()

    for i in range(img_rows):
        for j in range(img_cols):
            r, g, b = img_array[i, j]
            h, s, v = RGBtoHSV(r, g, b)
            if h > 300 and h <= 360 or h > 0 and h <= 25:
                h = 0
            elif h > 25 and h <= 41:
                h = 1
            elif h > 41 and h <= 75:
                h = 2
            elif h > 75 and h <= 156:
                h = 3
            elif h > 156 and h <= 201:
                h = 4
            elif h > 201 and h <= 272:
                h = 5
            elif h > 272 and h <= 285:
                h = 6
            elif h > 285 and h <= 330:
                h = 7
            if s > 0.1 and s < 0.65:
                s = 0
            elif s >= 0.65 and s <= 1:
                s = 1
            v = 0
            imgl[i][j] = 2 * h + s + v
    tmp_img = imgl
    ret_img = array(array(tmp_img).flatten(),'f')
    ret_img = ret_img.reshape(1, img_rows, img_cols, 1)
    ret_img = ret_img.astype('float32')
    # Normalize the data to 0-1 because the maximum image data is 15.
    ret_img /= 15
    return ret_img

def Model_Init():
    model = Sequential()  # build model
    # The first Convolution，64 filters，the size of kernel: 7*7, 1: then channel of input image (Gray channel)
    # border_mode use valid or full
    # activation function relu
    model.add(Conv2D(64, kernel_size=(7,7),strides=(2,2),
                            border_mode='valid',
                            input_shape=(img_rows, img_cols, 1)))
    convout1 = Activation('relu')
    model.add(convout1)

    # pooling，poolsize is (2,2)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # The second Convolution，128 filters，the size of kernel: 5*5
    # activation function relu
    model.add(Conv2D(128, kernel_size=(5,5),strides=(1,1),
                            border_mode='valid',
                            input_shape=(img_rows, img_cols, 1)))
    convout2 = Activation('relu')
    model.add(convout2)

    # The second Convolution，32 filters，the size of kernel: 3*3
    # activation function relu
    model.add(Conv2D(32, kernel_size=(3,3),strides=(1,1)))
    convout3 = Activation('relu')
    model.add(convout3)
    # The second Convolution，32 filters，the size of kernel: 3*3
    # activation function relu
    model.add(Conv2D(32, kernel_size=(3,3),strides=(1,1)))
    convout4 = Activation('relu')
    model.add(convout4)

    # pooling，poolsize is (2,2)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Zero the values of some elements in x by probability, and enlarge the other values 。
    # For dropout operations, to some extent prevent overfitting
    # X is a tensor, and keep_prob is a data the value between 0 and 1].
    # The probability of zero clearing for each element in x is independent of each other :1-keep_prob,
    # And the elements without zero will be multiplied by the unity 1/keep_prob,
    # The goal is to keep the overall value of x unchanged
    model.add(Dropout(0.5))

    # Full connection layer, first, the previous layer output 2-D
    # characteristic graph flatten as one-dimensional, pressure flat ready for full connection
    model.add(Flatten())
    model.add(Dense(512))  # Add full connection for 512 nodes
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(Img_class))  # add output for 2 classes
    model.add(Activation('softmax'))  # use softmax active
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

def Model_train(X_train, Y_train,X_test, Y_test):

    model = Model_Init()
    model.summary()

    #checkpointer = ModelCheckpoint(filepath=ModelSavePath, verbose=1, save_best_only=True)

    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=iepoch,
                     shuffle=True, verbose=1, validation_data=(X_test, Y_test))

    model.save(ModelSavePath)

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    Y_pred = model.predict(X_test)
    print(Y_pred)
    y_pred = np.argmax(Y_pred, axis=1)
    print(y_pred)

    target_names = ['class 0(Negative)', 'class 1(Positive)']
    print(classification_report(np.argmax(Y_test, axis=1), y_pred, target_names=target_names))

    TrainProcessPlot(hist,pathdir)

def TrainProcessPlot(history,savepath):

    print(history.history.keys())
    # summarize history for acc
    plt.plot(history.history['acc'])
    #plt.plot(history.history['val_acc'])
    fig1=plt.figure(1)
    plt.title('Accuracy of model')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(savepath+'Acc_eporch')
    print('Acc_eporch.png saved in '+savepath)
    fig1.show()
    #input()
    # summarize history for loss
    fig2=plt.figure(2)
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('Loss of model')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(savepath+"Loss_eproch")
    print('Loss_eproch.png saved in ' + savepath)
    fig2.show()
    plt.show()

    #input()

def Image_Predict(fullpath):

    model = Model_Init()
    model.load_weights(ModelSavePath,by_name=True)
    predimg=ImagePredictPrerocess(fullpath)

    pred=model.predict(predimg)
    print(pred)
    pred = np.argmax(pred, axis=1)
    print(pred)

    if pred[0]==1:
        tit = 'Positive'
    else:
        tit = 'Negtive'

    im_arr = mpimg.imread(fullpath)
    plt.title(tit)
    plt.imshow(im_arr) # plot image
    plt.axis('off')  # don't plot the axis
    plt.show()

def Main():

    if len(sys.argv)<=1:
        print('Please enter parameters correctly ！')
        sys.exit(0)
    elif len(sys.argv)==2:
        cmd = sys.argv[1]
        if cmd=='train':
            listing0=ImagePreprocess(path_neg)
            listing1=ImagePreprocess(path_pos)
            Imglist=listing0+listing1

            X_train,X_test,Y_train,Y_test = DatasetDivide(Imglist)
            print('X_train shape:', X_train.shape)
            print(X_train.shape[0], 'train samples')
            print(X_test.shape[0], 'test samples')
            print(X_train.shape[1])
            Model_train(X_train, Y_train,X_test, Y_test)
        elif cmd=='test':
            ImageFullPath=IFP
            Image_Predict(ImageFullPath)
    elif len(sys.argv)==3:
        cmd = sys.argv[1]
        ImageFullPath = sys.argv[2]
        if cmd=='test':
            Image_Predict(ImageFullPath)
        else:
            print('Please enter parameters correctly ！')
            sys.exit(0)
    else:
        print('Please enter parameters correctly ！')
        sys.exit(0)


Main()
