from keras import datasets, layers, models,losses
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import os

def renmae():
    train_df = pd.read_csv("C:/Users/Harish/Desktop/hand_gestures/sign_mnist_train/sign_mnist_train.csv")
    trainy = np.array(train_df.pop('label'))
    test_df = pd.read_csv("C:/Users/Harish/Desktop/hand_gestures/sign_mnist_test/sign_mnist_test.csv")
    testy = np.array(test_df.pop('label'))
    return train_df,trainy,test_df,testy
    print("preparing data done")




base='C:/Users/Harish/Desktop/poycharm projects/projects/ml/data/JPEG'
#'''C:\Users\Harish\Desktop\poycharm projects\projects\ml\data\JPEG'''
train_df,trainy,test_df,testy=renmae()
train_df=train_df/255
test_df=test_df/255



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(172, 264, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(13))

model.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])





history = model.fit(train_df, trainy, epochs=6,
                    validation_data=(test_df[0:1000],testy[0:1000]))

test_loss, test_acc = model.evaluate(test_df[1001:], testy[1001:], verbose=2)
print(test_acc)


