import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

train_df = pd.read_csv("C:/Users/Harish/Desktop/hand_gestures/sign_mnist_train/sign_mnist_train.csv")
trainy=np.array(train_df.pop('label'))
test_df = pd.read_csv("C:/Users/Harish/Desktop/hand_gestures/sign_mnist_test/sign_mnist_test.csv")
testy = np.array(test_df.pop('label'))
print(test_df,testy)
'''img=train_df[:1]
img=np.array(img)
img1=img[0]
img=np.resize(img1,(28,28))
print(img)
plt.imshow(img)
#cv2.imshow("img",img)
imgplot = plt.imshow(img)
plt.show()
cv2.waitKey('q')'''