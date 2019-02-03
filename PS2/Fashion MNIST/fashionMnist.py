import numpy as np
import keras
from mnist import MNIST #python-mnist package (available only through pip)
import os

#data from https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion 
#must be uncompresed without having been renamed
data = MNIST(os.getcwd())

trainImages, trainLabels = data.load_training()
testImages, testLabels = data.load_testing()

trainImages = np.reshape(trainImages,(60000,1,28,28)) #correct orientation, you can view the images and see below
testImages = np.reshape(testImages,(10000,1,28,28))

trainImages = trainImages.astype('uint8') #images loaded in as int64, 0 to 255 integers
testImages = testImages.astype('uint8')

"""
from matplotlib import pyplot as plt
plt.imshow(trainImages[7][0,:,:], interpolation='nearest') 
plt.show()
"""


trainLabels=keras.utils.to_categorical(trainLabels) #these preserve dtype
testLabels=keras.utils.to_categorical(testLabels)

trainLabels = trainLabels.astype('uint8')
testLabels = testLabels.astype('uint8')

np.save('trainImages.npy',trainImages)
np.save('testImages.npy',testImages)
np.save('trainLabels.npy',trainLabels)
np.save('testLabels.npy',testLabels)