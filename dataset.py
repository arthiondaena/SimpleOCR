import numpy as np
import cv2
from tensorflow.keras.datasets import mnist

def load_mnist_dataset():
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
    data = np.vstack([trainData, testData])
    labels = np.hstack([trainLabels, testLabels])
    return (data, labels)

def load_az_dataset(datasetPath):
    data=[]
    labels=[]

    for row in open(datasetPath):
        row=row.split(',')
        label = int(row[0])
        image = np.array([int(x) for x in row[1:]], dtype='uint8')
        image = image.reshape((28, 28))
        data.append(image)
        labels.append(label)
    
    data = np.array(data, dtype='float32')
    labels = np.array(labels, dtype='int')

    return (data, labels)

def dataset():
    (digitsData, digitsLabels) = load_mnist_dataset()
    (azData, azLabels) = load_az_dataset('data/A_Z Handwritten Data.csv')

    #the MNIST dataset occupies the labels 0-9, adding 10 so azLabels are not interfered
    azLabels += 10

    data = np.vstack([azData, digitsData])
    labels = np.hstack([azLabels, digitsLabels])

    # Each image in the A-Z and MNIST digts datasets are 28x28 pixels;
    # However, the architecture we're using is designed for 32x32 images,
    # So we need to resize them to 32x32

    data = [cv2.resize(image, (32, 32)) for image in data]
    data = np.array(data, dtype="float32")

    data = np.expand_dims(data, axis=-1)
    data /=255.0

    return (data, labels)