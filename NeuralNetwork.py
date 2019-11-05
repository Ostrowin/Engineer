import numpy as np
import cv2
import os
import random
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model


DATADIR = "./letters_dataset"
CATEGORIES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 
        'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
IMG_HEIGHT = 30
IMG_WIDTH = 15
training_data = []

class NeuralNetwork:
    def __init__(self):
        if(os.path.isfile('./model_e15.h5')):
            self.model = load_model('model_e15.h5')
            #print("tensorFlow: ",tf.VERSION)
            #print("keras: ", tf.keras.__version__)
        else:
            print("Error: no network to load, creating new one.")
            self.generateNetwork()
            self.model = load_model('model_e15.h5')
    def loadImage(self, path):
        img = cv2.imread('./PossibleChars/' + path, 0)
        img = cv2.bitwise_not(img)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = np.array(img).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
        return img

    def createModel(self, X, Y):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(10, 10), strides=(1, 1), activation='relu', input_shape=X.shape[1:]))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(35, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        model.fit(X, Y, batch_size=32, epochs=15)
        
        modelname = 'model_e15.h5'
        print("Saving model as " + modelname)
        model.save(modelname)

    def generateNetwork(self):
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)
            class_num = CATEGORIES.index(category)
            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                normalised_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
                training_data.append([normalised_array, class_num])
        random.shuffle(training_data)

        X = [] #feature set
        Y = [] #label set

        for features, label in training_data:
            X.append(features)
            Y.append(label)

        X = np.array(X).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
        X = X/255.0
        self.createModel(X,Y)

        
    def predict(self, path):
        #TODO: Analiza obrazków ktore mogą nie być znakami(na podstawie wartosci prawdopodobienstwa)
        answer = ''
        for img in sorted(os.listdir(path)):
            img = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            img = cv2.bitwise_not(img)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = np.array(img).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
            prediction = self.model.predict(img)
            answer = answer + str((CATEGORIES[prediction.argmax(axis=1)[0]]))
        print(path + " : " + answer)
        return answer

