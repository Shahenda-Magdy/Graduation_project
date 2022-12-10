import flwr as fl
import sys
import numpy as np
import random
import string
import socket
import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from noknow.core import ZK, ZKSignature, ZKParameters, ZKData, ZKProof
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model


SERVER_IP = '192.168.20.1'
SERVER_PORT = 5678

#     s.send(b'Done')

# Define Flower client

signature=''
x='client2'
x=x.encode('utf-8')
with socket.socket(socket.AF_INET , socket.SOCK_STREAM) as s:
    s.connect((SERVER_IP, SERVER_PORT))
    print(f'registering')
    data = s.recv(1024)
    data=data.decode('utf-8')
    signature=data.encode('utf-8')
    s.send(x)
with socket.socket(socket.AF_INET , socket.SOCK_STREAM) as s:
    s.connect((SERVER_IP, SERVER_PORT))
#     s.send(b'authenticate please')
    s.send(signature)

# cifar10 = tf.keras.datasets.cifar10
 
# # Distribute it to train and test set
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# # Reduce pixel values
# x_train, x_test = x_train / 255.0, x_test / 255.0

# # flatten the label values
# y_train, y_test = y_train.flatten(), y_test.flatten()
# # visualize data by plotting images
# fig, ax = plt.subplots(5, 5)
# k = 0

# for i in range(5):
#     for j in range(5):
#         ax[i][j].imshow(x_train[k], aspect='auto')
#         k += 1

# plt.show()
# # number of classes
# K = len(set(y_train))

# # calculate total number of classes
# # for output layer
# print("number of classes:", K)

# # Build the model using the functional API
# # input layer
# i = Input(shape=x_train[0].shape)
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
# x = BatchNormalization()(x)
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D((2, 2))(x)

# x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D((2, 2))(x)

# x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D((2, 2))(x)

# x = Flatten()(x)
# x = Dropout(0.2)(x)

# # Hidden layer
# x = Dense(1024, activation='relu')(x)
# x = Dropout(0.2)(x)

# # last hidden layer i.e.. output layer
# x = Dense(K, activation='softmax')(x)

# model = Model(i, x)

# # model description
# # model.summary()


# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model=[1,2,3,4,5,6,7]
class client2(fl.client.NumPyClient):
    def get_parameters(config):
        return model.get_weights()

    def fit(parameters, config):
        model.set_weights(parameters)
        r = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=0)
        hist = r.history
        print("Fit history : " ,hist)
        return model.get_weights(), len(x_train), {}

    def evaluate(parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("Eval accuracy : ", accuracy)
        return loss, len(x_test), {"accuracy": accuracy}
# for i in range(4):
soc = socket.socket()
print("Socket is created.")

soc.connect(("192.168.20.1" , 5678))
print("Connected to the server.")

model = pickle.dumps(model)
soc.sendall(model)
print("Client sent a message to the server.")

# received_data = soc.recv(8)
# received_data = pickle.loads(received_data)
# print("Received data from the client: {received_data}".format(received_data=received_data))

soc.close()
print("Socket closed.")
