import flwr as fl
import threading
# from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import pygad.kerasga
import time
import sys
import numpy as np
import random
import string
import socket
import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import average
from numpy import array
import pickle
from noknow.core import ZK, ZKSignature, ZKParameters, ZKData, ZKProof
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
# Synthetic dataset
from sklearn.datasets import make_classification
# Data processing
import pandas as pd
import numpy as np
from collections import Counter
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Model and performance
import tensorflow as tf
from tensorflow.keras import layers, losses
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from tensorflow.keras.optimizers import Adam
oldw=None

def getData(dist, x, y):
    dx = []
    dy = []
    counts = [0 for i in range(10)]
    for i in range(len(x)):
        if counts[y[i]] < dist[y[i]]:
            dx.append(x[i])
            dy.append(y[i])
            counts[y[i]] += 1

    return np.array(dx), np.array(dy)



# Load and compile Keras model
globalmodel = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

globalmodel.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0
dist = [4000, 4000, 4000, 3000, 10, 10, 10, 10, 4000, 10]
x_train, y_train = getData(dist, x_train, y_train)
train_data=x_train
test_data=x_test
# dataset=([tf.keras.datasets.mnist.load_data()])
# data_outputs=tf.keras.datasets.mnist.labels.load_data()
data_inputs=np.concatenate((x_train,x_test), axis=0)
data_outputs=np.concatenate((y_train,y_test), axis=0)


class SocketThread(threading.Thread):

    def __init__(self, buffer_size, recv_timeout,port):
        threading.Thread.__init__(self)
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout
        self.port=port
        self.soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.soc.bind(("192.168.56.1" , self.port))
        print("Socket is bound to an address & port number.")
        self.soc.listen(1)
        self.connection, client_info = self.soc.accept()

    def run(self):
        print("Running a Thread for the Connection with client")

        # This while loop allows the server to wait for the client to send data more than once within the same connection.
        while True:
            self.recv_start_time = time.time()
            time_struct = time.gmtime()
            date_time = "Waiting to Receive Data Starting from {day}/{month}/{year} {hour}:{minute}:{second} GMT".format(year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
            # print(date_time)
            received_data, status = self.recv()

            # print(received_data)
            self.reply(received_data)

    def recv(self):
        print("nnnn")
        received_data = b""
        while True:
            try:
                data = self.connection.recv(self.buffer_size)
                received_data += data

                try:
                    pickle.loads(received_data)
                    print("b55")
                    # If the previous pickle.loads() statement is passed, this means all the data is received.
                    # Thus, no need to continue the loop. The flag all_data_received_flag is set to True to signal all data is received.
                    all_data_received_flag = True
                except BaseException:
                    # An exception is expected when the data is not 100% received.
                    pass

                if data == b'': # Nothing received from the client.
                    received_data = b""
                    # If still nothing received for a number of seconds specified by the recv_timeout attribute, return with status 0 to close the connection.
                    if (time.time() - self.recv_start_time) > self.recv_timeout:
                        print("nothing received")# 0 means the connection is no longer active and it should be closed.

                elif all_data_received_flag:
                    # print("All data ({data_len} bytes) Received from {client_info}.".format(client_info=self.client_info, data_len=len(received_data)))
                    if len(received_data) > 0:
                        print('hnn')
                        try:
                            # Decoding the data (bytes).
                            received_data = pickle.loads(received_data)
                            # Returning the decoded data.
                            return received_data, 1

                        except BaseException as e:
                            print("Error Decoding the Client's Data:")

                else:
                    # In case data are received from the client, update the recv_start_time to the current time to reset the timeout counter.
                    self.recv_start_time = time.time()

            except BaseException as e:
                print("Error Receiving Data from the Client: ")



    def reply(self, received_data):
        global keras_ga,data_inputs, data_outputs,globalmodel
        # if (("data" in received_data.keys()) and ("subject" in received_data.keys())):
        #     subject = received_data["subject"]
        #  msg_model = received_data
            # print("Client's Message Subject is {subject}.".format(subject=subject))
    
        print("Replying to the Client.")
        if received_data is None:
            print("nothong received from client")
            # data_dict = {"population_weights": keras_ga.population_weights,
            #                 "model_json": globalmodel.to_json(),
            #                 "num_solutions": keras_ga.num_solutions}
            # data = {"subject": "model", "data": data_dict}
        else:
            predictions = globalmodel.predict(data_inputs)
            ba = tf.keras.metrics.BinaryAccuracy()
            ba.update_state(data_outputs, predictions)
            accuracy = ba.result().numpy()
            if accuracy == 1.0:
                print("model is updated correctly")
                return
            else:
                try:
                    response = pickle.dumps(globalmodel)
                except BaseException as e:
                    print("Error Encoding the Message: {msg}.\n".format(msg=e))
                try:
                    best_model_weights_vector = received_data
                    best_model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=globalmodel,
                                                                                                weights_vector=best_model_weights_vector)
                except BaseException as e:
                    print("reply(): Error Decoding the Client's Data: {msg}.\n".format(msg=e))
        self.model_averaging(globalmodel, best_model_weights_matrix)
        predictions = globalmodel.predict(data_inputs)
        print("Model Predictions: {predictions}".format(predictions=predictions))

        ba = tf.keras.metrics.BinaryAccuracy()
        ba.update_state(data_outputs, predictions)
        accuracy = ba.result().numpy()
        print("Accuracy = {accuracy}\n".format(accuracy=accuracy))
        if accuracy != 1.0:

            response = pickle.dumps(globalmodel)
        else:
            print("done")
            return
        try:
            self.connection.sendall(response)
        except BaseException as e:
            print("Error Sending Data to the Client: {msg}.\n".format(msg=e))


    def model_averaging(self, model, best_model_weights_matrix):
        model_weights_vector = pygad.kerasga.model_weights_as_vector(model=model)
        model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model,
                                                                     weights_vector=model_weights_vector)

        # new_weights = numpy.array(model_weights_matrix + best_model_weights_matrix)/2
        new_weights = model_weights_matrix
        print(new_weights)
        for idx, arr in enumerate(new_weights):
            new_weights[idx] = new_weights[idx] + best_model_weights_matrix[idx]
            new_weights[idx] = new_weights[idx] / 2
        x=model_weights_matrix-oldw
        H=(oldw+np.linalg.det(model_weights_matrix-oldw))
        n1=model_weights_matrix-oldw
        new=globalmodel+np.dot(H,n1)

        globalmodel.set_weights(weights=new_weights)
        oldw=model_weights_matrix
        


port= 5680
# for i in range(8):
listenThread = SocketThread(buffer_size=1024*1024*1024,
                                            recv_timeout=10,port=5680)
listenThread.start()
print("Listening for incoming connection ...")
port+=1
