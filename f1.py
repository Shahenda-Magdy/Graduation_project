import flwr as fl
import threading
import math
# import mxnet as mx
# from mxnet import nd, autograd, gluon
# import numpy as np
# from copy import deepcopy
# import time
# from sklearn.metrics import roc_auc_score
# from keras.utils import to_categorical
from scipy.spatial import distance
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
# from numpy import genfromtxt
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
# Synthetic dataset
from sklearn.datasets import make_classification
# Data processing
import pandas as pd
import numpy as np
# from collections import Counter
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




# def getData(dist, x, y):
#     dx = []
#     dy = []
#     counts = [0 for i in range(10)]
#     for i in range(len(x)):
#         if counts[y[i]] < dist[y[i]]:
#             dx.append(x[i])
#             dy.append(y[i])
#             counts[y[i]] += 1

#     return np.array(dx), np.array(dy)



# Load and compile Keras model
# globalmodel = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(256, activation='relu'),
#     keras.layers.Dense(10, activation='softmax')
# ])

# globalmodel.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# # Load dataset
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0
# dist = [4000, 4000, 4000, 3000, 10, 10, 10, 10, 4000, 10]
# x_train, y_train = getData(dist, x_train, y_train)
# train_data=x_train
# test_data=x_test
# dataset=([tf.keras.datasets.mnist.load_data()])
# data_outputs=tf.keras.datasets.mnist.labels.load_data()
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense
from keras.optimizers import Adam
#because of multiclass datasets
from keras.utils.np_utils import to_categorical 
import random
(x_train, y_train), (x_test, y_test) = mnist.load_data()
assert(x_train.shape[0] == y_train.shape[0]), "The number of images is not equal .."
assert(x_test.shape[0] == y_test.shape[0]), "The number of images is not equal .."
assert(x_train.shape[1:] == (28, 28)), "The dimension of the images are not 28x28"
assert(x_test.shape[1:] == (28, 28))
num_of_samples = []

cols = 5
num_of_classes = 10

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10) 
x_train = x_train/255 
x_test = x_test/255
num_pixels = 784
x_train = x_train.reshape(x_train.shape[0],
                         num_pixels)
x_test = x_test.reshape(x_test.shape[0],
                         num_pixels)
print(x_train.shape)
def create_model():
  model = Sequential()
  model.add(Dense(10, input_dim = num_pixels,
                  activation = 'relu'))
  model.add(Dense(30, activation='relu'))
  model.add(Dense(10, activation='relu'))
  model.add(Dense(num_of_classes, activation='softmax'))
  model.compile(Adam(lr=0.01),
                loss='categorical_crossentropy',
               metrics=['accuracy'])
  return model           
     

globalmodel = create_model()
# print(model.summary())
# history = model.fit(x_train, y_train, validation_split=0.1,
#          epochs=10, batch_size=200, verbose=1, shuffle=1)

score = globalmodel.evaluate(x_test, y_test, verbose=0)
print(type(score))
print('Test Score:', score[0])
print('Test Accuracy:', score[1])
     
# data_inputs=np.concatenate((x_train,x_test), axis=0)
# data_outputs=np.concatenate((y_train,y_test), axis=0)
# print(data_inputs.shape)
# print(data_outputs.shape)    
# data_inputs=np.concatenate((x_train,x_test), axis=0)
# data_outputs=np.concatenate((y_train,y_test), axis=0)
# print(data_inputs.shape)
# print(data_outputs.shape)

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
            received_data = self.recv()

            # print(received_data)
            self.reply(received_data)

    def recv(self):
        print("nnnn")
        received_data = b""
        while True:
            try:
                data = self.connection.recv(self.buffer_size)
                received_data = data

                # try:
                #     pickle.loads(received_data)
                #     print("b55")
                #     # If the previous pickle.loads() statement is passed, this means all the data is received.
                #     # Thus, no need to continue the loop. The flag all_data_received_flag is set to True to signal all data is received.
                #     all_data_received_flag = True
                # except BaseException:
                #     # An exception is expected when the data is not 100% received.
                #     pass

                # if data == b'': # Nothing received from the client.
                # received_data = b""
                # If still nothing received for a number of seconds specified by the recv_timeout attribute, return with status 0 to close the connection.
                if (time.time() - self.recv_start_time) > self.recv_timeout:
                    print("nothing received")
                    return # 0 means the connection is no longer active and it should be closed.

                    # print("All data ({data_len} bytes) Received from {client_info}.".format(client_info=self.client_info, data_len=len(received_data)))
                if len(received_data) > 0:
                    print('hnn')
                    try:
                        # Decoding the data (bytes).
                        received_data = pickle.loads(received_data)
                        # Returning the decoded data.
                        return received_data

                    except BaseException as e:
                        print("Error Decoding the Client's Data:")
                        return

                else:
                    # In case data are received from the client, update the recv_start_time to the current time to reset the timeout counter.
                    self.recv_start_time = time.time()

            except BaseException as e:
                print("Error Receiving Data from the Client: ")
                return

        return received_data

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
            score = globalmodel.evaluate(x_test, y_test, verbose=0)
            print(type(score))
            print('Test Score:', score[0])
            print('Test Accuracy:', score[1])
            if score[1] == 1.0:
                print("model is updated correctly")
                self.soc.close()
                return
            else:
                try:
                    response = pickle.dumps(globalmodel)
                except BaseException as e:
                    print("Error Encoding the Message: {msg}.\n".format(msg=e))
                # try:
                #     data_dict = {
                #                     "model_json": globalmodel.to_json()}
                #     data = {"subject": "model", "data": data_dict}
                #     best_model_weights_vector = received_data.get_weights()
                #     best_model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=globalmodel.get_weights(),
                #                                                                                 weights_vector=best_model_weights_vector)
                # except BaseException as e:
                #     print("reply(): Error Decoding the Client's Data: {msg}.\n".format(msg=e))
        self.model_averaging(globalmodel, received_data)
        score = globalmodel.evaluate(x_test, y_test, verbose=0)
        print(type(score))
        print('Test Score:', score[0])
        print('Test Accuracy:', score[1])
        if score[1] != 1.0:

            response = pickle.dumps(globalmodel)
        else:
            self.soc.close()
            print("done")
            return
        try:
            self.connection.sendall(response)
        except BaseException as e:
            print("Error Sending Data to the Client: {msg}.\n".format(msg=e))


    def model_averaging(self, model, received):
        model_weights_vector = pygad.kerasga.model_weights_as_vector(model=model)
        model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model,
                                                                     weights_vector=model_weights_vector)
        best_model_weights_vector = pygad.kerasga.model_weights_as_vector(model=received)
        best_model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=received,
                                                                     weights_vector=best_model_weights_vector)

        # new_weights = np.array(model_weights_matrix + best_model_weights_matrix)/2
        new_weights = model_weights_matrix
        print(type(new_weights))
        for idx, arr in enumerate(new_weights):
            new_weights[idx] = new_weights[idx] + best_model_weights_matrix[idx]
            new_weights[idx] = new_weights[idx] / 2
        N=10
        h=np.empty((0,0))
        for i in range(0, len(model.get_weights()) // N):
        
            # getting incremented chunks
            h.append(model.get_weights()[0: (i + 1) * N])
        x=[]
        zip_object = zip(new_weights, model.get_weights())

        for i, k in zip_object:
            x.append(i - k)
        
  
        re =np.empty((0,0))
        for i in range(0, len(x) // N):
        
            # getting incremented chunks
            re.append(x[0: (i + 1) * N])
        z=np.linalg.det(re)
        h=+z
            
        c=np.dot(h,x)
        g=model=+c
        distance = np.norm((nd.concat(*new_weights, dim=1) - np.concat(*g, dim=1)), axis=0).asnumpy()
        distance = np.norm((nd.concat(*new_weights, dim=1) - np.concat(*g, dim=1)), axis=0).asnumpy()
        distance = distance / np.sum(distance)
        print(distance)
        # g=g.tolist()
        # for i in range(0, len(g) // N):
        
        #     # getting incremented chunks
        # #     g.append(g[0: (i + 1) * N])
        # # print(new_weights.shape)
        # globalmodel.set_weights(weights=new_weights)
        # # new_weights=np.reshape(10,30)
        # dist=np.linalg.norm(new_weights-g)
        # # dist=0
        # # for i in range(0,len(g)):
        # #     dist=+new_weights[i]-g[i]
        # # dist=dist//len(g)


        # print(dist)
        # globalmodel.set_weights(new_weights)
        
        # print(new_weights)
#     def simple_mean(old_gradients, param_list, net, lr, b=0, hvp=None):
#         if hvp is not None:
#             pred_grad = []
#             distance = []
#             for i in range(len(old_gradients)):
#                 pred_grad.append(old_gradients[i] + hvp)
#             #distance.append((1 - nd.dot(pred_grad[i].T, param_list[i]) / (
#                         #nd.norm(pred_grad[i]) * nd.norm(param_list[i]))).asnumpy().item())

#             pred = np.zeros(100)
#             pred[:b] = 1
#             distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
#             auc1 = roc_auc_score(pred, distance)
#             distance = nd.norm((nd.concat(*pred_grad, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
#             auc2 = roc_auc_score(pred, distance)
#             print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))

#             #distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
#             #distance = nd.norm(nd.concat(*param_list, dim=1), axis=0).asnumpy()
#             # normalize distance
#             distance = distance / np.sum(distance)
#         else:
#             distance = None

#     mean_nd = nd.mean(nd.concat(*param_list, dim=1), axis=-1, keepdims=1)

#     idx = 0
#     for j, (param) in enumerate(net.collect_params().values()):
#         if param.grad_req == 'null':
#             continue
#         param.set_data(param.data() - lr * mean_nd[idx:(idx + param.data().size)].reshape(param.data().shape))
#         idx += param.data().size
#     return mean_nd, distance


# # trimmed mean
# def trim(old_gradients, param_list, net, lr, b=0, hvp=None):
#     '''
#     gradients: the list of gradients computed by the worker devices
#     net: the global model
#     lr: learning rate
#     byz: attack
#     f: number of compromised worker devices
#     b: trim parameter
#     '''
#     if hvp is not None:
#         pred_grad = []
#         for i in range(len(old_gradients)):
#             pred_grad.append(old_gradients[i] + hvp)

#         pred = np.zeros(100)
#         pred[:b] = 1
#         distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
#         auc1 = roc_auc_score(pred, distance)
#         distance = nd.norm((nd.concat(*pred_grad, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
#         auc2 = roc_auc_score(pred, distance)
#         print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))

#         #distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
#         #distance = nd.norm(nd.concat(*param_list, dim=1), axis=0).asnumpy()
#         # normalize distance
#         distance = distance / np.sum(distance)
#     else:
#         distance = None

#     # sort
#     sorted_array = nd.array(np.sort(nd.concat(*param_list, dim=1).asnumpy(), axis=-1), ctx=mx.gpu(5))
#     #sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
#     # trim
#     n = len(param_list)
#     m = n - b * 2
#     trim_nd = nd.mean(sorted_array[:, b:(b + m)], axis=-1, keepdims=1)

#     # update global model
#     idx = 0
#     for j, (param) in enumerate(net.collect_params().values()):
#         if param.grad_req == 'null':
#             continue
#         param.set_data(param.data() - lr * trim_nd[idx:(idx + param.data().size)].reshape(param.data().shape))
#         idx += param.data().size

#     return trim_nd, distance


# def median(old_gradients, param_list, net, lr, b=0, hvp=None):
#     if hvp is not None:
#         pred_grad = []
#         distance = []
#         for i in range(len(old_gradients)):
#             pred_grad.append(old_gradients[i] + hvp)
#             #distance.append((1 - nd.dot(pred_grad[i].T, param_list[i]) / (
#                         #nd.norm(pred_grad[i]) * nd.norm(param_list[i]))).asnumpy().item())

#         pred = np.zeros(100)
#         pred[:b] = 1
#         distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
#         auc1 = roc_auc_score(pred, distance)
#         distance = nd.norm((nd.concat(*pred_grad, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
#         auc2 = roc_auc_score(pred, distance)
#         print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))

#         #distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
#         #distance = nd.norm(nd.concat(*param_list, dim=1), axis=0).asnumpy()

#         # normalize distance
#         distance = distance / np.sum(distance)
#     else:
#         distance = None

#     if len(param_list) % 2 == 1:
#         median_nd = nd.concat(*param_list, dim=1).sort(axis=-1)[:, len(param_list) // 2]
#     else:
#         median_nd = nd.concat(*param_list, dim=1).sort(axis=-1)[:, len(param_list) // 2: len(param_list) // 2 + 1].mean(axis=-1, keepdims=1)

#     idx = 0
#     for j, (param) in enumerate(net.collect_params().values()):
#         if param.grad_req == 'null':
#             continue
#         param.set_data(param.data() - lr * median_nd[idx:(idx + param.data().size)].reshape(param.data().shape))
#         idx += param.data().size
#     return median_nd, distance


port= 5680
# for i in range(8):
listenThread = SocketThread(buffer_size=1024*1024*1024,
                                            recv_timeout=10,port=5680)
listenThread.start()
print("Listening for incoming connection ...")
