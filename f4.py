import flwr as fl
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import pygad.kerasga
import sys
import numpy as np
import random
import string
import socket
import tensorflow as tf
import tensorflow
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import average
from numpy import array
import pickle
from noknow.core import ZK, ZKSignature, ZKParameters, ZKData, ZKProof
import pandas as pd
import threading
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense
from keras.optimizers import Adam
model = None

# Preparing the NumPy array of the inputs.
data_inputs = np.array([[1, 1],
                           [1, 0],
                           [0, 1],
                           [0, 0]])

# Preparing the NumPy array of the outputs.
data_outputs = np.array([[1, 0], 
                            [0, 1], 
                            [0, 1], 
                            [1, 0]])

num_classes = 2
num_inputs = 2

# Build the keras model using the functional API.
input_layer  = tensorflow.keras.layers.Input(num_inputs)
dense_layer = tensorflow.keras.layers.Dense(4, activation="relu")(input_layer)
output_layer = tensorflow.keras.layers.Dense(num_classes, activation="softmax")(dense_layer)

model = tensorflow.keras.Model(inputs=input_layer, outputs=output_layer)

num_solutions = 10
# Create an instance of the pygad.kerasga.KerasGA class to build the initial population.
keras_ga = pygad.kerasga.KerasGA(model=model,
                                 num_solutions=num_solutions)

class RecvThread(threading.Thread):

    def __init__(self, buffer_size, recv_timeout):
        threading.Thread.__init__(self)
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout

    def recv(self,soc):
            received_data = b""
            while True: # str(received_data)[-2] != '.':
                print("hhh")
                try:
                    soc.settimeout(self.recv_timeout)
                    received_data += soc.recv(self.buffer_size)
                    print(received_data)

                    try:
                        pickle.loads(received_data)
                        return received_data
    #                     self.kivy_app.label.text = "All data is received from the server."
                        print("All data is received from the server.")
                        # If the previous pickle.loads() statement is passed, this means all the data is received.
                        # Thus, no need to continue the loop and a break statement should be excuted.
                        break
                    except BaseException:
                        # An exception is expected when the data is not 100% received.
                        pass

                except socket.timeout:
                    print("A socket.timeout exception occurred because the server did not send any data for {recv_timeout} seconds.".format(recv_timeout=self.recv_timeout))
                    return
    #                 self.kivy_app.label.text = "{recv_timeout} Seconds of Inactivity. socket.timeout Exception Occurred".format(recv_timeout=self.recv_timeout)
    #             except BaseException as e:
    #                 print("Error While Receiving Data from the Server: {msg}.".format(msg=e))
    # #                 self.kivy_app.label.text = "Error While Receiving Data from the Server"

    #         try:
    #             received_data = pickle.loads(received_data)
    #         except BaseException as e:
    #             print("Error Decoding the Data: {msg}.\n".format(msg=e))
    # #             self.kivy_app.label.text = "Error Decoding the Client's Data"
    #             return 
        
    #         return received_data
    def run(self):
            global server_data
            
            soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            print("Socket is created.")
            soc.connect(("192.168.56.1" , 5680))

            while True:

                # data = {"subject": subject, "data": keras_ga, "best_solution_idx": best_sol_idx}
                data_byte = pickle.dumps(model)

    #             self.kivy_app.label.text = "Sending a Message of Type {subject} to the Server".format(subject=subject)
                try:
                    soc.sendall(data_byte)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
                except BaseException as e:
    #                 self.kivy_app.label.text = "Error Connecting to the Server. The server might has been closed."
                    print("Error Connecting to the Server: {msg}".format(msg=e))
                    break

    #             self.kivy_app.label.text = "Receiving Reply from the Server"
                received_data= self.recv(soc)
                if len(received_data)< 1:
                    print("nothing received from server")
                else:
                    break

                # subject = received_data["subject"]
                # if subject == "model":
                #     server_data = received_data["data"]
                # elif subject == "done":
                #     print( "Model is Trained")
                #     break
                # else:
                #     print( "Unrecognized Message Type: {subject}".format(subject=subject))
                #     break

                # ga_instance = prepare_GA(server_data)

                # ga_instance.run()

                # subject = "model"
                # best_sol_idx = ga_instance.best_solution()[2]
                # best_model_weights_vector = ga_instance.population[best_sol_idx, :]


recvThread = RecvThread(buffer_size=1024*1024*1024, recv_timeout=10)
recvThread.start()
