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
import matplotlib.pyplot as plt
import pickle
from noknow.core import ZK, ZKSignature, ZKParameters, ZKData, ZKProof
import pandas as pd
def getDist(y):
    ax = sns.countplot(y)
    ax.set(title="Count of data classes")
    plt.show()


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
getDist(y_train)





# Load and compile Keras model
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(256, activation='relu'),
#     keras.layers.Dense(10, activation='softmax')
# ])

# model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# # Load dataset
# (x_train, y_train), (x_test1, y_test1) = tf.keras.datasets.mnist.load_data()
# x_train, x_test1 = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0
# dist = [10, 10, 10, 10, 10,10,10,10,10,10]
# x_train, y_train = getData(dist, x_train, y_train)
# getDist(y_train)
# print(model.evaluate(x_test1, y_test1, verbose=0))
clients=[]
clientsigs=[]

def registeration():
    password = random.randint(0,10)
    label = string.ascii_lowercase
    rand_string = ''.join(random.choice(label) for i in range(2))
    zk = ZK.new(curve_name="secp384r1", hash_alg="sha3_512")
    signature: ZKSignature = zk.create_signature(rand_string)
                                                            #         clientsig= ZK.sign(signature, rand_string)
    cL=str(signature)
    clientsigs.append(cL)
#     print(cL)
    encryptl=cL.encode('utf-8')
    return encryptl
SERVER_IP = '192.168.20.1'
SERVER_PORT = 5678
sig=''
with socket.socket(socket.AF_INET , socket.SOCK_STREAM) as s:
    x=0
    x+=1
    s.bind((SERVER_IP, SERVER_PORT))
    print('Server is listening')
    s.listen(x)
    conn,addr = s.accept()
    print(f'Connection accepted from :{addr}')
    with conn:
        while(True):
#             conn.send(b'Hello World')
#             data = conn.recv(1024)
#             data=data.decode('utf-8')
#             print(data)
            conn.send(registeration())
            client = conn.recv(1024)
            client=client.decode('utf-8')
            clients.append(client)
            break
with socket.socket(socket.AF_INET , socket.SOCK_STREAM) as s:
    x=0
    x+=1
    s.bind((SERVER_IP, SERVER_PORT))
    s.listen(x)
    conn,addr = s.accept()
    print(f'authentication:{addr}')
    with conn:
        while(True):
#             conn.send(b'send the secret please')
            data = conn.recv(1024)
            data=data.decode('utf-8')
            sig=data
#             print(sig)
            break
def authenticate(sig,i):
    if clientsigs[i]==sig:
        return True
    return False
for i in range(0,len(clients)):
    name=clients[i]
    print(name)
    if authenticate(sig,i):
#         for i in range(4):
        # Create strategy and run server
        soc = socket.socket()
        print("Socket is created.")

        soc.bind(("192.168.20.1" , 5678))
        print("Socket is bound to an address & port number.")

        soc.listen(1)
        print("Listening for incoming connection ...")

        connected = False
        accept_timeout = 10
        soc.settimeout(accept_timeout)
        try:
            connection, address = soc.accept()
            print("Connected to a client: {client_info}.".format(client_info=address))
            connected = True
        except socket.timeout:
            print("A socket.timeout exception occurred because the server did not receive any connection for {accept_timeout} seconds.".format(accept_timeout=accept_timeout))

        if connected:
            received_data = connection.recv(1024*1024*1024)
            model1= CombinedModel(globalmodel, received_data).predict
        #     received_data = pickle.loads(received_data)
            print("Received data from the client")


        #     msg = "Reply from the server."
        #     msg = pickle.loads(msg)
        #     connection.sendall(msg)
        #     print("Server sent a message to the client.")

            connection.close()
            print("Connection is closed with: {client_info}.".format(client_info=address))

        soc.close()
        print("Socket is closed.")

        class CombinedModel:
            def __init__(self, model1, model2):
                self.model1= model1
                self.model2= model2
                pd.concat[model1,model2].plot()
                plt.show()

#             def predict(self, X, **kwargs):
#                 ser_model1= X['x']==0.0
#                 print("midel is combined")
#                 return pd.concat([
#                         pd.Series(self.model1.predict(X[ser_model1]), index=X.index[ser_model1]),
#                         pd.Series(self.model2.predict(X[~ser_model1]), index=X.index[~ser_model1])
#                     ]
#                 ).sort_index()
#             plt.show()
        # create a model with the two trained sum models
        # and pickle it
    else:
        print("malicious user")
