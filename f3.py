import numpy
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils 
from keras import backend as K 
import threading
import pickle
import socket
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
# K.set_image_dim_ordering('tf')
from keras.datasets import cifar10
# let's load data 
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32') 
X_train = X_train / 255.0 
X_test = X_test / 255.0
y_train = np_utils.to_categorical(y_train) 
y_test = np_utils.to_categorical(y_test) 
num_classes = y_test.shape[1]
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32,32,3), activation='relu', padding='same')) 
model.add(Dropout(0.2)) 
model.add(Conv2D(32, (3, 3), activation='relu', padding='same')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Conv2D(64, (3, 3), activation='relu', padding='same')) 
model.add(Dropout(0.2)) 
model.add(Conv2D(64, (3, 3), activation='relu', padding='same')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Conv2D(128, (3, 3), activation='relu', padding='same')) 
model.add(Dropout(0.2)) 
model.add(Conv2D(128, (3, 3), activation='relu', padding='same')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Flatten()) 
model.add(Dropout(0.2)) 
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3))) 
model.add(Dropout(0.2)) 
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3))) 
model.add(Dropout(0.2)) 
model.add(Dense(num_classes, activation='softmax'))
lrate = 0.01 
decay = lrate/100 
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False) 
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32) 
# Final evaluation of the model 
scores = model.evaluate(X_test, y_test, verbose=0) 
print("Accuracy: %.2f%%" % (scores[1]*100))



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
