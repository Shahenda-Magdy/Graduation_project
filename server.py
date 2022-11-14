import flwr as fl
import sys
import numpy as np
import random
import string
import socket
from noknow.core import ZK, ZKSignature, ZKParameters, ZKData, ZKProof

queue=[]
clientsigs = ["0"]

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
#     s.send(b'Done')
    s.bind((SERVER_IP, SERVER_PORT))
    s.listen(1)
    conn,addr = s.accept()
    s.send(registeration())
    data = s.recv(1024)
    print(data)
#     s.send(b'Done')
with socket.socket(socket.AF_INET , socket.SOCK_STREAM) as s:
    s.bind((SERVER_IP, SERVER_PORT))
    s.listen(1)
    conn,addr = s.accept()
    print(f'registering :{addr}')
    with conn:
        while(True):
            conn.send(b'send the secret please')
            data = conn.recv(1024)
            data=data.decode('utf-8')
            sig=data
            client= conn.recv()
            queue.append(client)
#             print(sig)
            break
def authenticate(sig):
    print(sig)
    for i in clientsigs:
        print(i)
        if i==sig:
            return True
        else:
            continue
    return False
if authenticate(sig):
    class SaveModelStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(
            self,
            rnd,
            results,
            failures
        ):
            aggregated_weights = super().aggregate_fit(rnd, results, failures)
            if aggregated_weights is not None:
                # Save aggregated_weights
                print(f"Saving round {rnd} aggregated_weights...")
                np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
            return aggregated_weights

    # Create strategy and run server
    strategy = SaveModelStrategy()

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
            server_address ="localhost:"+str(sys.argv[1]) , 
            grpc_max_message_length = 1024*1024*1024,
            strategy = strategy
    )
else:
    print("malicious user")