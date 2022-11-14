from client1 import model as p
class Worker:
    #secret_features must of the form {'f_name_1': 'f_value_1', 'f_name_2': 'f_value_2', ...}
    def __init__(self, id_w, zkp_client_prototype, server, x, y, secret_features, prototypes_dict, encrypt=True):
        self.id = id_w
        self.zkp_client_prototype = zkp_client_prototype
        self.server = server
        self.model = {}
        self.x = x
        self.y = y
        self.secret_features = secret_features
        self.encrypt = encrypt
        self.zkp_clients = {}
        for f in secret_features.keys():
            c = copy.copy(prototypes_dict[f][secret_features[f]])
            self.zkp_clients[f] = c
        
    def send_registration(self):
        self.server.register_worker(self)
        
    def prepare_features(self):
        feature_to_send = []
        for k in self.zkp_clients.keys():
            c = self.zkp_clients[k]
            secret = c.encrypt_label(self.encrypt)
            feature_to_send += [{c.feature: secret}]
        return feature_to_send
                    
    def load_model(self, model):
        self.model = model
        
    def send_update(self, features, weights):
        self.server.load_update(self.id, features, weights)
        
    def train(self):
        features = self.prepare_features()
        if self.server.balanced_update(features):
            #print('Authorized to train')
            self.model.train(self.x, self.y)
        weights = self.model.get_weights()
        self.send_update(features, weights)
class LearningModel:
    def __init__(self, model, epochs=1, learning_rate=0.025):
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
    
    def set_weights(self, W):
        self.model.load_weights(W)
        
    def get_weights(self):
        return self.model.get_weights()
    
    def train(self, X, y):
        self.model.fit(X, y, self.epochs, self.learning_rate)
    
    def predict(self, x):
        return self.model.predict(x)