## Network
## ~~~~~~~

import numpy as np
import json, sys

# NN functions
class WillyNet(object):
    ''' Simple neural network based off of the one I made for Will. '''
    
    def __init__(self, shape, problem, hidden_act = 'relu', weights = 'XavierHe'):
        ''' Initialises weights and biases using Xavier-He initialisation. Prepares rest of network depending on problem (regression or classification), hidden activation and weight initialisation options. 
            h_act is hidden layer activation function, here ReLU;
            dh_act is its derivative;
            o_act is the output layer activation function, here linear;
            cost is the cost function.
            '''
        
        self.shape = shape
        self.n_layers = len(shape)
        
        # Weight initialisation
        if weights == 'XavierHe':
            
            self.W = [np.random.normal(0, np.sqrt(2 / (shape[i] + shape[i+1])), (shape[i], shape[i+1])) \
                 for i in range(self.n_layers - 1)]

            self.B = [np.random.normal(0, np.sqrt(2 / (shape[i] + shape[i+1])), (1, shape[i+1])) \
                 for i in range(self.n_layers - 1)]
        
        elif weights == 'He':
            
            self.W = [np.random.normal(0, np.sqrt(2/shape[i]), (shape[i], shape[i+1])) \
                 for i in range(self.n_layers - 1)]

            self.B = [np.random.normal(0, np.sqrt(2/shape[i]), (1, shape[i+1])) \
                 for i in range(self.n_layers - 1)]
        
        # Hidden layers
        if hidden_act == 'relu':
            self.h_act = lambda z: z * (z>0)
            self.dh_act = lambda z: 1 * (z>0)
        elif hidden_act == 'sigmoid':
            self.h_act = lambda z: 1 / (1 + np.exp(-z))
            self.dh_act = lambda z: self.h_act(z) * (1 - self.h_act(z))
        
        # Output layer and cost
        if problem == 'regression':
            self.o_act = lambda z: z
            self.cost = lambda a, y: 0.5 * np.mean((y - a)**2)
        elif problem == 'classification':
            self.o_act = lambda z: 1 / (1 + np.exp(-z))
            self.cost = -np.mean(np.nan_to_num(y*np.log(a)) + np.nan_to_num((1-y)*np.log(1-a)))
        
        #elif problem == 'softmax':
            #self.o_act = lambda z: np.exp(-z)/np.sum(np.exp(-z), axis = 1)
            #self.cost = lambda a, y: -np.sum(np.log(a*y), axis = 1)
            # NEED TO CHECK THE ABOVE BEFORE USING!!!
        
    def forward_prop(self, X):
        ''' Given X, performs a forward propagation. Calculates activation of each layer up to output layer. '''
        
        X = np.array(X)
        n_samples = X.shape[0]
        
        self.Z = [np.tile(self.B[0], (n_samples, 1)) + (X @ self.W[0])]
        self.A = [self.h_act(self.Z[0])]

        for l in range(1, self.n_layers-2):
            self.Z.append(np.tile(self.B[l], (n_samples, 1)) + (self.A[l-1] @ self.W[l]))
            self.A.append(self.h_act(self.Z[l]))
        
        self.Z.append(np.tile(self.B[-1], (n_samples, 1)) + (self.A[-1] @ self.W[-1]))
        self.A.append(self.o_act(self.Z[-1]))
        
        return np.array(self.A[-1])

    def backward_prop(self, X, y):
        ''' Backward propagates the errors given the batch of training labels, y. 
            dZ is dC/dZ and dW and dB are dC/dW and dC/dB. 
            Specific to cross-entropy cost and ReLU hidden activations and sigmoid output activations. '''
        
        n_samples = X.shape[0]
        
        XA = [X] + self.A
        
        # Output layer
        dZ = [(1/n_samples) * (XA[-1] - y)]
        
        dW = [XA[-1-1].transpose() @ dZ[-1]]
        dB = [np.sum(dZ[-1], axis = 0)]

        # Work backwards through other layers
        for l in range(1, self.n_layers-1):
            
            dZ = [(dZ[-l] @ self.W[-l].transpose()) * self.dh_act(self.Z[-l-1])] + dZ

            dW = [XA[-l-2].transpose() @ dZ[-l-1]] + dW
            dB = [np.sum(dZ[-l-1], axis = 0)] + dB
        
        return dW, dB
    
    @staticmethod
    def get_batches(X, y, batch_size):
        ''' Shuffles training data and gets random batch of desired size. '''
        
        if batch_size == -1 or batch_size >= X.shape[0]:
            return [X], [y]
        
        # Shuffle data
        shuffled_indices = np.random.permutation(len(X))
        shuffled_X = X[shuffled_indices]
        shuffled_y = y[shuffled_indices]

        # Get batch of desired size
        X_batches = []
        y_batches = []
        for i in range(X.shape[0]//batch_size):
            X_batches.append(shuffled_X[int(batch_size*i):int(batch_size*(i+1))])
            y_batches.append(shuffled_y[int(batch_size*i):int(batch_size*(i+1))])
        
        return X_batches, y_batches

    def train(self, X, y, 
              l_rate, batch_size, reg_rate, num_iterations, mom_rate = 0,
              verbose = False):
        ''' Builds network, trains using given data and training parameters. 
            Regularisation is L2. '''
        
        # Initialise momenta to 0
        vW = []
        vB = []
        for w, b in zip(self.W, self.B):
            vW.append(np.zeros(w.shape))
            vB.append(np.zeros(b.shape))
        
        # Train over given number of iterations
        for iteration in range(num_iterations):
            
            # Get batches
            X_batches, y_batches = self.get_batches(X, y, batch_size)
            
            for batchX, batchy in zip(X_batches, y_batches):
                
                # Forward propagate
                self.forward_prop(batchX)

                # Backward propagate
                dW, dB = self.backward_prop(batchX, batchy)

                # Update weights
                vW = [mom_rate * vw - l_rate * dw for vw, dw in zip(vW, dW)]
                vB = [mom_rate * vb - l_rate * db for vb, db in zip(vB, dB)]

                self.W = [w * (1 - reg_rate * l_rate / batch_size) + vw \
                          for w, vw in zip(self.W, vW)]
                self.B = [b + vb for b, vb in zip(self.B, vB)]

                # Print progress
                if verbose:
                    if iteration % verbose == 0:
                        print("Training cost: ", self.cost(self.A[-1], batchy))
        
    def predict(self, X, pred_type = 'as is'):
        ''' Predicts the output given input and weights. Returns the index of neuron in output layer with highest activation. '''

        X = np.array(X)
        self.forward_prop(X)
        
        if pred_type == 'as is':
            self.yhat = self.A[-1]
            
        elif pred_type == 'binary':
            self.yhat = 1*(self.A[-1] > 0.5)
            
        elif pred_type == 'argmax':
            self.yhat = np.argmax(self.A[-1], axis = 1).reshape(-1, 1)
        
        return self.yhat
    
    def accuracy(self, X, y, pred_type = 'as is'):
        ''' Gets accuracy of predictions. '''

        return np.mean(self.predict(X, pred_type) == y)
    
    def save(self, filename):
        data = {'shape': self.shape,
                'weights': [w.tolist() for w in self.W],
                'biases': [b.tolist() for b in self.B]}
        
        file = open(filename, "w")
        json.dump(data, file)
        file.close()
    
    @classmethod
    def load(cls, filename):
        
        file = open(filename, "r")
        data = json.load(file)
        file.close()
        
        network = cls(data['shape'])
        
        network.W = [np.array(w) for w in data['weights']]
        network.B = [np.array(b) for b in data['biases']]
        
        return network
        