## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Little library with a neural network class
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


## External libraries
## ~~~~~~~~~~~~~~~~~~

import numpy as np


## Activation Functions
## ~~~~~~~~~~~~~~~~~~~~

class ActivationFunc(object):
    
    def func():
        pass
    
    def derivative():
        pass
    
class Sigmoid(ActivationFunc):
    
    def func(z):
        return 1 / (1 + np.exp(-z))
    
    def derivative(z):
        return Sigmoid.func(z) * (1 - Sigmoid.func(z))
    
class Tanh(ActivationFunc):
    
    def func(z):
        return np.tanh(z)
    
    def derivative(z):
        return 1 - np.tanh(z)**2
    
class ReLU(ActivationFunc):
    
    def func(z):
        return z * (z >= 0)
    
    def derivative(z):
        return 1 * (z >= 0)

class Linear(ActivationFunc):
    
    def func(z):
        return z
    
    def derivative(z):
        return 1
    
class Softmax(ActivationFunc):
    
    def func(z):
        e = np.exp(z)
        return e / np.sum(e)
    
    def derivative(z):
        pass
    


## Cost functions
## ~~~~~~~~~~~~~~

class CostFunc(object):
    
    def func():
        ''' Network cost. '''
        pass
    
    def delta():
        ''' Output layer error. '''
        pass

class CrossEntropyCost(CostFunc):
    
    def func(a, y):
        ''' nan_to_num assures that nans resulting from a = 1, log(1 - a) = nan become a 0 instead. '''
        
        return -np.sum(np.nan_to_num(y*np.log(a) + (1 - y)*np.log(1 - a)))
    
    def delta(z, a, y, act_func):
        
        return np.where(a*(1-a) != 0, (a - y) / (a * (1 - a)) * act_func.derivative(z), 0)

class QuadraticCost(CostFunc):
    
    def func(a, y):
        
        return 0.5 * np.sum((a - y)**2)
    
    def delta(z, a, y, act_func):
        
        return (a - y) * act_func.derivative(z)
    
class LogLikelihoodCost(CostFunc):
    
    def func(a, y):
        
        return -np.log(a[y])
    
    def delta(z, a, y, act_func):
        
        return -1/a[y] * act_func.derivative(z)
    


## Neural Network
## ~~~~~~~~~~~~~~

class NeuralNet(object):
    
    @staticmethod
    def NormWeights(sizes):
        
        num_layers = len(sizes)
        
        W = [np.random.normal(0, 0.1/sizes[i], \
                (sizes[i], sizes[i-1])) \
                for i in range(num_layers - 1)]
        B = [np.random.normal(0, 0.1/sizes[i], \
                (1, sizes[i+1])) \
                for i in range(num_layers - 1)]
        
        return (W, B)
    
    @staticmethod
    def LargeWeights(sizes):
        
        num_layers = len(sizes)
        
        W = [np.random.normal(0, 0.1, (sizes[i], sizes[i+1])) \
                 for i in range(num_layers - 1)]
        B = [np.random.normal(0, 0.1, (1, sizes[i+1])) \
                 for i in range(num_layers - 1)]
        
        return (W, B)
    
    @staticmethod
    def XHWeights(sizes):
        ''' Xavier-He initialisation of weights. '''
        
        num_layers = len(sizes)
        
        W = [np.random.normal(0, np.sqrt(2 / (sizes[i] + sizes[i+1])), \
                (sizes[i], sizes[i+1])) \
                for i in range(num_layers - 1)]
        B = [np.random.normal(0, np.sqrt(2 / (sizes[i] + sizes[i+1])), \
                (1, sizes[i+1])) \
                for i in range(num_layers - 1)]
        
        return (W, B)
    
    def __init__(self, sizes, act_func = Sigmoid, cost_func = CrossEntropyCost, weights = 'XHWeights'):
        ''' Initialises weights and biases. No biases in input layer. 
            \param sizes: list of layer sizes, [in, ... hidden, ... out]. 
            '''
        
        self.num_layers = len(sizes)
        self.shape = sizes
        
        self.act = act_func
        self.cost = cost_func
        
        self.weight_func = getattr(self, weights)
        self.W, self.B = self.weight_func(self.shape)
    
    #@staticmethod
    def forward_prop(self, X):
        ''' Forward propagates X. '''
        
        num_samples = len(X)
        
        self.Z = [np.tile(self.B[0], (num_samples, 1)) + \
            np.matmul(X, self.W[0])]
        self.A = [self.act.func(self.Z[0])]
        
        for l in range(1, self.num_layers - 1):
            self.Z.append(np.tile(self.B[l], (num_samples, 1)) + \
                np.matmul(self.A[l-1], self.W[l]))
            self.A.append(self.act.func(self.Z[l]))
        
        return self.A[-1]
    
    #@staticmethod
    def backward_prop(self, X, y):
        ''' Backward propagates errors in network given X, y. '''
        
        batch_size = len(X)
        
        dZ = [self.cost.delta(self.Z[-1], self.A[-1], y, self.act)]
        dA = [dZ[-1] / self.act.func(self.Z[-1])]
        dW = [(1/batch_size) * np.matmul(self.A[-1-1].transpose(), dZ[-1])]
        dB = [(1/batch_size) * np.sum(dZ[-1], axis = 0)]
        
        for l in range(1, self.num_layers-1):
            
            dA = [np.matmul(dZ[-l], self.W[-l].transpose())]                          + dA
            dZ = [dA[-l-1] * self.act.derivative(self.Z[-l-1])]                       + dZ

            if l < self.num_layers - 2:
                dW = [(1/batch_size) * np.matmul(self.A[-l-2].transpose(), dZ[-l-1])] + dW
            elif l == self.num_layers - 2:
                dW = [(1/batch_size) * np.matmul(X.transpose(), dZ[-l-1])]            + dW

            dB = [(1/batch_size) * np.sum(dZ[-l-1], axis = 0)]                        + dB
        
        return (dW, dB)
    
    @staticmethod
    def get_batches(X, y, batch_size):
        
        assert len(X) == len(y)
        
        shuffled_indices = np.random.permutation(len(X))
        shuffled_X = X[shuffled_indices]
        shuffled_y = y[shuffled_indices]
        
        if len(X)%batch_size != 0:
            shuffled_X = shuffled_X[:-(len(X)%batch_size)]
            shuffled_y = shuffled_y[:-(len(y)%batch_size)]
        
        if len(X) == batch_size:
            return [(shuffled_X, shuffled_y)]
        else:
            return [(shuffled_X[i:i+batch_size], shuffled_y[i:i+batch_size]) \
                    for i in np.linspace(0, len(shuffled_X)-batch_size, \
                                         int(len(shuffled_X)/batch_size), dtype = int)]
    
    @staticmethod
    def L2(W):
        return W
    
    @staticmethod
    def L1(W):
        return np.sign(W)
    
    @staticmethod
    def no_reg(W):
        return 0
    
    def train(self, X, y, epochs, batch_size, eta, mu = None, lmbda = None, method = 'SGD', reg = 'L2', verbose = False):
        
        # Initialise parameters
        if batch_size == -1:
            batch_size = len(X)
        
        if reg == None:
            self.reg = self.no_reg
            lmbda = 0
        else:
            self.reg = getattr(self, reg)
            
        if method == 'SGD':
            mu = 0
        elif method == 'momentum':
            assert mu is not None, "Need to pass parameter mu if method == 'momentum'"
        
        vW = []
        vB = []
        for W, B in zip(self.W, self.B):
            vW.append(np.zeros(W.shape))
            vB.append(np.zeros(B.shape))
        
        
        n_samples = len(X)
        
        # Train network
        epoch = 0
        while epoch in range(epochs):
            
            # Get batches:
            batches = self.get_batches(X, y, batch_size)
            for batch in batches:
                
                batchX, batchy = batch
                
                # Forward and backward prop
                self.forward_prop(batchX)
                dW, dB = self.backward_prop(batchX, batchy)
                
                # Update weights
                vW = [mu*vw - eta*dw for vw, dw in zip(vW, dW)]
                vB = [mu*vb - eta*db for vb, db in zip(vB, dB)]
                
                self.W = [w - (lmbda*eta/n_samples)*self.reg(w) \
                          + vw for w, vw in zip(self.W, vW)]
                self.B = [b + vb for b, vb in zip(self.B, vB)]
                
                # Print if verbose
                if verbose:
                    if epoch % 1000 == 0:
                        loss = np.mean(np.square(self.A[-1] - batchy))

                        try:
                            last_loss
                        except NameError:
                            is_last_loss = False
                        else:
                            is_last_loss = True

                        if is_last_loss and last_loss < loss:
                            print("Train loss on last batch: ", loss, "  WARNING - Loss Increasing")
                        else:
                            print("Train loss on last batch: ", loss)

                        last_loss = loss
                
                epoch += 1
                if epoch not in range(epochs): break
                    
        return None # Something?
    
    def predict(self, X):
        
        output = self.forward_prop(X)
        
        if self.shape[-1] == 1:
            self.predictions = np.array(1 * (output > 0.5)).reshape(len(output), 1)
        else:
            self.predictions = np.argmax(output, axis = 1).reshape(-1, 1)
            
        return self.predictions
    
    def accuracy(self, y, X = None):
        try:
            self.predictions
        except NameError:
            assert X is not None, 'Missing arg X or predict first.'
            self.testaccuracy = np.mean(self.predict(X) == y)
        else:
            self.testaccuracy = np.mean(self.predictions == y)
        return self.testaccuracy
    
    def cost(self, y, X = None):
        try:
            self.predictions
        except NameError:
            if X == None:
                print('Missing arg X or predict first.')
                return None
            self.testcost = self.cost.func(self.predict(X), y)
            return self.testcost
        else:
            self.testcost = self.cost.func(self.predictions, y)
            return self.testcost
    
    def save(self, filename):
        pass # for now...
    
    
                
            

    
    
    
    
    
    
    
    
    
        