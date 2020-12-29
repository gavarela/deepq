## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Little Deep Network Class/Library
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## External Libraries
## ~~~~~~~~~~~~~~~~~~

import numpy as np, json, sys
from scipy import signal


## ~~~~~~~~~~~~~~~~~
## Pooling functions
## ~~~~~~~~~~~~~~~~~


## Base class
## ~~~~~~~~~~

class PoolingFunc(object):
    ''' Will assume shape of object to be pooled is:
        (n_samples, n_filter/sample, n_rows, n_cols)
        '''
    
    @staticmethod
    def pool(X, kernel_shape):
        ''' Pool. '''
        pass
    
    @staticmethod
    def depool(Z, kernel_shape, ri, ci):
        ''' Inverse pool.
            Need ri, ci for depooling max and RMS pool.
            '''
        pass


## ~~~~~~~~~~~
#  Max Pooling   
## ~~~~~~~~~~~

class MaxPooling(PoolingFunc):
    
    @staticmethod
    def pool(X, kernel_shape):
        
        assert X.shape[-2] % kernel_shape[0] == 0 and \
               X.shape[-1] % kernel_shape[1] == 0, \
               "X dimensions need to be divisible by kernel dimensions (kernel needs to cover X with strides of kernel_width/kernel_height)."
        
        kr, kc = kernel_shape
        
        # Reshape X:
        # Axis 0: Row of box to be pooled
        # Axis 1: Row within box to be pooled 
        # Axis 2: Column of boc to be pooled
        # Axis 3: Column within boc to be pooled
        
        I, J, nr, nc = X.shape
        new_Xshape = (I, J, nr//kr, kr, nc//kc, kc)
        X = X.reshape(new_Xshape)
        
        # Indices of row and column maxima
        ci = X.argmax(5)
        ri = X.max(5).argmax(3)
        
        return X.max(5).max(3), ri, ci
    
    @staticmethod
    def depool(pooledX, kernel_shape, ri, ci):
        ''' Currently only works for ri, ci (and this pooledX) or shape (num_imgs, num_rows, num_cols). Fix?
            Also, figure out a way to do this without loops... '''
        
        # Initialise X
        kr, kc = kernel_shape
        I, J, pr, pc = pooledX.shape
        
        X_shape = list(pooledX.shape)
        X_shape = (I, J, pr*kr, pc*kc)
        
        temp_Xshape = (I, J, pr, kr, pc, kc)
        X = np.zeros(temp_Xshape)
        
        # Fill X
        for i in range(I):
            for j in range(J):
                for r in range(pr):
                    for c in range(pc):
                        X[i, j, 
                          r, ri[i, j, r, c], 
                          c, ci[i, j, r, ri[i, j, r, c], c]] = \
                            pooledX[i, j, r, c]
        
        return X.reshape(X_shape)


## ~~~~~~~~~~~~~~~
#  Average Pooling   
## ~~~~~~~~~~~~~~~

class AveragePooling(PoolingFunc):
    
    @staticmethod
    def pool(X, kernel_shape):
        
        assert X.shape[-2] % kernel_shape[0] == 0 and \
               X.shape[-1] % kernel_shape[1] == 0, \
               "X dimensions need to be divisible by kernel dimensions (kernel needs to cover X with strides of kernel_width/kernel_height)."
        
        kr, kc = kernel_shape
        I, J, nr, nc = X.shape
        
        # Reshape X:
        # Axis 0: Row of box to be pooled
        # Axis 1: Row within box to be pooled 
        # Axis 2: Column of boc to be pooled
        # Axis 3: Column within boc to be pooled
        
        I, J, nr, nc = X.shape
        new_Xshape = (I, J, nr//kr, kr, nc//kc, kc)
        X = X.reshape(new_Xshape)
        
        return (1/(kr*kc)) * X.sum(5).sum(3), \
                    None, None
    
    @staticmethod
    def depool(pooledX, kernel_shape, ri = None, ci = None):
        
        assert ri == None and ci == None, \
               "AveragePooling does not need ri and ci as inputs."
        
        # Initialise depooled X
        kr, kc = kernel_shape
        I, J, pr, pc = pooledX.shape
        
        X_shape = (I, J, pr*kr, pc*kc)
        X = np.zeros(X_shape)
        
        # Fill it out
        for i in range(I):
            for j in range(J):
                for r in range(pr):
                    for c in range(pc):
                        
                        el = (1/(kr*kc)) * pooledX[i, j, r, c]
                        
                        X[i, j, kr*r    , kc*c    ] = el
                        X[i, j, kr*r    , kc*c + 1] = el
                        X[i, j, kr*r + 1, kc*c    ] = el
                        X[i, j, kr*r + 1, kc*c + 1] = el
        
        return X


## ~~~~~~~~~~~
#  RMS Pooling   
## ~~~~~~~~~~~

class RMSPooling(PoolingFunc):
    ''' Root-Mean-Square Pooling. Not tested. '''
    
    @staticmethod
    def pool(X, kernel_shape):
        
        pooledX, _, _ = AveragePooling.pool(X**2, 
                                            kernel_shape)
        pooledX = np.sqrt(pooledX)
        
        return pooledX, X, None
    
    @staticmethod
    def depool(pooledX, kernel_shape, ri, ci = None):
        ''' ri here is actually the X that went into the pooling. Need the signs for it to depool. '''
        
        assert ci == None, \
               "RMSPooling does not need ci as an input."
        
        # Get signs of X
        signX = np.sign(ri)
        
        # Initialise depooled X
        depooledX = AveragePooling.pool(pooledX**2, 
                                        kernel_shape)
        depooledX = np.sqrt(depooledX)
        
        # Multiply by signX or not?
        return signX * depooledX




## ~~~~~~~~~~~~~~~~
#  Layers (Willies)
## ~~~~~~~~~~~~~~~~


## Base Willy class
## ~~~~~~~~~~~~~~~~

class Willy(object):
    
    def __init__(self):
        
        self.is_set_up = False
    
    @staticmethod
    def init_weights(n_in, n_out):
        ''' Xavier-He initialisation of weights. '''
        
        W = np.random.normal(0, (2 / (n_in + n_out))**0.5, \
                (n_in, n_out))
        B = np.random.normal(0, 0.01, (1, n_out))
        
        W = W.astype(np.float32)
        B = B.astype(np.float32)
        
        return W, B
    
    def set_act(self, act_func):
        ''' Sets activation function and its derivative. '''
        
        if act_func == 'relu':
            self.act  = lambda z: z * (z > 0)
            self.dact = lambda z: 1 * (z > 0)
        
        elif act_func == 'sigmoid':
            self.act  = lambda z: 1 / (1 + np.exp(-z))
            self.dact = lambda z: self.act(z) * (1 - self.act(z))
            
        elif act_func == 'linear':
            self.act  = lambda z: z
            self.dact = lambda z: 1
        
        else:
            raise Exception('Unsupported activation: ' + act_func)
    
    def reset_momenta(self):
        ''' Set momenta (for learning) to 0. '''
        
        self.vW = np.zeros(self.W.shape)
        self.vB = np.zeros(self.B.shape)
        
    def update_weights(self, dA, X,
                       learn_rate, mom_rate, reg_rate):
        ''' Copies dA to self.dZ. '''
        
        self.dZ = dA
    
    def copy(self):
        ''' Returns copy of this willy. '''
        
        return self.__class__.load(self.save())


## Fully Connected Layer
## ~~~~~~~~~~~~~~~~~~~~~


class ConnectedWilly(Willy):
    
    ''' Fully connected (dense) layer. '''
    
    def __init__(self, n_out, 
                 act_func = 'relu', n_in = None):
        ''' Initialise weights. '''
        
        self.n_out = n_out
        self.out_shape = n_out
        
        self.act_func = act_func
        self.set_act(act_func)
        
        self.is_set_up = False
        if n_in is not None:
            self.set_up(n_in)
    
    def set_up(self, in_shape):
        ''' In shape must be an integer. '''
        
        if type(in_shape) != int:
            raise Exception('Invalid input shape into Connected Willy.')
        
        self.n_in = in_shape
        self.W, self.B = self.init_weights(self.n_in, self.n_out)
        self.reset_momenta()
        
        self.is_set_up = True
    
    def forward_prop(self, X):
        ''' Applies dropouts and forward propagates by multiplying weights and adding bias. '''
        
        # Dropout
        n_samples = X.shape[0]
        
        # Forward prop
        self.Z = np.tile(self.B, (n_samples, 1)) + (X @ self.W)
        self.A = self.act(self.Z)
        
        return self.A.copy()
    
    def update_weights(self, dA, X,
                       learn_rate, mom_rate, reg_rate):
        ''' Given this layer's dC/da, where a is act(z), gets errors dC/dz and then uses them to update this layer's weights with L2 regularisation and SGD with momentum. '''
        
        # Get errors
        self.dZ = dA * self.dact(self.Z)
        
        dW = X.transpose() @ self.dZ
        dB = np.sum(self.dZ, axis = 0)
        
        # Update weights
        self.vB = mom_rate * self.vB + (1 - mom_rate) * dB
        self.vW = mom_rate * self.vW + (1 - mom_rate) * dW
        
        batch_size = self.dZ.shape[0]
        
        self.W = self.W * (1 - learn_rate * reg_rate / batch_size) - learn_rate * self.vW
        self.B = self.B - learn_rate * self.vB
    
    def backward_prop(self):
        ''' Backward propagates by multiplying errors by (transpose) weights. '''
        
        return self.dZ @ self.W.transpose()
    
    def save(self):
        ''' Returns parameters necessary to load/save willy to file as a dictionary. '''
        
        data = {'willy': 'connected',
                'n_in': self.n_in,
                'n_out': self.n_out,
                'activation': self.act_func,
                'weights': self.W.tolist(),
                'biases': self.B.tolist()}
        
        return data
    
    @classmethod
    def load(cls, data):
        ''' Given data, loads willy. '''
        
        willy = cls(n_in = data['n_in'],
                    n_out = data['n_out'],
                    act_func = data['activation'])
        
        willy.W = np.array(data['weights'].copy()).reshape((self.n_in, self.n_out)).astype(np.float32)
        willy.B = np.array(data['biases'].copy()).reshape((1, self.n_out)).astype(np.float32)
        
        return willy
    
    def copy(self):
        ''' Makes copy of this willy. '''
        
        copy = ConnectedWilly(n_in = self.n_in,
                              n_out = self.n_out,
                              act_func = self.act_func)
        
        copy.W = np.copy(self.W)
        copy.B = np.copy(self.B)
        
        return copy


## Dropout Willy
## ~~~~~~~~~~~~~

class DropoutWilly(Willy):
    ''' The realest Will. '''
    
    def __init__(self, p_dropout):
        
        self.p_dropout = p_dropout
        self.is_set_up = False
    
    def set_up(self, in_shape):
        ''' In shape can be anything. '''
        
        self.out_shape = in_shape
        self.is_set_up = True
        
    def reset_momenta(self):
        pass
    
    def forward_prop(self, X):
        ''' Applies dropouts to inputs. '''
        
        self.dropouts = np.random.binomial(1, 1 - self.p_dropout, X.shape).astype(np.float32)
        
        self.A = X * self.dropouts        
        return self.A.copy()
    
    def backward_prop(self):
        ''' Applies dropouts to errors - deactivate neuron. '''
        
        return self.dZ * self.dropouts
    
    def save(self):
        ''' Returns parameters necessary to load/save willy to file as a dictionary. '''
        
        if isinstance(self.out_shape, int):
            out_shape = [self.out_shape]
        else:
            out_shape = list(self.out_shape)
        
        data = {'willy': 'dropout',
                'shape': out_shape,
                'p_dropout': self.p_dropout}
        
        return data
    
    @classmethod
    def load(cls, data):
        ''' Creates willy from data. '''
        
        willy = cls(data['p_dropout'])
        willy.set_up(tuple(data['shape']))
        
        return willy
        


## Convolutional Willy
## ~~~~~~~~~~~~~~~~~~~

def makecols(A, filter_shape, stride):
    ''' \A of shape (N, C, nr, nc): N examples, each with C channels of nr x nc matrices,
        \filter_shape is 2D (fr, fc)
        \stride is an integer
        Within examples of A, takes flattened boxes of shape \filter_shape from each channel, appending corresponding boxes from different channels. Returns matrix where each row corresponds to an example-box combo (wow, confusing, but whatevs). 
        '''
    
    N, C, nr, nc = A.shape
    fr, fc = filter_shape
    
    # Shape of convolved examples
    ir = nr - (fr - 1)      
    ic = nr - (fr - 1)
    
    # Indices for start of each conv
    conv_ind = np.arange(C*nr*nc).reshape((C, nr, nc))[:, :-(fr-1), :-(fc-1)]
    
    # Offset from conv_start for rest of conv coverage
    offset = np.arange(fr)[:, None]*nc + np.arange(fc)
    
    # Final indices, put in place
    reshapen_conv_inds = conv_ind[:, :, ::stride].reshape((C, -1)).transpose().ravel()[:, None]
    final_inds = (reshapen_conv_inds + offset.ravel()).reshape((-1, C*fr*fc))
    fi_shape = final_inds.shape
    
    final_inds = final_inds.ravel()[None, :] + (np.arange(N)*C*nr*nc)[:, None]
    final_inds = final_inds.reshape((N, -1, C*fr*fc))
    return np.take(A, final_inds)

def fast_matmul(w, A, use_BLAS = False):
    ''' w shape (n, m), A shape (l, m, k).
        Performs l (n, m) x (m, k) matrix multiplications and returns an (l, n, k) matrix. '''
    
    a1, a2, a3 = A.shape
    w1, w2 = w.shape
    s1, s2, s3 = A.transpose((1, 0, 2)).shape
    
    A = A.transpose((1, 0, 2)).reshape((s1, s2*s3))
    wA = w @ A
    return wA.reshape((w1, a1, a3)).transpose((1, 0, 2))

def conv(A, w, stride = 1):
    ''' \A has shape (N, C, nr, nc): N examples, each with C channels of nr x nc matrices,
        \w has shape (F, C, fr, fc): F filters each with C channels of shape fr x fc,
        \stride is an integer.
        Convolves A and w with given stride using makecols. Final shape: (N, F, (nr-fr)//stride+1, (nc-fc)//stride+1) I think...
        ''' 
    
    N, C, nr, nc = A.shape
    F, C, fr, fc = w.shape
    
    matA = makecols(A, (fr, fc), stride)
    matAT = matA.transpose((0, 2, 1))
    
    matw = w.reshape((F, C*fr*fc))
    
    ans = fast_matmul(matw, matAT)
    ans_shape = ((N, F, nr-fr+1, (nc-fc)//stride + 1))
    return ans.reshape(ans_shape)

def pad(M, row_pad, col_pad):
    ''' Pads \M with \row_pad zeros all around the penulrimate dim and \col_pad zeros around the last dim. '''
    
    pad_width = [(0, 0)] * (len(M.shape)-2)
    pad_width += [(row_pad, row_pad), (col_pad, col_pad)]
    
    return np.pad(M, pad_width = tuple(pad_width), mode = 'constant', constant_values = 0)

def rot180(a):
    return np.rot90(np.rot90(a, axes = (-2, -1)), axes = (-2, -1))

class ConvolutionalWilly(Willy):
    
    @staticmethod
    def init_weights(in_shape, n_filters, filter_shape):
        ''' Initialises weights. Note: param \in_shape is (n_channels, X_dim, Y_dim). '''
        
        n_in = (in_shape[1]-filter_shape[0])*(in_shape[2]*filter_shape[1])
        n_out = n_in
        n_channels = in_shape[0]
        
        W_shape = (n_filters, n_channels, filter_shape[0], filter_shape[1])
        B_shape = (n_filters, 1, 1)
        
        W = np.random.normal(0, (2/(n_in + n_out))**0.5, W_shape)
        B = np.random.normal(0, 0.01, B_shape)
        
        W = W.astype(np.float32)
        B = B.astype(np.float32)
        
        return W, B
    
    def __init__(self, n_filters, filter_shape, stride = 1,
                 in_shape = None, act_func = 'relu'):
        ''' Note: param \in_shape is (n_channels, X_dim, Y_dim). '''
        
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.stride = stride
        
        self.act_func = act_func
        self.set_act(act_func)
        
        self.is_set_up = False
        if in_shape is not None:
            self.set_up(in_shape)
        
    def set_up(self, in_shape):
        ''' In shape must be 3D array. '''
        
        if len(in_shape) != 3:
            raise Exception('Invalid input shape to Convolutional Willy.')
        
        if (in_shape[1], in_shape[2]) == self.filter_shape:
            raise Exception('Cannot have input shape be the same as filter shape in Convolutional Willy.')
        
        self.in_shape = in_shape
        self.n_channels = in_shape[0]

        self.W, self.B = self.init_weights(in_shape, self.n_filters, self.filter_shape)
        self.reset_momenta()

        self.out_shape = (self.n_filters, 
                          in_shape[1] - self.filter_shape[0] + 1,
                          in_shape[2] - self.filter_shape[1] + 1)
        self.is_set_up = True
        
    def forward_prop(self, X):
        ''' Convolves X with all filters and applies the activation.
            Handles multiple channels by convolving each with a separate channel-filter and summing all channels within an example (before applying both the bias and the activation).
            Returns array of examples, each containing n_filters 2d convolutions.
            '''
        
        # Convolution
        Bs = np.array([self.B]*X.shape[0])
        self.Z = conv(X, self.W, self.stride) + Bs
        
        # Activation
        self.Z = np.array(self.Z)
        self.A = self.act(self.Z)
        
        return self.A.copy()
    
    def update_weights(self, dA, X,
                       learn_rate, mom_rate, reg_rate):
        ''' Given this layer's dC/da, updates weights via L2-regularised SDG with momentum. Calculates dC/dz and gets weight updates by convolving a rotation of inputs with the errors, dC/dz. '''
        
        # X : (N, C, nr, nc)
        # W : (F, C, fr, fc)
        # Z : (N, F, zr, zc), where zi = (ni-fi)//stride +1
        
        self.dZ = dA * self.dact(self.Z)
        
        # Get weight and bias updates
        rotX = rot180(X)
        rotXT = np.transpose(rotX, axes = (1, 0, 2, 3))
        dZT = np.transpose(self.dZ, axes = (1, 0, 2, 3))
        
        # rotX  : (N, C, nr, nc)
        # rotXT : (C, N, nr, nc)
        # dZT   : (F, N, zr, zc)
        
        dW = conv(rotXT, dZT, self.stride)
        
        # dW : (F, C, wr, wc), where wi = (zr-nr)//stride + 1
        
        dW = np.transpose(dW, axes = (1, 0, 2, 3))
        dB = self.dZ.sum(axis = 3).sum(axis = 2).sum(axis = 0).reshape(self.B.shape)
        
        # Update weights
        self.vB = mom_rate * self.vB + (1 - mom_rate) * dB
        self.vW = mom_rate * self.vW + (1 - mom_rate) * dW
        
        batch_size = self.dZ.shape[0]
        
        self.W = self.W * (1 - learn_rate * reg_rate / batch_size) - learn_rate * self.vW
        self.B = self.B - learn_rate * self.vB
    
    def backward_prop(self):
        ''' Backward propagates this layer's errors to previous layer's by convolving with rotation of weights. '''
        
        # X : (N, C, nr, nc)
        # W : (F, C, fr, fc)
        # Z : (N, F, zr, zc), where zi = (ni-fi)//stride +1
        
        # Rotate weights/filters
        rotW = rot180(self.W)
        rotWT = np.transpose(rotW, axes = (1, 0, 2, 3))
        
        padZ = pad(self.Z, self.filter_shape[0]-1, self.filter_shape[1]-1)
        
        # rotWT : (C, F, nr, nc)
        # padZ  : (N, F, zr+fr-1, zc+fc-1)
        
        # Backpropagate
        # prev_dZ = (N, C, pr, pc), where pi=(zi-1)//stride+1
        return conv(padZ, rotWT, self.stride)
    
    def save(self):
        ''' Returns data needed to load/save this willy to/from file. '''
        
        data = {'willy': 'convolutional',
                'n_filters': self.n_filters,
                'filter_shape': list(self.filter_shape),
                'stride': self.stride,
                'in_shape': self.in_shape,
                'activation': self.act_func,
                'weights': self.W.tolist(),
                'biases': self.B.tolist()}
        
        return data
    
    @classmethod
    def load(cls, data):
        ''' Makes willy from data. '''
        
        willy = cls(in_shape = data['in_shape'],
                    n_filters = data['n_filters'],
                    filter_shape = tuple(data['filter_shape']),
                    stride = data['stride'],
                    act_func = data['activation'])
        
        willy.W = np.array(data['weights'].copy()).reshape((self.n_filters, self.in_shape[1], self.filter_shape[0], self.filter_shape[1])).astype(np.float32)
        willy.B = np.array(data['biases'].copy()).reshape((self.n_filters, 1, 1)).astype(np.float32)
        
        return willy
    
    def copy(self):
        ''' Makes copy of this willy. '''
        
        copy = ConvolutionalWilly(in_shape = self.in_shape,
                                  n_filters = self.n_filters,
                                  filter_shape = self.filter_shape,
                                  act_func = self.act_func)
        
        copy.W = np.copy(self.W)
        copy.B = np.copy(self.B)
        
        return copy


## Pooling Willy
## ~~~~~~~~~~~~~

class PoolingWilly(Willy):
    
    ''' Note: Currently slow. Need to re-work this. '''
    
    def reset_momenta(self):
        pass
    
    def __init__(self, kernel_shape, 
                 pool_type = 'average', in_shape = None):
        
        self.kernel_shape = kernel_shape
        
        self.pool_type = pool_type
        if pool_type == 'max':
            self.pooling = MaxPooling
        
        elif pool_type == 'average':
            self.pooling = AveragePooling
            
        elif pool_type == 'rms':
            self.pooling = RMSPooling
        
        else:
            raise Exception('Unsupported pooling type: ' + pool_type)
        
        self.is_set_up = False
        if in_shape is not None:
            self.set_up(in_shape)
            
    def set_up(self, in_shape):
        ''' In shape must be 3D array. '''
        
        if len(in_shape) != 3:
            raise Exception('Invalid input shape to Pooling Willy.')
            
        if in_shape[1] % self.kernel_shape[0] != 0 or \
            in_shape[2] % self.kernel_shape[1] != 0:
            raise Exception('Incompatible input and kernel shapes in Pooling Willy.')
        
        self.in_shape = in_shape
        self.out_shape = (in_shape[0],
                          in_shape[1] / self.kernel_shape[0],
                          in_shape[2] / self.kernel_shape[1])
        
        self.is_set_up = True
        
    def forward_prop(self, X):
        ''' Pools input. '''
        
        self.Z, self.ri, self.ci = self.pooling.pool(X, self.kernel_shape)
        self.A = self.Z
        
        return self.A.copy()
        
    def backward_prop(self):
        ''' Depools errors. '''
        
        return self.pooling.depool(self.dZ, self.kernel_shape, self.ri, self.ci)
    
    def save(self):
        ''' Returns data needed to load/save willy from/to file. '''
        
        data = {'willy': 'pooling',
                'kernel_shape': list(self.kernel_shape),
                'in_shape': list(self.in_shape),
                'pool_type': self.pool_type}
        
        return data
    
    @classmethod
    def load(cls, data):
        ''' Recreates willy from data. '''
        
        willy = cls(kernel_shape = tuple(data['kernel_shape']),
                    in_shape = tuple(data['in_shape']),
                    pool_type = self.pool_type)
        
        return willy


## Stacking Willy
## ~~~~~~~~~~~~~~

class StackingWilly(Willy):
    
    def reset_momenta(self):
        pass
    
    def set_up(self, in_shape):
        ''' In must be 3D. '''
        
        if len(in_shape) != 3:
            raise Exception('Invalid input shape to Stacking Willy.')
        
        self.out_shape = int(np.prod(in_shape))
        self.is_set_up = True
    
    def forward_prop(self, X):
        ''' Flatten and store X shape. '''
        
        self.X_shape = X.shape
        n_samples = self.X_shape[0]
        
        self.A = X.reshape((n_samples, -1))
        
        return self.A.copy()
    
    def backward_prop(self):
        ''' Reshape back to X shape. '''
        
        return self.dZ.reshape(self.X_shape)
    
    def save(self):
        ''' Returns data needed to save/load willy to/from file. '''
        
        data = {'willy': 'stacking',
                'out_shape': self.out_shape}
        
        return data
    
    @classmethod
    def load(cls, data):
        ''' Recreates willy from data. '''
        
        willy = cls()
        willy.out_shape = data['out_shape']
        willy.is_set_up = True
        
        return willy


## ~~~~~~~~~~~~~~~~~~
#  Deep Willy Network
## ~~~~~~~~~~~~~~~~~~

class DeepWilly(object):
    
    _willy_classes = {'connected': ConnectedWilly,
                      'dropout': DropoutWilly,
                      'convolutional': ConvolutionalWilly,
                      'pooling': PoolingWilly,
                      'stacking': StackingWilly}
    
    def __init__(self, cost_func = 'cross entropy'):
        
        self.willies = []
        self.n_willies = 0
        
        self.cost_func = cost_func
        if cost_func == 'cross entropy': 
            self.cost  = lambda a, y: -np.mean(y * np.log(a + (a == 0)) + (1-y) * np.log(1-a + (a == 1)))
            self.dcost = lambda a, y: (1/a.shape[0]) * (a - y) / (a * (1 - a) + 1e-99)
        
        elif cost_func == 'quadratic':
            self.cost  = lambda a, y: 0.5 * np.mean((a - y)**2)
            self.dcost = lambda a, y: (1/a.shape[0]) * (a - y)
        
        else:
            raise Exception('Unsupported cost function: ' + cost_func)
    
    def add(self, willy):
        ''' Add a willy to the network and set it up based on the previous willy's output (if not the first one). '''
        
        if len(self.willies) == 0:
            if not willy.is_set_up:
                raise Exception('Input shape or number must be provided to first Willy.')
                
        else:
            willy.set_up(self.willies[-1].out_shape)
            
        self.willies.append(willy)
        self.n_willies += 1
        
    def forward_prop(self, X):
        ''' Forward propagates X through the willies. '''
        
        for willy in self.willies:
            X = willy.forward_prop(X)
        
        return X
    
    def backward_prop(self, X, y,
                      learn_rate, mom_rate, reg_rate):
        ''' Backward propagates errors while simultaneously updating weights in each layer. '''
        
        # Output layer dC/da
        batch_size = X.shape[0]
        dA = self.dcost(self.willies[-1].A, y)
        
        # Backpropagate, updating weights
        XA = [X] + [willy.A for willy in self.willies[:-1]]
        XA = XA[::-1]
        
        for w, willy in enumerate(reversed(self.willies)):
            willy.update_weights(dA, XA[w],
                                 learn_rate, mom_rate, reg_rate)
            dA = willy.backward_prop()
    
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
              num_iterations, batch_size, 
              learn_rate, reg_rate, mom_rate = 0,
              verbose = False):
        ''' Builds network, trains using given data and training parameters. '''
        
        # Change dtypes
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        # Initialise momenta to 0
        for willy in self.willies:
            willy.reset_momenta()
        
        # Train network
        for iteration in range(num_iterations):
            
            # Get batches:
            X_batches, y_batches = self.get_batches(X, y, batch_size)
            
            for batchX, batchy in zip(X_batches, y_batches):
                
                # Forward propagate
                self.forward_prop(batchX)

                # Backward propagate & update weights
                self.backward_prop(batchX, batchy,
                                   learn_rate, mom_rate, reg_rate)

            # Print progress
            if verbose:
                if iteration % verbose == 0:
                    print("Training cost on last batch: ", self.cost(self.willies[-1].A, batchy))
    
    def predict(self, X, pred_type = 'as is'):
        
        self.yhat = self.forward_prop(X)
        
        if pred_type == 'binary':
            self.yhat = 1 * (self.yhat > 0.5)
            
        elif pred_type == 'argmax':
            self.yhat = np.argmax(self.yhat, axis = 1).reshape(-1, 1)
        
        else:
            assert pred_type == 'as is', \
                "Provided argument pred_type (" + pred_type + ") not supported."
        
        return self.yhat
    
    def accuracy(self, X, y, pred_type = 'as is'):
        ''' Gets accuracy of predictions. '''

        return np.mean(self.predict(X, pred_type) == y)
    
    def save(self, filename):
        ''' Saves deep willy to file \filename. '''
        
        willy_data = []
        for willy in self.willies:
            willy_data.append(willy.save())
        
        data = {'cost': self.cost_func,
                'willies': willy_data}
        
        file = open(filename, "w")
        json.dump(data, file)
        file.close()
    
    @classmethod
    def load(cls, filename):
        ''' Loads deep willy from file \filename. '''
        
        file = open(filename, "r")
        data = json.load(file)
        file.close()
        
        deep_willy = cls(data['cost_func'])
        for willy in data['willies']:
            willy_class = DeepWilly._willy_classes[willy['willy']]
            deep_willy.add(willy_class.load(willy))
            
        return deep_willy
    
    def copy(self):
        ''' Replicates this deep willy using its attributes. '''
        
        copy = DeepWilly(cost_func = self.cost_func)
        
        for willy in self.willies:
            copy.willies.append(willy.copy())
            copy.n_willies += 1
        
        return copy

