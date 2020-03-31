import numpy as np
import torch
import torch.nn.init as init

### ------
### Utility functions
### ------

def zscore(X, meanX, stdX):
    """
    Compute the z scores of matrix X using the given mean and std. 
    Transformation is applied column-wise.

    Args:
        X: numpy array of size mxn
        meanX: numpy array of length n, containing the means to be applied when
               calculating the z score
        stdX: numpy array of length n, containing the standard deviations to be
              applied when calculating the z scores

    Returns:
        Z: numpy array of size mxn, resulting from substracting the mean and 
           dividing by the standard deviation (applied to each column)

    Example: 
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        means = np.mean(X, axis=0)
        stdevs = np.std(X, axis=0)
        zscore(X, means, stdevs) 
            --> returns array([[-1.22474487, -1.22474487],
                              [ 0.        ,  0.        ],
                              [ 1.22474487,  1.22474487]])

    """
    Z = np.zeros(np.shape(X))
    n_cols = np.shape(X)[1]
    # Add the machine epsilon to the standard dev in order to 
    # avoid division by zero
    eps = np.finfo(meanX.dtype).eps
    for i in range(n_cols):
        Z[:,i] = (X[:,i] - meanX[i]) / (stdX[i] + eps)
    return Z


def glorot_weight_zero_bias(model):
    """
    Initalize parameters of all modules
    by initializing weights with glorot  uniform/xavier initialization, and 
    setting biases to zero.
    Weights from batch norm layers are set to 1.

    Args:
        model: PyTorch model
    Returns: 
        No return value; initializes model weights
    """
    for module in model.modules():
        if hasattr(module, 'weight') and not module.weight is None:
            if not ('BatchNorm' in module.__class__.__name__):
                init.xavier_uniform_(module.weight, gain=1)
            else:
                init.constant_(module.weight, 1)
        if hasattr(module, 'bias'):
            if module.bias is not None:
                init.constant_(module.bias, 0)
