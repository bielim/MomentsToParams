# Import modules
import numpy as np
import random
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from sklearn.model_selection import train_test_split
from numpy import genfromtxt # to read csv data file
import scipy.stats as stats 
import time
import copy
from utils import zscore

# PyTorch
import torch
torch.manual_seed(0) # for reproducibility
import torch.nn.functional as F
from torch.utils.data import Dataset


### *********************************************************************** ###
###                                                                         ### 
###              Feedforward Neural Network for Regression                  ###
###                  --- Single-output, One Layer ---                       ###
###                                                                         ### 
### *********************************************************************** ### 

### GOAL: Learn map from the first and second moment (M1, M2) of a Gamma 
###       distribution to the underlying parameters (alpha, theta) of the 
###       distribution.


### ------
### Settings 
### ------
test_fraction = 0.1   # fraction of total samples used for testing
valid_fraction = 0.2  # fraction of training data used for validation
N_epochs = 100        # number of epochs
batch_size = 512      # batch size
target_param = 'alpha' # parameter to predict ('alpha' or 'theta')
model_name = 'FeedForwardNet_2layer_single_output_' + target_param 


### ------
### Define model class
### ------
class FeedForwardNet(torch.nn.Module):
    def __init__(self, D_in, H1, D_out):
        # Note: super is shortcut to access a base class without having to know 
        # its type or name.
        super(FeedForwardNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, D_out)

        self.bn1 = torch.nn.BatchNorm1d(H1)

    def forward(self, x):
        h1 = F.leaky_relu(self.bn1(self.linear1(x)))
        y_pred = self.linear2(h1)
        return y_pred
    

### ------
### Load and prepare data
### ------
class Data(Dataset):
    """ moments-parameters data """

    def __init__(self, X, y):

        self.inputs = X
        self.targets = y
        self.len = np.shape(X)[0]
        
    def __getitem__(self, index):
        
        x = self.inputs[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.inputs)


# Read csv file containing data samples
# Columns are: M1, M2, alpha, theta
all_data = genfromtxt('../../data/training_data_gamma_M1_M2.csv', delimiter=',', 
                      skip_header=1)
np.random.shuffle(all_data)
trainx_all = all_data[:, :2] # M1, M2
if target_param == 'alpha':
    trainy_all = all_data[:, -2]
elif target_param == 'theta':
    trainy_all = all_data[:, -1]
else:
    raise ValueError('target_param has to be either alpha or theta')
X_temp, X_test, y_temp, y_test = train_test_split(trainx_all, trainy_all, 
                                                  test_size=test_fraction, 
                                                  random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp,
                                                      test_size=valid_fraction, 
                                                      random_state=42)
    
# array of length n_inputs, containing the mean of each feature
train_mean = np.mean(X_train, axis=0) 
# array of length n_inputs, containing the standard dev of each feature
train_std = np.std(X_train, axis=0)
train_data = Data(torch.FloatTensor(zscore(X_train, train_mean, train_std)), 
                  torch.FloatTensor(y_train).unsqueeze_(1))
valid_data = Data(torch.FloatTensor(zscore(X_valid, train_mean, train_std)), 
                  torch.FloatTensor(y_valid).unsqueeze_(1))
test_data = Data(torch.FloatTensor(zscore(X_test, train_mean, train_std)), 
                 torch.FloatTensor(y_test).unsqueeze_(1))

# Check for invalid data
# print(np.isnan(valid_data.x.data.numpy()).any())
# print(np.isinf(valid_data.x.data.numpy()).any())


### ------
### Training
### ------
def train_model(model, dset_loaders, dset_sizes, criterion, optimizer, 
                lr_scheduler, N_epochs=100, use_gpu=False):

    since = time.time()
    cost = {'train': [],
            'valid': []}

    best_model = model
    best_loss = np.Inf 

    for epoch in range(N_epochs):
        print('Epoch {}/{}'.format(epoch, N_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over batches
            for dataset in dset_loaders[phase]:
                # get the inputs and targets
                inputs, targets = dataset

                if use_gpu:
                    inputs = torch.tensor(inputs.cuda())
                    targets = torch.tensor(targets.cuda())
                else:
                    inputs = torch.FloatTensor(inputs)
                    targets = torch.FloatTensor(targets)

                # forward
                y_pred = model(inputs)
                loss = criterion(y_pred, targets)
  
                # backward + optimize only if in training phase
                if phase == 'train':
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.detach() 

            # Get epoch loss (per sample)
            epoch_loss = running_loss / dset_sizes[phase]
            cost[phase].append(epoch_loss)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = copy.deepcopy(model)
                
            if phase == 'train':
                lr_scheduler.step(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, 
                                                        time_elapsed % 60))

    return best_model, cost          
            

D_in = np.shape(train_data.inputs)[1] # number of features
D_out = 1 # 1 if only 1 parameter is predicted, 2 if both are predicted
H1 = 20 # number of nodes in hidden layer
model = FeedForwardNet(D_in, H1, D_out)

criterion = torch.nn.MSELoss(reduction='sum')

# Define an Optimizer and a learning rate scheduler
learning_rate = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                          factor=0.1, 
                                                          patience=5, 
                                                          verbose=True, 
                                                          threshold=0.0001)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
dset_loaders = {'train': train_loader,
                'valid': valid_loader}
dset_sizes = {'train': train_data.len,
              'valid': valid_data.len}

trained_model, cost = train_model(model, dset_loaders, dset_sizes, criterion, 
                                  optimizer, lr_scheduler, N_epochs=N_epochs)

# Save model
torch.save(trained_model.state_dict(), '../../output/' + model_name + '.pt')


### ------
### Evaluate performance
### ------
# Plot cost in training phase
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.plot(range(len(cost['train'])), cost['train'], lw=2.0, c='red', 
         label='training')
ax.plot(range(len(cost['valid'])), cost['valid'], lw=2.0, c='blue', 
         label='validation')
ax.set_xlabel('epoch')
ax.set_ylabel('cost')
plt.legend()
plt.show()
fig_name = model_name + '_train_and_valid_cost.png'
fig.savefig('../../plots/' + fig_name)

# Compute performance on the test set
model.eval()
y_pred_test = trained_model(test_data.inputs)
test_cost_sum = criterion(y_pred_test, test_data.targets)
test_cost = test_cost_sum / test_data.len
print('MSE on the test set: {0:2.4f}'.format(test_cost))
