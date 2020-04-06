# Import modules
import numpy as np
from numpy import genfromtxt
import math, random
import scipy.stats as stats 
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import time


### *********************************************************************** ###
###                                                                         ### 
###                       Random Forest Regression                          ###
###                          --- Multioutput ---                            ###
###                                                                         ###
### *********************************************************************** ### 

### GOAL: Learn map from the first and second moment (M1, M2) of a Gamma 
###       distribution to the underlying parameters (alpha, theta) of the 
###       distribution. This is a three-layer, multioutput implementation that 
###       simultaneous learns both alpha and theta.


### ------
### Settings 
### ------
test_fraction = 0.05        # fraction of total samples used for testing
max_depth = 15              # maximum depth of the trees
N_estimators = 50           # number of trees ("forest size")
model_name = 'RandomForest' # for file names etc.


### ------
### Load and prepare data
### ------
all_data = genfromtxt('../../data/training_data_gamma.csv', delimiter=',', 
                      skip_header=1)

trainx_all = all_data[:, :3] # M0, M1, M2
trainy_all = all_data[:, -2:] # -2 for α, -1 for θ, -2: for both
X_train, X_test, y_train, y_test = train_test_split(trainx_all, trainy_all, 
                                                    test_size=test_fraction, 
                                                    random_state=42, 
                                                    shuffle=True)

### ------
### Fit the model
### ------
start = time.time()
model = RandomForestRegressor(random_state=42, max_depth=max_depth, 
                              n_estimators=N_estimators)
model.fit(X_train, y_train)
end = time.time()
print("Training time: ")
print(end - start)


### ------
### Evaluate performance
### ------
y_pred = model.predict(X_test)
# calculate MSE on the test set
test_cost = mean_squared_error(y_pred, y_test)
print('MSE on the test set: {0:2.4f}'.format(test_cost))

# Compare the true to the predicted Gamma distributions
max_n_plots = 10
for i in range(min(np.shape(y_test)[0], max_n_plots)):
    xmax = 20
    ymax = 0.5
    x = np.linspace (0, xmax, 200) 
    pred = model.predict(np.expand_dims(X_test[i,:], axis=0))
    alpha_hat, theta_hat = pred[0][0], pred[0][1]
    yhat = stats.gamma.pdf(x, a=alpha_hat, loc=theta_hat) 
    alpha_true, theta_true = y_test[i, 0], y_test[i, 1]
    ytrue = stats.gamma.pdf(x, a=alpha_true, loc=theta_true)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    label_hat = 'alpha_hat={0:2.3f}, theta_hat={1:2.3f}'.format(alpha_hat, 
                                                                theta_hat)
    ax.plot(x, yhat, 'r--', lw=2.0, label=label_hat, zorder=20) 
    label_true = 'alpha_true={0:2.3f}, theta_true={1:2.3f}'.format(alpha_true, 
                                                                   theta_true)
    ax.plot(x, ytrue, 'b', lw=2.0, label=label_true) 
    plt.legend()
    ax.set_ylim([0, ymax])
    ax.set_xlim([0, xmax])
    plt.show()
    fig_name = str(i).zfill(3) + '_gamma_hat_vs_true_' + model_name + '.png'
    fig.savefig('../../plots/' + fig_name)
