# MLP for Pima Indians Dataset with grid search via sklearn
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy

# https://www.kaggle.com/code/tobikaggle/keras-pima-indians-diabetes-optimizer-grid-search


# Function to create model, required for KerasClassifier
def create_model(optimizer='Nadam', init='uniform', activation='relu'):
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer=init, activation=activation))
    model.add(Dropout(0.2))

    model.add(Dense(8, kernel_initializer=init, activation=activation))
    model.add(Dropout(0.2))

    # final layer (binary)
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",", skiprows=1)

# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

# create model
model = KerasClassifier(model=create_model, verbose=0)

# grid search epochs, batch size and optimizer ('TFOptimizer' throws error)
optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

# possible activations 'softmax','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear'
activations = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

# 'glorot_uniform', 'normal', 'uniform'
# init = ['uniform','lecun_uniform','normal','identity','orthogonal','zero','one','glorot_normal','glorot_uniform', 'he_normal', 'he_uniform']
# init = ['uniform','lecun_uniform','normal','orthogonal','zero','one','glorot_normal','glorot_uniform', 'he_normal', 'he_uniform']
init = ['normal']

# just one epoch to save computational time (300 is optimum for this 12,8,1 model)
epochs = [2]

# batches can be 1,3,5,10,20,40
batches = [5]

param_grid = dict(epochs=epochs, batch_size=batches, model__init=init, model__activation=activations, model__optimizer=optimizers,)
grid = GridSearchCV(cv=3, estimator=model, param_grid=param_grid, n_jobs=-1, verbose=3)
# print(model.get_params().keys())
# dict_keys(['model', 'build_fn', 'warm_start', 'random_state', 'optimizer', 'loss', 'metrics', 'batch_size',
# 'validation_batch_size', 'verbose', 'callbacks', 'validation_split', 'shuffle', 'run_eagerly', 'epochs', 'class_weight'])
grid_result = grid.fit(X, Y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
