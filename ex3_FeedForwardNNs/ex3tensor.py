
#############################################################
#
#  Classic FeedForward NN 
#
#############################################################
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist

# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()

#############################################################
#  visulalization 
#############################################################
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

# plot first few images
for i in range(9):
	# define subplot
	plt.subplot(330 + 1 + i)
	# plot raw pixel data
	plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
# show the figure
plt.show()
#############################################################

model = keras.Sequential([
    #  flatten means (28,28) array becomes 784 pixels (neurons)
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    #  has 128 neurons
    keras.layers.Dense(128, activation='sigmoid'),  # hidden layer (2)
    # 1O neurons, each for a classification
    keras.layers.Dense(10, activation='sigmoid') # output layer (3)
])

# Optimizers are algorithms or methods used to change the attributes of the neural network 
#		such as weights and learning rate to reduce the losses. 
#		Optimizers are used to solve optimization problems by minimizing the function.

# loss: The purpose of loss functions is to compute the quantity that a model should seek 
#       to minimize during training.

# Metrics: A metric is a function that is used to judge the performance of your model.

model.compile(optimizer='SGD',
              loss='mean_squared_error',
              metrics=['accuracy'])

#  batch_size * num_of_iterations = 1 epoch
model.fit(trainX, trainy, epochs=10) 

# evaluating the model
# verbose = 1 will show you an animated progress bar:
test_loss, test_acc = model.evaluate(testX,  testy, verbose=1) 
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

predictions = model.predict(testX)

predictions[0]

print( np.argmax(predictions[0]))

# And we can check if this is correct by looking at the value of the cooresponding test label.
testy[0]
