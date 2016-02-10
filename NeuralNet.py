import tensorflow as tf

import tensorflow.examples.tutorials.mnist.input_data as inputdata

# Load data using built in script. Makes it easy on us :)!
mnist_data = inputdata.read_data_sets('MNIST_data/', one_hot=True)

# Create placeholders for tensorflow so that it can build the computation graph
# x is the input data.
# In this case it is an n by 784 matrix. Where n is the size of our training data.
# 784 comes from the imade dataset. This is 28*28=784, which is the total number of pixels.

mnist_image_vector_size = 784

x = tf.placeholder(tf.float32, [None, mnist_image_vector_size])

# Create variables for our weight matrices and bias vectors.
# Variables are used by tensorflow to create the computation graph and then are modfied
# during trainig time.
# Variables must be initialized before use. In this case we use a normal distribution.
# Utilizing zeros as the initializing is a bad idea. In practice weights are usually initialized randomly in a
# variety of ways. There are many papers and discccsions onine about what method to use.
# Here we also set the number of neurons in the hidden layer. It's more of an art than a science, so
# we just use 625.
# We also set the number of classes. This is what we are trying to predict. There are 10 numbers
# 0,1,2,3,4,5,6,7,8, and 9 that we are trying to classify so we must tell tensorflow this.

n_neurons = 625
n_classes = 10

W = tf.Variable(tf.random_normal([mnist_image_vector_size, n_neurons]))
b = tf.Variable(tf.random_normal([n_neurons]))

W_2 = tf.Variable(tf.random_normal([n_neurons, n_classes]))
b_2 = tf.Variable(tf.random_normal([n_classes]))

# Create the operations for the neural network. That is multiplying the data matrix
# by our weight matrix, adding the bias vector, and finally applying a nonliterary, which in this case
# is a simple sigmoid function.

input_layer = tf.nn.sigmoid(tf.matmul(x, W) + b)
hidden_layer = tf.nn.sigmoid(tf.matmul(input_layer, W_2))

# Notice we do not apply a nonliterary to the final output layer. This is because we apply the softmax later on
# in the cross_entropy section.

y = tf.matmul(input_layer, W_2) + b_2
y_ = tf.placeholder(tf.float32, [None, n_classes])


# Here we define our cost function. Tensorflow will automatically calculate our gradients.
# All we have to do is tell it what cost function we want to use.
# We use cross entropy and use gradient descent to optimize the network.

learning_rate = .05

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
optim = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Go over the dataset n number of times. In this case it's 20.
# Define the total number of minibatches. In this case its 550. That is 55000/100.
n_epochs = 20
mini_batch_size = 100
total_batch = int(mnist_data.train.num_examples/mini_batch_size)

for epoch in range(n_epochs):
    avg_mini_batch_cost = 0

    for i in range(total_batch):
        batch_x, batch_y = mnist_data.train.next_batch(mini_batch_size)

        # Train the network on each minibatch and calcualte the cost for each minibatch.
        sess.run(optim, feed_dict={x: batch_x, y_: batch_y})
        avg_mini_batch_cost  += sess.run(cross_entropy, feed_dict={x: batch_x, y_:batch_y})/total_batch

    print("Epoch: {}, cost: {}".format(epoch + 1, avg_mini_batch_cost))


# Use equal() to check if out predictions are correct. arg_max() returns the class of the predicted y, since
# y is a one hot vector. That is represent the number 2 that we are trying to classify as a vector <0,0,1,0,0,0,0,0,0,0>

correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
print(sess.run(accuracy, feed_dict={x: mnist_data.test.images, y_: mnist_data.test.labels}))