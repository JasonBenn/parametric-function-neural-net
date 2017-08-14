import tensorflow as tf
import numpy as np

# Create placeholders
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Create function
f = tf.stack([tf.pow(x, 3), tf.square(x), x, tf.ones_like(x)], 1)

# Create randomly init'd weights and yhat
weights = tf.get_variable("w", shape=[4, 1])
yhat = tf.squeeze(tf.matmul(f, weights), 1) # unclear what squeeze does

# Create some training data
xs = np.random.uniform(-10, 10, size=500)
ys = -0.5 * np.power(xs, 3) + 2 * np.square(xs) + xs - 5

# Loss function
loss = tf.nn.l2_loss(yhat - y) + 0.1 * tf.nn.l2_loss(weights)

# Specify training algorithm
train = tf.train.AdamOptimizer(0.1).minimize(loss)

# Start tensorflow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())  # don't know what this does either

for i in xrange(1000):
  _, loss_val = sess.run([train, loss], { x: xs, y: ys })
  if i % 100 == 0:
    print "After epoch %i: %i" % (i, loss_val)
    print sess.run([weights])
