# Hello World in TensorFlow
# Run with: python hello-tensorflow.py
# See http://tensorflow.org
# https://github.com/tobigithub/tensorflow-deep-learning/wiki

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print sess.run(hello)

# END
