import tensorflow as tf

# Arbitrarity, we'll use placeholders and allow batch size to vary,
# but fix vector dimensions.
# You can change this as you see fit
a = tf.placeholder(tf.float32, shape=(None, 2))
b = tf.placeholder(tf.float32, shape=(None, 2))
#a = tf.nn.l2_normalize(a, 1)
#b = tf.nn.l2_normalize(b, 1)
c = a / tf.reduce_sum(tf.square(a), 1, keep_dims=True)
d = tf.reduce_sum(tf.multiply(c, c))
#a = tf.square(a)
#c = tf.reduce_sum( tf.multiply( a, b ), 1, keep_dims=True )
#c = a
with tf.Session() as session:
    print( d.eval(
        feed_dict={ a: [[4,8],[1,6]], b: [[5,2],[9,4]] }
    ) )
