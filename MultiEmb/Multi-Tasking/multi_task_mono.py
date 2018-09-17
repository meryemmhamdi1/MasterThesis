import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import data_util
from model_components import task_specific_attention, bidirectional_rnn

tf.reset_default_graph()


class MultiTaskEmbeddingsHANClassifier:
    """ Implementation of Multi-Tasking of the training of alignment of multilingual embeddings
    and crosslingual document classification model described in `Hierarchical Attention Networks for Document Classification (Yang et al., 2016)`
    (https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)"""

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 classes,
                 word_cell,
                 sentence_cell,
                 word_output_size,
                 sentence_output_size,
                 max_grad_norm,
                 dropout_keep_proba,
                 is_training=None,
                 learning_rate=1e-2,
                 device="/job:localhost/replica:0/task:0/device:GPU:0",
                 # '/job:localhost/replica:0/task:0/device:GPU:1',
                 scope=None):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.classes = classes
        self.word_cell = word_cell
        self.word_output_size = word_output_size
        self.sentence_cell = sentence_cell
        self.sentence_output_size = sentence_output_size
        self.max_grad_norm = max_grad_norm
        self.dropout_keep_proba = dropout_keep_proba

        if is_training is not None:
            self.is_training = is_training
        else:
            self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        """ TASK 2: Cross-lingual Document Classification Variables """
        # [document x sentence x word] X
        self.inputs_doc = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='inputs_doc')

        # [document x sentence] X
        self.word_lengths_doc = tf.placeholder(shape=(None, None), dtype=tf.int32, name='word_lengths_doc')

        # [document] X
        self.sentence_lengths_doc = tf.placeholder(shape=(None,), dtype=tf.int32, name='sentence_lengths_doc')

        # [document] Y, weight
        self.labels = tf.placeholder(shape=(None,), dtype=tf.int32, name='labels')
        self.sample_weights = tf.placeholder(shape=(None,), dtype=tf.float32, name='sample_weights')

        # [document x sentence x word] Shapes

        (self.document_size, self.sentence_size, self.word_size) = tf.unstack(tf.shape(self.inputs_doc))

        with tf.device("/device:CPU:0"):
            self._init_embedding(scope)  # embeddings cannot be placed on GPU

        with tf.device('/device:GPU:0'): #"/device:GPU:0"
            self._init_task2(scope)  # TASK 2 Variables Initialization

        """ Training and evaluation Stuff """

        with tf.device("/device:GPU:1"):
            with tf.variable_scope(scope or 'tcm') as scope:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
            with tf.variable_scope('train'):

                ## Task 2
                self.cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits,
                                                                        weights=self.sample_weights)
                #self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)

                self.loss = tf.reduce_mean(tf.multiply(self.cross_entropy, self.sample_weights))
                tf.summary.scalar('loss2', self.loss)

                self.accuracy2 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits, self.labels, 1), tf.float32))
                tf.summary.scalar('accuracy2', self.accuracy2)

                tvars = tf.trainable_variables()

                grads, global_norm = tf.clip_by_global_norm(
                    tf.gradients(self.loss, tvars),
                    self.max_grad_norm)

                tf.summary.scalar('global_grad_norm', global_norm)

                optimizer = tf.train.AdamOptimizer(learning_rate)
                gradients, variables = zip(*optimizer.compute_gradients(self.loss))
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.train_task2_op = optimizer.apply_gradients(zip(gradients, variables))

    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def _init_embedding(self, scope):
        with tf.variable_scope("embedding") as scope:
            """
            self.embedding_matrix = tf.get_variable(
                name="embedding_matrix",
                shape=[self.vocab_size, self.embedding_size],
                initializer=tf.random_normal_initializer(),
                # layers.xavier_initializer(),  # --TODO: CHANGE INITIALIZER OF THE EMBEDDING LAYER
                dtype=tf.float32, trainable=False)
            """
            W = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.embedding_size]),
                            trainable=False, name="W")
            
            self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_size])
            self.embedding_matrix = W.assign(self.embedding_placeholder)
            self.inputs_doc_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.inputs_doc)

    def _init_task2(self, scope):
        """
            TASK 2: Cross-lingual Document Classification Variables
        :param scope:
        :return:
        """
        # Word level
        with tf.variable_scope("doc") as scope:
            word_level_inputs = tf.reshape(self.inputs_doc_embedded, [
                self.document_size * self.sentence_size,
                self.word_size,
                self.embedding_size
            ])
            word_level_lengths = tf.reshape(
                self.word_lengths_doc, [self.document_size * self.sentence_size])

        with tf.variable_scope('word') as scope:
            word_encoder_output, _ = bidirectional_rnn(
                self.word_cell, self.word_cell,
                word_level_inputs, word_level_lengths,
                scope=scope)

        with tf.variable_scope('attention', reuse=True) as scope:
            word_level_output = task_specific_attention(
                word_encoder_output,
                self.word_output_size,
                scope=scope)

        with tf.variable_scope('dropout'):
            word_level_output = layers.dropout(
                word_level_output, keep_prob=self.dropout_keep_proba,
                is_training=self.is_training,
            )

        # Sentence_level
        sentence_inputs = tf.reshape(
            word_level_output, [self.document_size, self.sentence_size, self.word_output_size])

        with tf.variable_scope('sentence') as scope:
            sentence_encoder_output, _ = bidirectional_rnn(
                self.sentence_cell, self.sentence_cell, sentence_inputs, self.sentence_lengths_doc, scope=scope)

        with tf.variable_scope('attention_sent') as scope:
            sentence_level_output = task_specific_attention(
                sentence_encoder_output, self.sentence_output_size, scope=scope)

        with tf.variable_scope('dropout'):
            sentence_level_output = layers.dropout(
                sentence_level_output, keep_prob=self.dropout_keep_proba,
                is_training=self.is_training,
            )

        with tf.variable_scope('classifier', reuse=tf.AUTO_REUSE):
            self.logits = layers.fully_connected(
                sentence_level_output, self.classes, activation_fn=None)

            self.prediction = tf.argmax(self.logits, axis=-1)

