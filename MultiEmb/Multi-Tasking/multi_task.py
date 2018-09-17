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

        """ TASK 1: Multilingual Embeddings Alignment Variables """
        self.inputs_src = tf.placeholder(shape=(None, None), dtype=tf.int32, name='inputs_src')
        (self.sentence_size_src, self.word_size_src) = tf.unstack(tf.shape(self.inputs_src))
        self.word_lengths_src = tf.placeholder(shape=(None, None), dtype=tf.int32, name='word_lengths_src')

        self.inputs_trg = tf.placeholder(shape=(None, None), dtype=tf.int32, name='inputs_trg')
        (self.sentence_size_trg, self.word_size_trg) = tf.unstack(tf.shape(self.inputs_trg))
        self.word_lengths_trg = tf.placeholder(shape=(None, None), dtype=tf.int32, name='word_lengths_trg')

        self.sentsim_labels = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='sentsim-outlabel')

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

        with tf.device("/device:GPU:0"):
            self._init_embedding(scope)  # embeddings cannot be placed on GPU

        #for d in ['/device:GPU:0', '/device:GPU:1']:
        #with tf.device("/device:GPU:0"):
            # with tf.device(device):
        #for d in ['/device:GPU:0', '/device:GPU:1']:
        with tf.device('/device:GPU:0'): #"/device:GPU:0"
            self._init_task1(scope)  # TASK 1 Variables Initialization
            #with tf.device("/device:GPU:1"):
            self._init_task2(scope)  # TASK 2 Variables Initialization
            # self._init_body(scope)

        """ Training and evaluation Stuff """

        # gradients_1 = []
        # variables_1 = []
        # gradients_2 = []
        # variables_2 = []
        # for d in ['/device:GPU:0', '/device:GPU:1']:
        #     with tf.device(d):
        #         with tf.variable_scope(scope or 'tcm') as scope:
        #             self.global_step = tf.Variable(0, name='global_step', trainable=False)
        #         with tf.variable_scope('train'):
        #             # Task 1 cosine, mismatch_loss, loss_match, loss, total_labels, loss_mean, loss_match_mean, loss_match_mean
        #             self.src_magnitudes, self.trg_magnitudes, self.cosine, self.mismatch_loss, self.loss_match, self.loss,\
        #             self.total_labels, self.cost_mean, self.cost_match_mean, self.cost_mismatch_mean = self.loss_src_trg(
        #                 self.word_level_output_src,
        #                 self.word_level_output_trg,
        #                 self.sentsim_labels)
        #
        #             self.accuracy1 = tf.maximum(0., 1 - self.cost_mean)
        #
        #             ## Task 2
        #             self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
        #
        #             self.loss = tf.reduce_mean(tf.multiply(self.cross_entropy, self.sample_weights))
        #             tf.summary.scalar('loss2', self.loss)
        #
        #             self.accuracy2 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits, self.labels, 1), tf.float32))
        #             tf.summary.scalar('accuracy2', self.accuracy2)
        #
        #             tvars = tf.trainable_variables()
        #
        #             grads, global_norm = tf.clip_by_global_norm(
        #                 tf.gradients(self.loss, tvars),
        #                 self.max_grad_norm)
        #
        #             tf.summary.scalar('global_grad_norm', global_norm)
        #
        #             grads_l2, global_norm_l2 = tf.clip_by_global_norm(
        #                 tf.gradients(self.cost_mean, tvars),
        #                 self.max_grad_norm)
        #
        #             tf.summary.scalar('global_grad_norm_l2', global_norm_l2)
        #
        #             optimizer = tf.train.AdamOptimizer(learning_rate)
        #             gradients_before, variables = zip(*optimizer.compute_gradients(self.cost_mean))
        #             gradients, _ = tf.clip_by_global_norm(gradients_before, 1.0)
        #             # grad_check = tf.check_numerics(gradients_before, "gradients is NAN")
        #             # with tf.control_dependencies([grad_check]):
        #             gradients_1.append(gradients)
        #             variables_1.append(variables)
        #
        #
        #             # params = tf.trainable_variables()
        #             # optimizer = tf.train.AdamOptimizer(learning_rate)
        #             # gradients = tf.gradients(self.loss_match, params)
        #             #
        #             # clipped_gradients, variables = tf.clip_by_global_norm(gradients, 1.0)
        #             # #grad_check = tf.check_numerics(clipped_gradients, "gradients is NAN")
        #             # #with tf.control_dependencies([grad_check]):
        #             # self.train_task1_op = optimizer.apply_gradients(zip(clipped_gradients, variables))
        #             #
        #
        #             gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        #             gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        #             gradients_2.append(gradients)
        #             variables_2.append(variables)
        #
        #
        #
        #             # gvs_task1 = optimizer.compute_gradients(self.loss_match)
        #             # capped_gvs_task1 = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs_task1]
        #             # self.train_task1_op = optimizer.apply_gradients(capped_gvs_task1)
        #             #
        #             # gvs_task2 = optimizer.compute_gradients(self.loss)
        #             # capped_gvs_task2 = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs_task2]
        #             # self.train_task2_op = optimizer.apply_gradients(capped_gvs_task2)
        #             #
        #             # self.class_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        #             # self.l2_opt = tf.train.GradientDescentOptimizer(0.000001).minimize(self.loss_match)  # (self.cost_mean)
        #             #
        #             #
        #             # self.train_class_opt = class_opt.apply_gradients(
        #             #     zip(grads, tvars), name='train_class_op',
        #             #     global_step=self.global_step)
        #             #
        #             # self.train_l2_opt = l2_opt.apply_gradients(
        #             #     zip(grads_l2, tvars), name='train_l2_op',
        #             #     global_step=self.global_step_l2)
        #             #
        #             # self.summary_op = tf.summary.merge_all()
        #
        #
        # #gradients_avg_1 = self.average_gradients(gradients_1)
        # #gradients_avg_2 = self.average_gradients(gradients_2)
        # #self.train_task1_op = optimizer.apply_gradients(zip(gradients_avg_1, variables))
        # #self.train_task2_op = optimizer.apply_gradients(zip(gradients_avg_2, variables))
        #
        # self.train_task1_op = tf.group(*[optimizer.apply_gradients(grad) for grad in gradients_1])
        #
        # self.train_task2_op = tf.group(*[optimizer.apply_gradients(grad) for grad in gradients_2])

        with tf.device("/device:GPU:1"):
            with tf.variable_scope(scope or 'tcm') as scope:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
            with tf.variable_scope('train'):
                # Task 1 cosine, mismatch_loss, loss_match, loss, total_labels, loss_mean, loss_match_mean, loss_match_mean
                self.src_magnitudes, self.trg_magnitudes, self.cosine, self.mismatch_loss, self.loss_match, self.loss, \
                self.total_labels, self.cost_mean, self.cost_match_mean, self.cost_mismatch_mean = self.loss_src_trg(
                    self.word_level_output_src,
                    self.word_level_output_trg,
                    self.sentsim_labels)

                self.accuracy1 = tf.maximum(0., 1 - self.cost_mean)

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

                grads_l2, global_norm_l2 = tf.clip_by_global_norm(
                    tf.gradients(self.cost_mean, tvars),
                    self.max_grad_norm)

                tf.summary.scalar('global_grad_norm_l2', global_norm_l2)

                train_net = tf.cond(tf.equal(self.is_training, tf.constant(True)), lambda: True, lambda:False)
                #if is_training:
                print("Training Mode")

                #learning_rate1 = tf.train.exponential_decay(0.01, self.global_step, 50000, 0.98, staircase=True)

                self.train_task1_op = tf.train.GradientDescentOptimizer(0.001).minimize(self.cost_mean,
                                                                                        global_step=self.global_step)

                #gradients_before, variables = zip(*optimizer_1.compute_gradients(self.cost_mean))
                #gradients, _ = tf.clip_by_global_norm(gradients_before, 1.0)
                #self.train_task1_op = optimizer_1.apply_gradients(zip(gradients, variables))

                #learning_rate2 = tf.train.exponential_decay(0.001, self.global_step, 10000, 0.98, staircase=True)

                self.train_task2_op = tf.train.AdamOptimizer(0.01).minimize(self.loss, global_step=self.global_step)
                #gradients, variables = zip(*optimizer_2.compute_gradients(self.loss))
                #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                #self.train_task2_op = optimizer_2.apply_gradients(zip(gradients, variables))

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
            self.embedding_matrix = tf.get_variable(
                name="embedding_matrix",
                shape=[self.vocab_size, self.embedding_size],
                initializer=tf.random_normal_initializer(),
                # layers.xavier_initializer(),  # --TODO: CHANGE INITIALIZER OF THE EMBEDDING LAYER
                dtype=tf.float32, trainable=True)
            """
            W = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.embedding_size]),
                            trainable=False, name="W")
            
            self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_size])
            self.embedding_matrix = W.assign(self.embedding_placeholder)
            """
            self.inputs_src_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.inputs_src)
            self.inputs_trg_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.inputs_trg)
            self.inputs_doc_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.inputs_doc)

    def _init_task1(self, scope):
        """
        TASK 1: Multilingual Embeddings Alignment Variables
        :param scope:
        :return:
        """
        ## SOURCE
        with tf.variable_scope("src") as scope:
            self.word_level_inputs_src = tf.reshape(self.inputs_src_embedded,
                                                    [self.sentence_size_src, self.word_size_src, self.embedding_size])

            word_level_lengths_src = tf.reshape(self.word_lengths_src, [self.sentence_size_src])

        with tf.variable_scope('word') as scope:
            self.word_encoder_output_src, _ = bidirectional_rnn(
                self.word_cell, self.word_cell,
                self.word_level_inputs_src, word_level_lengths_src,
                scope=scope)

        with tf.variable_scope('attention') as scope:
            word_level_output_src = task_specific_attention(
                self.word_encoder_output_src,
                self.word_output_size,
                scope=scope)

        with tf.variable_scope('dropout'):
            self.word_level_output_src = layers.dropout(
                word_level_output_src, keep_prob=self.dropout_keep_proba,
                is_training=self.is_training,
            )
        ## TARGET
        with tf.variable_scope("trg") as scope:
            word_level_inputs_trg = tf.reshape(
                self.inputs_trg_embedded, [self.sentence_size_trg, self.word_size_trg, self.embedding_size])

            word_level_lengths_trg = tf.reshape(self.word_lengths_trg, [self.sentence_size_trg])

        with tf.variable_scope('word') as scope:
            word_encoder_output_trg, _ = bidirectional_rnn(
                self.word_cell, self.word_cell,
                word_level_inputs_trg, word_level_lengths_trg,
                scope=scope)

        with tf.variable_scope('attention', reuse=True) as scope:
            word_level_output_trg = task_specific_attention(
                word_encoder_output_trg,
                self.word_output_size,
                scope=scope)

        with tf.variable_scope('dropout'):
            self.word_level_output_trg = layers.dropout(
                word_level_output_trg, keep_prob=self.dropout_keep_proba,
                is_training=self.is_training,
            )

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

    def old_loss(self, src, trg, y, margin=0.0):
        normalize_src = tf.norm(src, ord="euclidean")
        normalize_trg = tf.norm(trg, ord="euclidean")
        self.cos_similarity = tf.reduce_sum(tf.multiply(normalize_src, normalize_trg))

        match_loss = 1 - self.cos_similarity
        labels = tf.to_float(y)
        labels = tf.transpose(labels, [1, 0])
        loss_match = tf.reduce_sum(tf.multiply(labels, match_loss))

        return loss_match

    def loss_src_trg(self, src, trg, y, margin=0.0):
        '''
        calucaltes loss depending on cosine similarity and labels
        if label == 1:
            loss = 1 - cosine
        else:
            loss = max(0,cosine - margin)
        x1 : a 2D tensor ( batch_size, embed)
        x2 : a 2D tensor
        y : batch label tensor
        margin : margin for negtive samples loss
        '''

        # take dot product of src,trg : [batch_size,1]
        # dot_products = tf.reduce_sum(tf.multiply(src, trg), axis=1)
        regularizer = tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

        # calulcate magnitude of two 1d tensors
        # normalize_src = tf.sqrt(tf.abs(tf.reduce_sum(tf.square(src), axis=1))) + 1e-5 * regularizer
        # normalize_trg = tf.sqrt(tf.abs(tf.reduce_sum(tf.square(trg), axis=1))) + 1e-5 * regularizer

        # calculate cosine distances between them

        # normalize_src = tf.norm(src, ord="euclidean", axis=1)
        # normalize_trg = tf.norm(trg, ord="euclidean", axis=1)
        # cosine = dot_products / tf.multiply(normalize_src, normalize_trg)

        # cosine = cosine + 1e-5 * regularizer

        # normalize_src = tf.norm(src, ord="euclidean", axis=1)
        normalize_src = tf.nn.l2_normalize(src, 0)
        # normalize_trg = tf.norm(trg, ord="euclidean", axis=1)
        normalize_trg = tf.nn.l2_normalize(trg, 0)
        cos_distance = tf.losses.cosine_distance(normalize_src, normalize_trg, 1)
        cosine = cos_distance + 1e-10 * regularizer

        """
        if cosine is None:
            print("Cosine is nan")
            tf.Print(normalize_src, [normalize_src])
            tf.Print(normalize_trg, [normalize_trg])
        """

        # convert it into float and make it a row vector
        labels = tf.to_float(y)
        labels = tf.transpose(labels, [1, 0])

        # you can try margin parameters, margin helps to set bound for mismatch cosine
        margin = tf.constant(margin)

        # calculate number of match and mismatch pairs
        total_labels = tf.to_float(tf.shape(labels)[1])
        match_size = tf.reduce_sum(labels)
        # mismatch_size = tf.subtract(total_labels, match_size)

        # loss culation for match and mismatch separately
        match_loss = cosine
        tf.Print(cosine, [cosine])
        mismatch_loss = tf.maximum(0., tf.subtract(cosine, margin), 'mismatch_term')
        tf.Print(mismatch_loss, [mismatch_loss])

        # combined loss for a batch
        loss_match = tf.reduce_sum(tf.multiply(labels, match_loss))
        # loss_mismatch = tf.reduce_sum(tf.multiply((1 - labels), mismatch_loss))
        tf.Print(loss_match, [loss_match])

        # combined total loss
        # if label is 1, only match_loss will count, otherwise mismatch_loss
        loss = tf.add(tf.multiply(labels, match_loss), \
                      tf.multiply((1 - labels), mismatch_loss), 'loss_add')

        tf.Print(loss, [loss])
        # take average for losses according to size
        loss_match_mean = tf.divide(loss_match, match_size)
        tf.Print(loss_match_mean, [loss_match_mean])
        tf.Print(total_labels, [total_labels])
        # loss_mismatch_mean = tf.divide(loss_mismatch, mismatch_size)
        loss_mean = tf.divide(tf.reduce_sum(loss), total_labels)

        return normalize_src, normalize_trg, cosine, mismatch_loss, loss_match, loss, total_labels, loss_mean, loss_match_mean, loss_match_mean  # loss_mismatch_mean

    """
        # Old loss 1 Stuff
        print("self.word_level_output_src: ", self.word_level_output_src.shape, " self.word_level_output_trg: ", self.word_level_output_trg)
        self.l2_norm = tf.norm(self.word_level_output_src-self.word_level_output_trg, ord='euclidean')
        #self.l2_loss = tf.reduce_mean(-tf.reduce_sum(self.l2_norm))#, reduction_indices=1))
        normalize_src = tf.norm(self.word_level_output_src, ord="euclidean")
        normalize_trg = tf.norm(self.word_level_output_trg, ord="euclidean")
        #normalize_src = self.word_level_output_src / tf.reduce_sum(tf.square(self.word_level_output_src), 0, keep_dims=True) #tf.nn.l2_normalize(self.word_level_output_src, 1)
        #normalize_trg = self.word_level_output_trg / tf.reduce_sum(tf.square(self.word_level_output_trg), 0, keep_dims=True)#tf.nn.l2_normalize(self.word_level_output_trg, 1)
        #print("tf.tensordot(normalize_src, normalize_trg, axes=2).shape:", tf.tensordot(normalize_src, normalize_trg, axes=2).shape)
        self.cos_similarity = tf.reduce_sum(tf.multiply(normalize_src, normalize_trg))
        #self.cos_similarity = tf.tensordot(normalize_src, normalize_trg, axes=2) #tf.nn.l2_normalize(tf.tensordot(normalize_src, normalize_trg, axes=2), 1)
        #self.l2_loss = -1 * tf.cast(cos_similarity, tf.float32) + 2
        #self.cos_sim = tf.losses.cosine_distance(self.word_level_output_src, self.word_level_output_trg, dim=1, weights=1.0, scope=None, loss_collection=tf.GraphKeys.LOSSES, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        #self.l2_loss = -1 * tf.cast(self.cos_sim, tf.float32) + 2
        #scaled_loss = tf.multiply(loss,
        #self.cos_similarity = tf.matmul(self.word_level_output_src, tf.transpose(self.word_level_output_trg, [1, 0]))
        self.l2_loss = self.cos_similarity + 1#- 0.5* tf.cast(self.cos_similarity, tf.float32) + 0.5
        tf.summary.scalar('loss1', self.l2_loss)
        
        self.accuracy1 = self.cos_similarity #0.5*self.cos_similarity+0.5 #1 / (1+self.l2_norm)
        tf.summary.scalar('accuracy1', self.accuracy1)
    """

    """
    def get_feed_data(self, doc, src, trg, y=None, class_weights=None, is_training=True):
        x_doc, doc_sizes, sent_sizes_doc = data_util.doc_batch(doc)
        x_s, x_t, sent_sizes_src = data_util.src_trg_batch(src, trg)
        fd = {
            self.inputs_src: x_s,
            self.inputs_trg: x_t,
            self.inputs_doc: x_doc,
            self.sentence_lengths_doc: doc_sizes,  # how many sentences
            self.word_lengths_doc: sent_sizes,  # how many words per sentence
        }
        if y is not None:
            fd[self.labels] = y
            if class_weights is not None:
                fd[self.sample_weights] = [class_weights[yy] for yy in y]
            else:
                fd[self.sample_weights] = np.ones(shape=[len(x_doc)], dtype=np.float32)
        fd[self.is_training] = is_training
        return fd
    """


"""
 with tf.variable_scope('train'):
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)

            self.loss = tf.reduce_mean(tf.multiply(self.cross_entropy, self.sample_weights))
            tf.summary.scalar('loss', self.loss)

            self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits, self.labels, 1), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

            tvars = tf.trainable_variables()

            grads, global_norm = tf.clip_by_global_norm(
                tf.gradients(self.loss, tvars),
                self.max_grad_norm)
            tf.summary.scalar('global_grad_norm', global_norm)

            class_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            self.train_class_opt = class_opt.apply_gradients(
                zip(grads, tvars), name='train_op',
                global_step=self.global_step)

            self.summary_op = tf.summary.merge_all()

"""
