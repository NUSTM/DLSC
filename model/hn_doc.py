#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import os, sys
sys.path.append(os.getcwd())
import numpy as np

from utils.config import *
from utils.data_helper import load_w2v, load_inputs_document, load_word2id, batch_index, load_y2id_id2y
from newbie_nn.nn_layer import bi_dynamic_rnn, softmax_layer
from newbie_nn.att_layer import mlp_attention_layer


class HN_DOC(object):

    def __init__(self, filter_list=(2, 3, 4), filter_num=100):
        self.config = FLAGS
        self.filter_list = filter_list
        self.filter_num = filter_num
        self.add_placeholder()
        inputs = self.add_embedding()
        if self.config.model == 'cnn_rnn':
            self.doc_logits = self.cnn_rnn(inputs)
        elif self.config.model == 'rnn_cnn':
            self.doc_logits = self.rnn_cnn(inputs)
        elif self.config.model == 'han':
            self.doc_logits = self.han(inputs)
        else:
            self.doc_logits = self.hnn(inputs)
        self.doc_loss = self.add_loss(self.doc_logits)
        self.accuracy, self.accuracy_num = self.add_accuracy(self.doc_logits)
        self.train_op = self.add_train_op(self.doc_loss)

    def add_placeholder(self):
        self.x = tf.placeholder(tf.int32, [None, self.config.max_doc_len, self.config.max_sentence_len])
        self.doc_y = tf.placeholder(tf.float32, [None, self.config.n_class])
        self.sen_len = tf.placeholder(tf.int32, [None, self.config.max_doc_len])
        self.doc_len = tf.placeholder(tf.int32, [None])
        self.keep_prob1 = tf.placeholder(tf.float32)
        self.keep_prob2 = tf.placeholder(tf.float32)

    def add_embedding(self):
        if self.config.pre_trained == 'yes':
            self.word2id, w2v = load_w2v(self.config.embedding_file, self.config.embedding_dim, is_skip=True)
        else:
            self.word2id = load_word2id(self.config.word2id_file)
            self.vocab_size = len(self.word2id) + 1
            w2v = tf.random_uniform([self.vocab_size, self.config.embedding_dim], -1.0, 1.0)
        if self.config.embedding_type == 'static':
            self.embedding = tf.constant(w2v, dtype=tf.float32, name='word_embedding')
        else:
            self.embedding = tf.Variable(w2v, dtype=tf.float32, name='word_embedding')
        inputs = tf.nn.embedding_lookup(self.embedding, self.x)
        return inputs

    def create_feed_dict(self, x_batch, sen_len_batch, doc_len_batch, y_batch=None, kp1=1.0, kp2=1.0):
        if y_batch is None:
            holder_list = [self.x, self.sen_len, self.doc_len, self.keep_prob1, self.keep_prob2]
            feed_list = [x_batch, sen_len_batch, doc_len_batch, kp1, kp2]
        else:
            holder_list = [self.x, self.sen_len, self.doc_len, self.doc_y, self.keep_prob1, self.keep_prob2]
            feed_list = [x_batch, sen_len_batch, doc_len_batch, y_batch, kp1, kp2]
        return dict(zip(holder_list, feed_list))

    def add_cnn_layer(self, inputs, inputs_dim, max_len, scope_name='cnn'):
        inputs = tf.expand_dims(inputs, -1)
        pooling_outputs = []
        for i, filter_size in enumerate(self.filter_list):
            ksize = [filter_size, inputs_dim]
            conv = tf.contrib.layers.conv2d(inputs=inputs,
                                            num_outputs=self.filter_num,
                                            kernel_size=ksize,
                                            stride=1,
                                            padding='VALID',
                                            activation_fn=tf.nn.relu,
                                            scope='conv_' + scope_name + str(i))
            ksize = [max_len - filter_size + 1, 1]
            pooling = tf.contrib.layers.max_pool2d(inputs=conv,
                                                   kernel_size=ksize,
                                                   stride=1,
                                                   padding='VALID',
                                                   scope='pooling_' + scope_name)
            pooling_outputs.append(pooling)
        hiddens = tf.concat(pooling_outputs, 3)
        hiddens = tf.reshape(hiddens, [-1, self.filter_num * len(self.filter_list)])
        return hiddens

    def cnn_rnn(self, inputs):
        print 'I am CNN-RNN!'
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob1)
        inputs = tf.reshape(inputs, [-1, self.config.max_sentence_len, self.config.embedding_dim])
        # word-sentence
        outputs_sen = self.add_cnn_layer(inputs, self.config.embedding_dim, self.config.max_sentence_len, 'sen')
        outputs_sen_dim = self.filter_num * len(self.filter_list)
        outputs_sen = tf.reshape(outputs_sen, [-1, self.config.max_doc_len, outputs_sen_dim])
        # sentence-document
        cell = tf.contrib.rnn.LSTMCell
        outputs_doc = bi_dynamic_rnn(cell, outputs_sen, self.config.n_hidden, self.doc_len,
                                     self.config.max_doc_len, 'doc', 'last')
        # fully-connection
        logits = softmax_layer(outputs_doc, 2 * self.config.n_hidden, self.config.random_base, self.keep_prob2,
                               self.config.l2_reg, self.config.n_class, 'doc_softmax')
        return logits

    def rnn_cnn(self, inputs):
        print 'I am RNN-CNN!'
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob1)
        inputs = tf.reshape(inputs, [-1, self.config.max_sentence_len, self.config.embedding_dim])
        # word-sentence
        cell = tf.contrib.rnn.LSTMCell
        outputs_sen = bi_dynamic_rnn(cell, inputs, self.config.n_hidden, tf.reshape(self.sen_len, [-1]),
                                     self.config.max_sentence_len, 'sen', 'last')
        outputs_sen = tf.reshape(outputs_sen, [-1, self.config.max_doc_len, 2 * self.config.n_hidden])
        # sentence-document
        outputs_doc = self.add_cnn_layer(outputs_sen, 2 * self.config.n_hidden, self.config.max_doc_len, 'doc')
        outputs_doc_dim = self.filter_num * len(self.filter_list)
        # fully-connection
        logits = softmax_layer(outputs_doc, outputs_doc_dim, self.config.random_base, self.keep_prob2,
                               self.config.l2_reg, self.config.n_class, 'doc_softmax')
        return logits

    # hierarchical neural network (word-sentence-document)
    def hnn(self, inputs):
        print 'I am LSTM-LSTM!'
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob1)
        inputs = tf.reshape(inputs, [-1, self.config.max_sentence_len, self.config.embedding_dim])
        # word-sentence
        cell = tf.contrib.rnn.LSTMCell
        outputs_sen = bi_dynamic_rnn(cell, inputs, self.config.n_hidden, tf.reshape(self.sen_len, [-1]),
                                     self.config.max_sentence_len, 'sen', 'last')
        outputs_sen = tf.reshape(outputs_sen, [-1, self.config.max_doc_len, 2 * self.config.n_hidden])
        # sentence-document
        outputs_doc = bi_dynamic_rnn(cell, outputs_sen, self.config.n_hidden, self.doc_len,
                                     self.config.max_doc_len, 'doc', 'last')
        # fully-connection
        doc_logits = softmax_layer(outputs_doc, 2 * self.config.n_hidden, self.config.random_base,
                                   self.keep_prob2, self.config.l2_reg, self.config.n_class, 'doc_softmax')
        return doc_logits

    # hierarchical attention neural network
    def han(self, inputs):
        print 'I am HAN!'
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob1)
        inputs = tf.reshape(inputs, [-1, self.config.max_sentence_len, self.config.embedding_dim])
        # word-sentence
        cell = tf.contrib.rnn.LSTMCell
        sen_len = tf.reshape(self.sen_len, [-1])
        hiddens_sen = bi_dynamic_rnn(cell, inputs, self.config.n_hidden, sen_len,
                                     self.config.max_sentence_len, 'sen', 'all')
        alpha = mlp_attention_layer(hiddens_sen, sen_len, 2 * self.config.n_hidden, self.config.l2_reg,
                                    self.config.random_base, 'sen')
        outputs_sen = tf.squeeze(tf.matmul(alpha, hiddens_sen))
        outputs_sen = tf.reshape(outputs_sen, [-1, self.config.max_doc_len, 2 * self.config.n_hidden])
        # sentence-document
        hiddens_doc = bi_dynamic_rnn(cell, outputs_sen, self.config.n_hidden, self.doc_len,
                                     self.config.max_doc_len, 'doc', 'all')
        alpha = mlp_attention_layer(hiddens_doc, self.doc_len, 2 * self.config.n_hidden, self.config.l2_reg,
                                    self.config.random_base, 'doc')
        outputs_doc = tf.squeeze(tf.matmul(alpha, hiddens_doc))
        # fully-connection
        doc_logits = softmax_layer(outputs_doc, 2 * self.config.n_hidden, self.config.random_base,
                                   self.keep_prob2, self.config.l2_reg, self.config.n_class, 'doc_softmax')
        return doc_logits

    def add_loss(self, doc_scores):
        doc_loss = tf.nn.softmax_cross_entropy_with_logits(logits=doc_scores, labels=self.doc_y)
        self.doc_vars = [var for var in tf.global_variables() if 'doc' in var.name or 'sen' in var.name]
        print self.doc_vars
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='doc_softmax')
        print reg_loss
        loss = tf.reduce_mean(doc_loss) + tf.add_n(reg_loss)
        return loss

    def add_accuracy(self, scores):
        correct_predicts = tf.equal(tf.argmax(scores, 1), tf.argmax(self.doc_y, 1))
        accuracy_num = tf.reduce_sum(tf.cast(correct_predicts, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_predicts, tf.float32), name='accuracy')
        return accuracy, accuracy_num

    def add_train_op(self, doc_loss):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.train.exponential_decay(self.config.lr, global_step, self.config.decay_steps,
                                             self.config.decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(self.lr)
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(doc_loss, self.doc_vars), 5.0)
        train_op = optimizer.apply_gradients(zip(grads, self.doc_vars), name='train_op', global_step=global_step)
        # train_op = optimizer.minimize(doc_loss, global_step=global_step, var_list=self.doc_vars)
        return train_op

    def run_op(self, sess, op, data_x, sen_len, doc_len, doc_y=None, kp1=1.0, kp2=1.0):
        res_list = []
        len_list = []
        for indices in batch_index(len(data_x), self.config.batch_size, 1, False, False):
            if doc_y is not None:
                feed_dict = self.create_feed_dict(data_x[indices], sen_len[indices], doc_len[indices], doc_y[indices], kp1, kp2)
            else:
                feed_dict = self.create_feed_dict(data_x[indices], sen_len[indices], doc_len[indices], None, kp1, kp2)
            res = sess.run(op, feed_dict=feed_dict)
            res_list.append(res)
            len_list.append(len(indices))
        if type(res_list[0]) is list:
            res = np.concatenate(res_list, axis=1)
        elif op is self.accuracy_num:
            res = sum(res_list)
        elif op is self.doc_logits:
            res = np.concatenate(np.asarray(res_list), 0)
        else:
            res = sum(res_list) * 1.0 / len(len_list)
        return res

    def run(self, sess, feed_dict):
        _, loss, acc_num = sess.run([self.train_op, self.doc_loss, self.accuracy_num], feed_dict=feed_dict)
        return loss, acc_num


def test_case(sess, classifier, data_x, sen_len, doc_len, doc_y):
    loss = classifier.run_op(sess, classifier.doc_loss, data_x, sen_len, doc_len, doc_y)
    acc_num = classifier.run_op(sess, classifier.accuracy_num, data_x, sen_len, doc_len, doc_y)
    return acc_num * 1.0 / len(doc_y), loss


def train_run(_):
    sys.stdout.write('Training start:\n')
    with tf.device('/gpu:0'):
        classifier = HN_DOC()
    saver = tf.train.Saver(tf.global_variables())
    save_path = classifier.config.weights_save_path + '/weights'

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        best_accuracy = 0
        best_val_epoch = 0
        best_test_acc = 0
        train_x, train_sen_len, train_doc_len,  train_doc_y = load_inputs_document(
            FLAGS.train_file, classifier.word2id, FLAGS.max_sentence_len, FLAGS.max_doc_len)
        test_x, test_sen_len, test_doc_len, test_doc_y = load_inputs_document(
            FLAGS.test_file, classifier.word2id, FLAGS.max_sentence_len, FLAGS.max_doc_len)
        val_x, val_sen_len, val_doc_len, val_doc_y = load_inputs_document(
            FLAGS.val_file, classifier.word2id, FLAGS.max_sentence_len, FLAGS.max_doc_len)
        for epoch in range(FLAGS.n_iter):
            print '=' * 20 + 'Epoch ', epoch, '=' * 20
            total_loss = []
            total_acc_num = []
            total_num = []
            for step, indices in enumerate(batch_index(len(train_doc_y), FLAGS.batch_size, 1), 1):
                feed_dict = classifier.create_feed_dict(train_x[indices], train_sen_len[indices],
                                                        train_doc_len[indices], train_doc_y[indices],
                                                        FLAGS.keep_prob1, FLAGS.keep_prob2)
                loss, acc_num = classifier.run(sess, feed_dict=feed_dict)
                total_loss.append(loss)
                total_acc_num.append(acc_num)
                total_num.append(len(indices))
                verbose = FLAGS.display_step
                if step % verbose == 0:
                    print '\n[INFO] Epoch {} - {} : loss = {}, acc = {}'.format(
                        epoch, step, np.mean(total_loss[-verbose:]),
                        sum(total_acc_num[-verbose:]) * 1.0 / sum(total_num[-verbose:])
                    )
            loss = np.mean(total_loss)
            acc = sum(total_acc_num) * 1.0 / sum(total_num)
            print '[INFO] Epoch {} : mean loss = {}, mean acc = {}'.format(epoch, loss, acc)
            if np.isnan(loss):
                print '[Error] loss is not a number!'
                break
            print '=' * 50
            val_accuracy, val_loss = test_case(sess, classifier, val_x, val_sen_len, val_doc_len, val_doc_y)
            print '[INFO] val loss: {}, val acc: {}'.format(val_loss, val_accuracy)
            test_accuracy, test_loss = test_case(sess, classifier, test_x, test_sen_len, test_doc_len, test_doc_y)
            print '[INFO] test loss: {}, test acc: {}'.format(test_loss, test_accuracy)
            if best_accuracy < val_accuracy:
                best_accuracy = val_accuracy
                best_val_epoch = epoch
                best_test_acc = test_accuracy
                if not os.path.exists(classifier.config.weights_save_path):
                    os.makedirs(classifier.config.weights_save_path)
                saver.save(sess, save_path=save_path)
            if epoch - best_val_epoch > classifier.config.early_stopping:
                print 'Normal early stop!'
                break
        print 'Best val acc = {}'.format(best_accuracy)
        print 'Test acc = {}'.format(best_test_acc)

        saver.restore(sess, save_path)
        print 'Model restored from %s' % save_path
        run_test(sess, classifier, test_x, test_sen_len, test_doc_len)

    print 'Training complete!'


def run_test(sess, classifier, data_x, sen_len, doc_len, doc_y=None):
    score = classifier.run_op(sess, classifier.doc_logits, data_x, sen_len, doc_len, doc_y)
    py = np.argmax(score, axis=1)
    _, id2y = load_y2id_id2y('data/y2id.txt')
    fp = open('prediction_label.txt', 'w')
    for id in py:
        fp.write(id2y[id] + '\n')
    print 'Testing complete!'


if __name__ == '__main__':
    tf.app.run(train_run)







