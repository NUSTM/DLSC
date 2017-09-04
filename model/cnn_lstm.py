#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import os, sys
sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score

from newbie_nn.config import *
from newbie_nn.nn_layer import dynamic_rnn, bi_dynamic_rnn, softmax_layer, cnn_layer, reduce_mean_with_len
from newbie_nn.att_layer import mlp_attention_layer
from data_prepare.utils import load_w2v, batch_index, load_word_embedding, load_inputs_document


def cnn_lstm(inputs, sen_len, doc_len, keep_prob1, keep_prob2):
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    inputs = tf.reshape(inputs, [-1, FLAGS.max_sentence_len, FLAGS.embedding_dim, 1])

    conv1 = cnn_layer(inputs, [3, FLAGS.embedding_dim, 1, FLAGS.n_hidden], [1, 1, 1, 1], 'VALID', FLAGS.random_base, FLAGS.l2_reg, scope_name='conv1')
    conv1 = tf.reshape(conv1, [-1, FLAGS.max_sentence_len - 2, FLAGS.n_hidden])
    conv1 = reduce_mean_with_len(conv1, sen_len - 2)

    conv2 = cnn_layer(inputs, [2, FLAGS.embedding_dim, 1, FLAGS.n_hidden], [1, 1, 1, 1], 'VALID', FLAGS.random_base, FLAGS.l2_reg, scope_name='conv2')
    conv2 = tf.reshape(conv2, [-1, FLAGS.max_sentence_len - 1, FLAGS.n_hidden])
    conv2 = reduce_mean_with_len(conv2, sen_len - 1)

    conv3 = cnn_layer(inputs, [1, FLAGS.embedding_dim, 1, FLAGS.n_hidden], [1, 1, 1, 1], 'VALID', FLAGS.random_base, FLAGS.l2_reg, scope_name='conv3')
    conv3 = tf.reshape(conv3, [-1, FLAGS.max_sentence_len - 0, FLAGS.n_hidden])
    conv3 = reduce_mean_with_len(conv3, sen_len - 0)

    outputs = (conv1 + conv2 + conv3) / 3.

    outputs = tf.reshape(outputs, [-1, FLAGS.max_doc_len, FLAGS.n_hidden])
    cell = tf.nn.rnn_cell.LSTMCell
    outputs = bi_dynamic_rnn(cell, outputs, FLAGS.n_hidden, doc_len, FLAGS.max_doc_len, 'doc', 'all_avg')

    prob = softmax_layer(outputs, 2 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class)
    return prob


def main(_):
    word_id_mapping, w2v = load_w2v(FLAGS.embedding_file_path, FLAGS.embedding_dim, True)
    word_embedding = tf.constant(w2v, dtype=tf.float32, name='word_embedding')
    # word_embedding = tf.Variable(w2v, name='word_embedding')

    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sentence_len])
        y = tf.placeholder(tf.float32, [None, FLAGS.n_class])
        sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
        doc_len = tf.placeholder(tf.int32, None)
        keep_prob1 = tf.placeholder(tf.float32)
        keep_prob2 = tf.placeholder(tf.float32)

    inputs = tf.nn.embedding_lookup(word_embedding, x)
    prob = cnn_lstm(inputs, sen_len, doc_len, keep_prob1, keep_prob2)

    loss = loss_func(y, prob)
    acc_num, acc_prob = acc_func(y, prob)
    global_step = tf.Variable(0, name='tr_global_step', trainable=False)
    optimizer = train_func(loss, FLAGS.learning_rate, global_step)
    true_y = tf.argmax(y, 1)
    pred_y = tf.argmax(prob, 1)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        import time
        timestamp = str(int(time.time()))
        _dir = 'summary/' + str(timestamp) + '_cnn'
        test_loss = tf.placeholder(tf.float32)
        test_acc = tf.placeholder(tf.float32)
        train_summary_op, test_summary_op, validate_summary_op, train_summary_writer, test_summary_writer, \
        validate_summary_writer = summary_func(loss, acc_prob, test_loss, test_acc, _dir, title, sess)

        save_dir = 'temp_model/' + str(timestamp) + '_cnn' + '/'
        saver = saver_func(save_dir)

        init = tf.initialize_all_variables()
        sess.run(init)

        # saver.restore(sess, '/-')

        tr_x, tr_y, tr_sen_len, tr_doc_len = load_inputs_document(
            FLAGS.train_file_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            FLAGS.max_doc_len,
            'CNN'
        )
        te_x, te_y, te_sen_len, te_doc_len = load_inputs_document(
            FLAGS.test_file_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            FLAGS.max_doc_len,
            'CNN'
        )
        # v_x, v_y, v_sen_len, v_doc_len = load_inputs_document(
        #     FLAGS.validate_file_path,
        #     word_id_mapping,
        #     FLAGS.max_sentence_len,
        #     FLAGS.max_doc_len,
        #     'CNN'
        # )

        def get_batch_data(x_in, y_in, sen_len_in, doc_len_in, batch_size, kp1, kp2, is_shuffle=True):
            for index in batch_index(len(y_in), batch_size, 1, is_shuffle):
                feed_dict = {
                    x: x_in[index],
                    y: y_in[index],
                    sen_len: sen_len_in[index],
                    doc_len: doc_len_in[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)

        max_acc = 0.
        max_prob, max_ty, max_py = None, None, None
        step = None
        for i in xrange(FLAGS.n_iter):
            for train, _ in get_batch_data(tr_x, tr_y, tr_sen_len, tr_doc_len, FLAGS.batch_size,
                                           FLAGS.keep_prob1, FLAGS.keep_prob2):
                _, step, summary = sess.run([optimizer, global_step, train_summary_op], feed_dict=train)
                train_summary_writer.add_summary(summary, step)

            acc, cost, cnt = 0., 0., 0
            p, ty, py = [], [], []
            for test, num in get_batch_data(te_x, te_y, te_sen_len, te_doc_len, 2000, 1.0, 1.0, False):
                _loss, _acc, _p, _ty, _py = sess.run([loss, acc_num, prob, true_y, pred_y], feed_dict=test)
                p += list(_p)
                ty += list(_ty)
                py += list(_py)
                acc += _acc
                cost += _loss * num
                cnt += num
            print 'all samples={}, correct prediction={}'.format(cnt, acc)
            acc = acc / cnt
            cost = cost / cnt
            print 'Iter {}: mini-batch loss={:.6f}, test acc={:.6f}'.format(i, cost, acc)
            summary = sess.run(test_summary_op, feed_dict={test_loss: cost, test_acc: acc})
            test_summary_writer.add_summary(summary, step)
            if acc > max_acc:
                max_acc = acc
                max_prob = p
                max_ty = ty
                max_py = py
        print 'P:', precision_score(max_ty, max_py, average=None)
        print 'R:', recall_score(max_ty, max_py, average=None)
        print 'F:', f1_score(max_ty, max_py, average=None)

        fp = open(FLAGS.prob_file, 'w')
        for item in max_prob:
            fp.write(' '.join([str(it) for it in item]) + '\n')

        print 'Optimization Finished! Max acc={}'.format(max_acc)

        print 'Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
            FLAGS.learning_rate,
            FLAGS.n_iter,
            FLAGS.batch_size,
            FLAGS.n_hidden,
            FLAGS.l2_reg
        )


if __name__ == '__main__':
    tf.app.run()
