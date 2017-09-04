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
from newbie_nn.nn_layer import dynamic_rnn, bi_dynamic_rnn, softmax_layer
from newbie_nn.att_layer import mlp_attention_layer, Mlp_attention_layer
from data_prepare.utils import load_w2v, batch_index, load_word_embedding, load_inputs_document


def hn_att(inputs, sen_len, doc_len, keep_prob1, keep_prob2):
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    cell = tf.contrib.rnn.LSTMCell
    sen_len = tf.reshape(sen_len, [-1])
    hiddens_sen = bi_dynamic_rnn(cell, inputs, FLAGS.n_hidden, sen_len, FLAGS.max_sentence_len, 'sentence', 'all')
    alpha_sen = Mlp_attention_layer(hiddens_sen, sen_len, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 1)
    outputs_sen = tf.reshape(tf.batch_matmul(alpha_sen, hiddens_sen), [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden])

    sen_len = tf.reshape(sen_len, [-1, FLAGS.max_doc_len])
    # alpha = 1.0 - tf.cast(tf.reshape(sen_len / (tf.reduce_sum(sen_len, 1, keep_dims=True) + 1), [-1, FLAGS.max_doc_len, 1]), tf.float32)
    # outputs_new = alpha * outputs_sen

    hiddens_doc = bi_dynamic_rnn(cell, outputs_sen, FLAGS.n_hidden, doc_len, FLAGS.max_doc_len, 'doc', 'all')
    alpha_doc = Mlp_attention_layer(hiddens_doc, doc_len, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 2)
    outputs_doc = tf.reshape(tf.batch_matmul(alpha_doc, hiddens_doc), [-1, 2 * FLAGS.n_hidden])

    prob = softmax_layer(outputs_doc, 2 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class)
    return prob, tf.reshape(alpha_sen, [-1, FLAGS.max_doc_len, FLAGS.max_sentence_len]), tf.reshape(alpha_doc, [-1, FLAGS.max_doc_len])


def hn(inputs, sen_len, doc_len, keep_prob1, keep_prob2, id_=1):
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    cell = tf.contrib.rnn.LSTMCell
    sen_len = tf.reshape(sen_len, [-1])
    hiddens_sen = bi_dynamic_rnn(cell, inputs, FLAGS.n_hidden, sen_len, FLAGS.max_sentence_len, 'sentence' + str(id_), FLAGS.t1)
    hiddens_sen = tf.reshape(hiddens_sen, [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden])
    hidden_doc = bi_dynamic_rnn(cell, hiddens_sen, FLAGS.n_hidden, doc_len, FLAGS.max_doc_len, 'doc' + str(id_), FLAGS.t2)

    return softmax_layer(hidden_doc, 2 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class)


def main(_):
    with tf.device('/gpu:0'):
        word_id_mapping, w2v = load_w2v(FLAGS.embedding_file_path, FLAGS.embedding_dim, True)
        word_embedding = tf.constant(w2v, dtype=tf.float32, name='word_embedding')
        # word_embedding = tf.Variable(w2v, name='word_embedding')

        keep_prob1 = tf.placeholder(tf.float32)
        keep_prob2 = tf.placeholder(tf.float32)

        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sentence_len])
            y = tf.placeholder(tf.float32, [None, FLAGS.n_class])
            sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
            doc_len = tf.placeholder(tf.int32, None)

        inputs = tf.nn.embedding_lookup(word_embedding, x)
        inputs = tf.reshape(inputs, [-1, FLAGS.max_sentence_len, FLAGS.embedding_dim])

        if FLAGS.method == 'ATT':
            prob, alpha_sen, alpha_doc = hn_att(inputs, sen_len, doc_len, keep_prob1, keep_prob2)
        else:
            prob = hn(inputs, sen_len, doc_len, keep_prob1, keep_prob2)

        loss = loss_func(y, prob)
        acc_num, acc_prob = acc_func(y, prob)
        global_step = tf.Variable(0, name='tr_global_step', trainable=False)
        optimizer = train_func(loss, FLAGS.learning_rate, global_step)
        true_y = tf.argmax(y, 1)
        pred_y = tf.argmax(prob, 1)

        title = '-d1-{}d2-{}b-{}r-{}l2-{}sen-{}dim-{}h-{}c-{}'.format(
            FLAGS.keep_prob1,
            FLAGS.keep_prob2,
            FLAGS.batch_size,
            FLAGS.learning_rate,
            FLAGS.l2_reg,
            FLAGS.max_sentence_len,
            FLAGS.embedding_dim,
            FLAGS.n_hidden,
            FLAGS.n_class
        )

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        import time
        timestamp = str(int(time.time()))
        _dir = 'summary/' + str(timestamp) + '_' + title
        test_loss = tf.placeholder(tf.float32)
        test_acc = tf.placeholder(tf.float32)
        train_summary_op, test_summary_op, validate_summary_op, train_summary_writer, test_summary_writer, \
            validate_summary_writer = summary_func(loss, acc_prob, test_loss, test_acc, _dir, title, sess)

        save_dir = 'temp_model/' + str(timestamp) + '_' + title + '/'
        saver = saver_func(save_dir)

        init = tf.global_variables_initializer()
        sess.run(init)

        # saver.restore(sess, '/-')

        tr_x, tr_y, tr_sen_len, tr_doc_len = load_inputs_document(
            FLAGS.train_file_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            FLAGS.max_doc_len
        )
        te_x, te_y, te_sen_len, te_doc_len = load_inputs_document(
            FLAGS.test_file_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            FLAGS.max_doc_len
        )
        # v_x, v_y, v_sen_len, v_doc_len = load_inputs_document(
        #     FLAGS.validate_file_path,
        #     word_id_mapping,
        #     FLAGS.max_sentence_len,
        #     FLAGS.max_doc_len
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
        max_alpha_s, max_alpha_d = None, None
        step = None
        for i in xrange(FLAGS.n_iter):
            train_alpha_doc = []
            for train, _ in get_batch_data(tr_x, tr_y, tr_sen_len, tr_doc_len, FLAGS.batch_size,
                                                FLAGS.keep_prob1, FLAGS.keep_prob2):
                _, step, summary = sess.run([optimizer, global_step, train_summary_op], feed_dict=train)
                train_summary_writer.add_summary(summary, step)
                # embed_update = tf.assign(word_embedding, tf.concat(0, [tf.zeros([1, FLAGS.embedding_dim]), word_embedding[1:]]))
                # sess.run(embed_update)

            # saver.save(sess, save_dir, global_step=step)

            acc, cost, cnt = 0., 0., 0
            p, ty, py = [], [], []
            alpha_s, alpha_d = [], []
            for test, num in get_batch_data(te_x, te_y, te_sen_len, te_doc_len, 2000, 1.0, 1.0, False):
                if FLAGS.method == 'ATT':
                    _loss, _acc, _p, _ty, _py, _alpha_sen, _alpha_doc = sess.run(
                        [loss, acc_num, prob, true_y, pred_y, alpha_sen, alpha_doc], feed_dict=test)
                    alpha_s += list(_alpha_sen)
                    alpha_d += list(_alpha_doc)
                else:
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
                max_alpha_s = alpha_s
                max_alpha_d = alpha_d
        print 'P:', precision_score(max_ty, max_py, average=None)
        print 'R:', recall_score(max_ty, max_py, average=None)
        print 'F:', f1_score(max_ty, max_py, average=None)

        if FLAGS.method == 'ATT':
            fp = open('alpha_sen_' + FLAGS.prob_file, 'w')
            for doc in max_alpha_s:
                for item in doc:
                    fp.write(' '.join([str(it) for it in item]) + '\n')
            fp = open('alpha_doc_' + FLAGS.prob_file, 'w')
            for item in max_alpha_d:
                fp.write(' '.join([str(it) for it in item]) + '\n')

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
