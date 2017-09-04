#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import os, sys
sys.path.append(os.getcwd())

from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import tensorflow as tf
from newbie_nn.nn_layer import bi_dynamic_rnn, softmax_layer
from newbie_nn.att_layer import dot_produce_attention_layer
from newbie_nn.config import *
from data_prepare.utils import load_w2v, batch_index, load_inputs_sentence


def bilstm(inputs, length, keep_prob1, keep_prob2, type_='last'):
    print 'I am bilstm.'
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    cell = tf.nn.rnn_cell.LSTMCell
    hidden = bi_dynamic_rnn(cell, inputs, FLAGS.n_hidden, length, FLAGS.max_sentence_len, 'bilstm', type_)
    return softmax_layer(hidden, 2 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class)


def main(_):
    word_id_mapping, w2v = load_w2v(FLAGS.embedding_file_path, FLAGS.embedding_dim, True)
    word_embedding = tf.constant(w2v, dtype=tf.float32, name='word_embedding')

    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)

    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len], name='x')
        y = tf.placeholder(tf.float32, [None, FLAGS.n_class], name='y')
        sen_len = tf.placeholder(tf.int32, [None], name='sen_len')

    inputs = tf.nn.embedding_lookup(word_embedding, x)

    prob = bilstm(inputs, sen_len, keep_prob1, keep_prob2, FLAGS.t1)

    y_p = tf.argmax(prob, 1)

    loss = loss_func(y, prob)
    acc_num, acc_prob = acc_func(y, prob)
    global_step = tf.Variable(0, name='tr_global_step', trainable=False)
    optimizer = train_func(loss, FLAGS.learning_rate, global_step)

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

    with tf.Session() as sess:
        import time
        timestamp = str(int(time.time()))
        _dir = 'summary/' + str(timestamp) + '_' + title
        _dir = 'summary/' + FLAGS.train_file_path + '_' + str(timestamp) + '/'
        test_loss = tf.placeholder(tf.float32)
        test_acc = tf.placeholder(tf.float32)
        train_summary_op, test_summary_op, validate_summary_op, train_summary_writer, test_summary_writer, \
        validate_summary_writer = summary_func(loss, acc_prob, test_loss, test_acc, _dir, title, sess)

        save_dir = 'temp_model/' + str(timestamp) + '_' + title + '/'
        save_dir = 'temp_model/' + FLAGS.train_file_path + '/'
        saver = saver_func(save_dir)

        init = tf.initialize_all_variables()
        sess.run(init)


        tr_x, tr_sen_len, tr_y = load_inputs_sentence(
            FLAGS.train_file_path,
            word_id_mapping,
            FLAGS.max_sentence_len
        )
        te_x, te_sen_len, te_y = load_inputs_sentence(
            FLAGS.test_file_path,
            word_id_mapping,
            FLAGS.max_sentence_len
        )

        def get_batch_data(xi, sen_leni, yi, batch_size, kp1, kp2, is_shuffle=True):
            for index in batch_index(len(yi), batch_size, 1, is_shuffle):
                feed_dict = {
                    x: xi[index],
                    y: yi[index],
                    sen_len: sen_leni[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)

        max_acc, max_prob = 0., []
        for i in xrange(FLAGS.n_iter):
            for train, _ in get_batch_data(tr_x, tr_sen_len, tr_y, FLAGS.batch_size,
                                                FLAGS.keep_prob1, FLAGS.keep_prob2):
                _, step, summary = sess.run([optimizer, global_step, train_summary_op], feed_dict=train)
                train_summary_writer.add_summary(summary, step)
                if step % FLAGS.display_step == 0:
                    saver.save(sess, save_dir, global_step=step)
                if step % FLAGS.display_step == 0:
                    acc, cost, cnt = 0., 0., 0
                    y_prob = []
                    y_pred = []
                    for test, num in get_batch_data(te_x, te_sen_len, te_y, 2200, 1.0, 1.0, False):
                        _loss, _acc, _summary, _step, y_pred_tmp, y_prob_tmp = sess.run(
                            [loss, acc_num, test_summary_op, global_step, y_p, prob],
                            feed_dict=test)
                        y_prob.extend(y_prob_tmp)
                        y_pred.extend(y_pred_tmp)
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
                        max_prob = y_prob
        fp = open(FLAGS.prob_file, 'w')
        for pb in max_prob:
            fp.write(str(pb[0]) + ' ' + str(pb[1]) + '\n')
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

