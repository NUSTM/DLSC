#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import os, sys
sys.path.append(os.getcwd())

from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from newbie_nn.nn_layer import softmax_layer, cnn_layer
from newbie_nn.config import *
from data_prepare.utils import load_w2v, batch_index, load_inputs_sentence

tf.app.flags.DEFINE_integer('max_sentence_len', 150, 'max number of tokens per sentence')


def pooling(inputs, dim=1, _type='max'):
    if _type == 'max':
        return tf.reduce_max(inputs, dim, False)
    elif _type == 'avg':
        return tf.reduce_mean(inputs, dim, False)


def cnn(inputs, sen_len, keep_prob1, keep_prob2, _id='1'):
    print 'I am cnn.'
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    inputs = tf.reshape(inputs, [-1, FLAGS.max_sentence_len, FLAGS.embedding_dim, 1])

    conv1 = cnn_layer(inputs, [3, FLAGS.embedding_dim, 1, FLAGS.n_hidden], [1, 1, 1, 1], 'VALID', FLAGS.random_base,
                      FLAGS.l2_reg, scope_name='conv1'+_id)
    conv1 = tf.reshape(conv1, [-1, FLAGS.max_sentence_len - 2, FLAGS.n_hidden])
    conv1 = pooling(conv1, 1, 'max')

    conv2 = cnn_layer(inputs, [2, FLAGS.embedding_dim, 1, FLAGS.n_hidden], [1, 1, 1, 1], 'VALID', FLAGS.random_base,
                      FLAGS.l2_reg, scope_name='conv2'+_id)
    conv2 = tf.reshape(conv2, [-1, FLAGS.max_sentence_len - 1, FLAGS.n_hidden])
    conv2 = pooling(conv2, 1, 'max')

    conv3 = cnn_layer(inputs, [1, FLAGS.embedding_dim, 1, FLAGS.n_hidden], [1, 1, 1, 1], 'VALID', FLAGS.random_base,
                      FLAGS.l2_reg, scope_name='conv3'+_id)
    conv3 = tf.reshape(conv3, [-1, FLAGS.max_sentence_len - 0, FLAGS.n_hidden])
    conv3 = pooling(conv3, 1, 'max')

    outputs = (conv1 + conv2 + conv3) / 3.

    return softmax_layer(outputs, FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class, _id)


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

    prob = cnn(inputs, sen_len, keep_prob1, keep_prob2, FLAGS.t1)

    y_p = tf.argmax(prob, 1)
    y_t = tf.argmax(y, 1)

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

        max_acc, max_prob, max_ty, max_py = 0., None, None, None
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
                    y_true = []
                    for test, num in get_batch_data(te_x, te_sen_len, te_y, 2200, 1.0, 1.0, False):
                        _loss, _acc, _step, y_true_tmp, y_pred_tmp, y_prob_tmp = sess.run(
                            [loss, acc_num, global_step, y_t, y_p, prob], feed_dict=test)
                        y_prob.extend(y_prob_tmp)
                        y_pred.extend(y_pred_tmp)
                        y_true.extend(y_true_tmp)
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
                        max_py = y_pred
                        max_ty = y_true
        P = precision_score(max_ty, max_py, average=None)
        R = recall_score(max_ty, max_py, average=None)
        F1 = f1_score(max_ty, max_py, average=None)
        print 'P:', P, 'avg=', sum(P) / FLAGS.n_class
        print 'R:', R, 'avg=', sum(R) / FLAGS.n_class
        print 'F1:', F1, 'avg=', sum(F1) / FLAGS.n_class

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

