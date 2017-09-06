#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 25, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_float('lr', 0.01, 'learning rate')
tf.app.flags.DEFINE_integer('n_class', 5, 'number of distinct class')
tf.app.flags.DEFINE_integer('n_sen_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_doc_len', 20, 'max number of tokens per sentence')
tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
tf.app.flags.DEFINE_float('random_base', 0.01, 'initial random base')
tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
tf.app.flags.DEFINE_integer('n_iter', 10, 'number of train iter')
tf.app.flags.DEFINE_float('keep_prob1', 1.0, 'dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'dropout keep prob')
tf.app.flags.DEFINE_string('t1', 'last', 'type of hidden output')
tf.app.flags.DEFINE_string('t2', 'last', 'type of hidden output')
tf.app.flags.DEFINE_integer('n_layer', 3, 'number of stacked rnn')
tf.app.flags.DEFINE_string('pre_trained', 'yes', 'whether has pre-trained embedding')
tf.app.flags.DEFINE_string('embedding_type', 'static', 'embedding type: static or non-static')
tf.app.flags.DEFINE_integer('early_stopping', 5, 'the number of early stopping epoch')
tf.app.flags.DEFINE_integer('decay_steps', 1000, 'decay steps of learning rate')
tf.app.flags.DEFINE_float('decay_rate', 0.9, 'decay rate of learning rate')
tf.app.flags.DEFINE_string('model', 'hnn', 'models: hnn, han, rnn_cnn, cnn_rnn')


tf.app.flags.DEFINE_string('train_file', 'data/restaurant/rest_2014_lstm_train_new.txt', 'training file')
tf.app.flags.DEFINE_string('val_file', 'data/restaurant/rest_2014_lstm_test_new.txt', 'validating file')
tf.app.flags.DEFINE_string('test_file', 'data/restaurant/rest_2014_lstm_test_new.txt', 'testing file')
tf.app.flags.DEFINE_string('train_file_r', 'data/restaurant/rest_2014_lstm_train_new.txt', 'training file')
tf.app.flags.DEFINE_string('val_file_r', 'data/restaurant/rest_2014_lstm_test_new.txt', 'validating file')
tf.app.flags.DEFINE_string('test_file_r', 'data/restaurant/rest_2014_lstm_test_new.txt', 'testing file')
tf.app.flags.DEFINE_string('embedding_file', 'data/restaurant/rest_2014_word_embedding_300_new.txt', 'embedding file')
tf.app.flags.DEFINE_string('word2id_file', 'data/restaurant/word_id_new.txt', 'word-id mapping file')
tf.app.flags.DEFINE_string('prob_file', 'prob.txt', 'prob')
tf.app.flags.DEFINE_string('weights_save_path', 'weights_save', 'the path of saving weights')
