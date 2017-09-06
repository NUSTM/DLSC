#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import numpy as np


def batch_index(length, batch_size, n_iter=100, is_shuffle=True, is_train=True):
    index = range(length)
    for j in xrange(n_iter):
        if is_shuffle:
            np.random.shuffle(index)
        if is_train:
            batch_num = length / batch_size
        else:
            batch_num = length / batch_size + (1 if length % batch_size else 0)
        for i in xrange(batch_num):
            yield index[i * batch_size:(i + 1) * batch_size]


def load_word2id(word2id_file, encoding='utf8'):
    """
    :param word2id_file: word-id mapping file path
    :param encoding: file's encoding, for changing to unicode
    :return: word-id mapping, like hello=5
    """
    word2id = dict()
    for line in open(word2id_file):
        line = line.decode(encoding, 'ignore').lower().split()
        word2id[line[0]] = int(line[1])
    print '\nload word-id mapping done!\n'
    return word2id


def load_w2v(w2v_file, embedding_dim, is_skip=False):
    fp = open(w2v_file)
    if is_skip:
        fp.readline()
    w2v = []
    word_dict = dict()
    # [0,0,...,0] represent absent words
    w2v.append([0.] * embedding_dim)
    cnt = 0
    for line in fp:
        line = line.decode('utf8').split()
        # line = line.split()
        if len(line) != embedding_dim + 1:
            print u'a bad word embedding: {}'.format(line[0])
            continue
        cnt += 1
        w2v.append([float(v) for v in line[1:]])
        word_dict[line[0]] = cnt
    w2v = np.asarray(w2v, dtype=np.float32)
    w2v = np.row_stack((w2v, np.sum(w2v, axis=0) / cnt))
    print np.shape(w2v)
    word_dict['$t$'] = (cnt + 1)
    # w2v -= np.mean(w2v, axis=0)
    # w2v /= np.std(w2v, axis=0)
    print word_dict['$t$'], len(w2v)
    return word_dict, w2v


def change_y_to_onehot(y):
    from collections import Counter
    print Counter(y)
    class_set = set(y)
    n_class = len(class_set)
    y_onehot_mapping = dict(zip(class_set, range(n_class)))
    print y_onehot_mapping
    fp = open('data/y2id.txt', 'w')
    for k, v in y_onehot_mapping.items():
        fp.write(str(k) + ' ' + str(v) + '\n')
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)


def load_y2id_id2y(f):
    y2id = dict()
    id2y = dict()
    for line in open(f):
        y, id = line.split()
        y2id[y] = int(id)
        id2y[int(id)] = y
    return y2id, id2y


def load_inputs_document(input_file, word_id_file, max_sen_len, max_doc_len, _type=None, encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word2id(word_id_file)
    else:
        word_to_id = word_id_file
    print 'load word-to-id done!'

    x, y, sen_len, doc_len = [], [], [], []
    f1 = open(input_file)
    for l1 in f1:
        l1 = l1.lower().decode('utf8', 'ignore').split('||')

        t_sen_len = [0] * max_doc_len
        t_x = np.zeros((max_doc_len, max_sen_len))
        doc = ' '.join(l1[1:])
        sentences = doc.split('<sssss>')
        i = 0
        flag = False
        for sentence in sentences:
            j = 0
            words = sentence.split()
            # if '<pos>' not in words and '<neg>' not in words:
            #     continue
            for word in words:
                if j < max_sen_len:
                    if word in word_to_id:
                        t_x[i, j] = word_to_id[word]
                        j += 1
                else:
                    break
            if j > 2:
                t_sen_len[i] = j
                i += 1
                flag = True
            if i >= max_doc_len:
                break
        if flag:
            doc_len.append(i)
            sen_len.append(t_sen_len)
            x.append(t_x)
            y.append(l1[0])

    y = change_y_to_onehot(y)
    print 'load input {} done!'.format(input_file)

    return np.asarray(x), np.asarray(sen_len), np.asarray(doc_len), np.asarray(y)


def load_inputs_document_sen_3(input_file, word_id_file, max_sen_len, max_doc_len, _type=None, encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word2id(word_id_file)
    else:
        word_to_id = word_id_file
    print 'load word-to-id done!'

    x, y, sen_len, doc_len, sen_y = [], [], [], [], []
    f1 = open(input_file)
    c1, c2, c3 = 0, 0, 0
    for l1 in f1:
        l1 = l1.lower().decode('utf8', 'ignore').split('||')
        # y.append(line[0])

        t_sen_len = [0] * max_doc_len
        t_sen_y = [[0, 0, 0]] * max_doc_len
        t_x = np.zeros((max_doc_len, max_sen_len))
        doc = ' '.join(l1[1:])
        sentences = doc.split('<sssss>')
        i = 0
        flag = False
        for sentence in sentences:
            j = 0
            for word in sentence.split():
                if j < max_sen_len:
                    if word in word_to_id:
                        t_x[i, j] = word_to_id[word]
                        j += 1
                else:
                    break
            if j > 2:
                t_sen_len[i] = j
                if '<pos>' in sentence:
                    c1 += 1
                    t_sen_y[i] = [1, 0, 0]
                elif '<neg>' in sentence:
                    c2 += 1
                    t_sen_y[i] = [0, 1, 0]
                else:
                    c3 += 1
                    t_sen_y[i] = [0, 0, 1]
                i += 1
                flag = True
            if i >= max_doc_len:
                break
        if flag:
            doc_len.append(i)
            sen_len.append(t_sen_len)
            x.append(t_x)
            y.append(l1[0])
            sen_y.append(t_sen_y)

    print c1, c2, c3
    y = change_y_to_onehot(y)
    # sen_y = change_y_to_onehot(sen_y)
    print 'load input {} done!'.format(input_file)

    return np.asarray(x), np.asarray(sen_len), np.asarray(doc_len), np.asarray(sen_y), np.asarray(y)

def load_inputs_document_sen_2(input_file, word_id_file, max_sen_len, max_doc_len, _type=None, encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word2id(word_id_file)
    else:
        word_to_id = word_id_file
    print 'load word-to-id done!'

    x, y, sen_len, doc_len, sen_y = [], [], [], [], []
    f1 = open(input_file)
    c1, c2, c3 = 0, 0, 0
    for l1 in f1:
        l1 = l1.lower().decode('utf8', 'ignore').split('||')
        # y.append(line[0])

        t_sen_len = [0] * max_doc_len
        t_sen_y = [[0, 0]] * max_doc_len
        t_x = np.zeros((max_doc_len, max_sen_len))
        doc = ' '.join(l1[1:])
        sentences = doc.split('<sssss>')
        i = 0
        flag = False
        for sentence in sentences:
            j = 0
            for word in sentence.split():
                if j < max_sen_len:
                    if word in word_to_id:
                        t_x[i, j] = word_to_id[word]
                        j += 1
                else:
                    break
            if j > 2:
                t_sen_len[i] = j
                if '<pos>' in sentence:
                    c1 += 1
                    t_sen_y[i] = [1, 0]
                elif '<neg>' in sentence:
                    c2 += 1
                    t_sen_y[i] = [1, 0]
                else:
                    c3 += 1
                    t_sen_y[i] = [0, 1]
                i += 1
                flag = True
            if i >= max_doc_len:
                break
        if flag:
            doc_len.append(i)
            sen_len.append(t_sen_len)
            x.append(t_x)
            y.append(l1[0])
            sen_y.append(t_sen_y)

    print c1, c2, c3
    y = change_y_to_onehot(y)
    # sen_y = change_y_to_onehot(sen_y)
    print 'load input {} done!'.format(input_file)

    return np.asarray(x), np.asarray(sen_len), np.asarray(doc_len), np.asarray(sen_y), np.asarray(y)

def load_inputs_sentence(input_file, word_id_file, sentence_len, encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word2id(word_id_file)
    else:
        word_to_id = word_id_file
    print 'load word-to-id done!'

    x, y, sen_len = [], [], []
    for line in open(input_file):
        line = line.lower().decode('utf8', 'ignore').split('||')
        y.append(line[0])

        words = ' '.join(line[1:]).split()
        xx = []
        i = 0
        for word in words:
            if word in word_to_id:
                xx.append(word_to_id[word])
                i += 1
                if i >= sentence_len:
                    break
        sen_len.append(len(xx))
        xx = xx + [0] * (sentence_len - len(xx))
        x.append(xx)
    y = change_y_to_onehot(y)
    print 'load input {} done!'.format(input_file)

    return np.asarray(x), np.asarray(sen_len), np.asarray(y)