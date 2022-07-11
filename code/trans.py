# -*- coding:utf-8 -*-
import os
import datetime
import json
import warnings
from collections import Counter
import gensim
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.1

def calc(g, valid_mask):
    bins=10
    momentum = 0
    edges_left = [float(x) / bins for x in range(bins)]
    edges_left = tf.constant(edges_left)  # [bins]
    edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1]
    edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1, 1]
    edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1, 1, 1]

    edges_right = [float(x) / bins for x in range(1, bins + 1)]
    edges_right[-1] += 1e-3
    edges_right = tf.constant(edges_right)  # [bins]
    edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1]
    edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1, 1]
    edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1, 1, 1]
    alpha = momentum
    # valid_mask = tf.cast(valid_mask, dtype=tf.bool)
    tot = tf.maximum(tf.reduce_sum(tf.cast(valid_mask, dtype=tf.float32)), 1.0)
    inds_mask = tf.logical_and(tf.greater_equal(g, edges_left), tf.less(g, edges_right))
    zero_matrix = tf.cast(tf.zeros_like(inds_mask), dtype=tf.float32)  # [bins, batch_num, class_num]

    inds = tf.cast(tf.logical_and(inds_mask, valid_mask), dtype=tf.float32)  # [bins, batch_num, class_num]

    num_in_bin = tf.reduce_sum(inds, axis=[1, 2, 3])  # [bins]
    valid_bins = tf.greater(num_in_bin, 0)  # [bins]

    num_valid_bin = tf.reduce_sum(tf.cast(valid_bins, dtype=tf.float32))

    if momentum > 0:
        acc_sum = [0.0 for _ in range(bins)]
        acc_sum = tf.Variable(acc_sum, trainable=False)

    if alpha > 0:
        update = tf.assign(acc_sum,tf.where(valid_bins, alpha * acc_sum + (1 - alpha) * num_in_bin, acc_sum))
        with tf.control_dependencies([update]):
            acc_sum_tmp = tf.identity(acc_sum, name='updated_accsum')
            acc_sum = tf.expand_dims(acc_sum_tmp, -1)  # [bins, 1]
            acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1]
            acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1, 1]
            acc_sum = acc_sum + zero_matrix  # [bins, batch_num, class_num]
            weights = tf.where(tf.equal(inds, 1), tot / acc_sum, zero_matrix)
            weights = tf.reduce_sum(weights, axis=0)
    else:
        num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1]
        num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1]
        num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1, 1]
        num_in_bin = num_in_bin + zero_matrix  # [bins, batch_num, class_num]
        weights = tf.where(tf.equal(inds, 1), tot / num_in_bin, zero_matrix)
        weights = tf.reduce_sum(weights, axis=0)
    weights = weights / num_valid_bin
    return weights, tot

def ghm(logits, targets, masks=None):
    train_mask = (1 - tf.cast(tf.equal(targets, -1), dtype=tf.float32))
    g_v = tf.abs(tf.sigmoid(logits) - targets)  # [batch_num, class_num]
    g = tf.expand_dims(g_v, axis=0)  # [1, batch_num, class_num]
    if masks is None:
        masks = tf.ones_like(targets)
    valid_mask = masks > 0
    weights, tot = calc(g, valid_mask)
    print(weights.shape)
    ghm_class_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets * train_mask,logits=logits)
    ghm_class_loss = tf.reduce_sum(ghm_class_loss * weights) / tot
    return ghm_class_loss

def dice(preds, labels, smooth=1):  # DL
    inse = tf.reduce_sum(preds * labels, axis=1)
    l = tf.reduce_sum(preds * preds, axis=1)
    r = tf.reduce_sum(labels * labels, axis=1)
    dice = (2. * inse + smooth) / (l + r + smooth)
    loss = tf.reduce_mean(1 - dice)
    return loss

class TrainingConfig(object):
    epoches = 100
    learningRate = 2 * 1e-3

class ModelConfig(object):
    embeddingSize = 300
    filters = 128
    numHeads = 8  # attention head
    numBlocks = 1  # transformer block
    epsilon = 1e-8
    keepProp = 0.9

    dropoutKeepProb = 0.5
    l2RegLambda = 0.0

class Config(object):
    sequenceLength = 300
    batchSize = 128
    stopWordSource = "./trans_refer/data/english"
    numClasses = 10
    rate = 0.8
    training = TrainingConfig()
    model = ModelConfig()

class Dataset(object):
    def __init__(self, config):  # #config
        self.config = config
        self._stopWordSource = config.stopWordSource

        self._sequenceLength = config.sequenceLength
        self._embeddingSize = config.model.embeddingSize
        self._batchSize = config.batchSize
        self._rate = config.rate

        self._stopWordDict = {}

        self.trainReviews = []
        self.trainLabels = []

        self.evalReviews = []
        self.evalLabels = []

        self.wordEmbedding = None

        self.labelList = []

    def _readData(self, lang, mode):
        reviews = []
        labels = []
        path = '/doc/zsy/data/data/' + mode + '/data-' + lang + '.txt'
        with open(path, 'r', encoding='utf-8') as fp:

            for f in fp:
                if f == '\n': continue
                word = f.split('<SPLIT>')
                reviews.append(word[-2] + ' ' + word[-1])
        path = '/doc/zsy/data/answer/' + mode + '/' + lang + '.txt'
        with open(path, 'r', encoding='utf-8') as fp:
            answer = fp.read().split('\n')
            for line in answer:
                if line == '': continue
                word = line.split('<SPLIT>')
                labels.append(word[-1])

        reviews = [line.strip().split() for line in reviews]

        l = int(0.8 * len(labels))
        np.random.seed(0)
        per = np.random.permutation(len(reviews))
        reviews1=np.array(reviews)[per]
        labels1=np.array(labels)[per]

        return list(reviews1)[:l], list(reviews1)[l:], list(labels1)[:l], list(labels1)[l:]

    def _labelToIndex(self, labels, label2idx):

        labelIds = [label2idx.get(label, 10) for label in labels]
        return labelIds  # #label indexs

    def _wordToIndex(self, reviews, word2idx):

        reviewIds = [[word2idx.get(item, word2idx["UNK"]) for item in review] for review in reviews]
        return reviewIds

    def _genTrainEvalData(self, x, y, word2idx, rate):

        reviews = []
        for review in x:
            if len(review) >= self._sequenceLength:
                reviews.append(review[:self._sequenceLength])
            else:
                reviews.append(review + [word2idx["PAD"]] * (self._sequenceLength - len(review)))


        trainReviews = np.asarray(reviews, dtype="int64")
        trainLabels = np.array(y, dtype="float32")
        return trainReviews, trainLabels

    def _genVocabulary(self, reviews, labels):

        allWords = [word for review in reviews for word in review]

        wordCount = Counter(allWords)
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)

        words = [item[0] for item in sortWordCount if item[1] >= 10]

        vocab, wordEmbedding = self._getWordEmbedding(words)
        self.wordEmbedding = wordEmbedding

        word2idx = dict(zip(vocab, list(range(len(vocab)))))

        uniqueLabel = list(set(labels))
        label2idx = dict(zip(uniqueLabel, list(range(len(uniqueLabel)))))
        self.labelList = list(range(len(uniqueLabel)))

        with open("./trans_refer/word2idx.json", "w", encoding="utf-8") as f:
            json.dump(word2idx, f)
        with open("./trans_refer/label2idx.json", "w", encoding="utf-8") as f:
            json.dump(label2idx, f)

        return word2idx, label2idx

    def _getWordEmbedding(self, words):
        wordVec = gensim.models.KeyedVectors.load_word2vec_format("./word2vec.bin", binary=True)
        vocab = []
        wordEmbedding = []

        vocab.append("PAD")
        vocab.append("UNK")
        wordEmbedding.append(np.zeros(self._embeddingSize))
        wordEmbedding.append(np.random.randn(self._embeddingSize))

        for word in words:
            try:
                vector = wordVec.wv[word]
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                continue
        return vocab, np.array(wordEmbedding)

    def _readStopWord(self, stopWordPath):

        with open(stopWordPath, "r") as f:
            stopWords = f.read()
            stopWordList = stopWords.splitlines()
            self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))

    def dataGen(self, lang, mode):
        self._readStopWord(self._stopWordSource)
        reviews_train, reviews_test, labels_train, labels_test = self._readData(lang, mode)
        word2idx, label2idx = self._genVocabulary(reviews_train, labels_train)
        labelIds = self._labelToIndex(labels_train, label2idx)
        reviewIds = self._wordToIndex(reviews_train, word2idx)
        labelIds_test = self._labelToIndex(labels_test, label2idx)
        reviewIds_test = self._wordToIndex(reviews_test, word2idx)
        train = []
        for inst in labelIds:
            a = [0] * config.numClasses
            a[int(inst) - 1] = 1
            train.append(a)
        eval = []
        for inst in labelIds_test:
            a = [0] * config.numClasses
            a[int(inst) - 1] = 1
            eval.append(a)

        labelIds = train
        labelIds_test = eval

        trainReviews, trainLabels = self._genTrainEvalData(reviewIds, labelIds, word2idx, self._rate)
        evalReviews, evalLabels = self._genTrainEvalData(reviewIds_test, labelIds_test, word2idx, self._rate)

        self.trainReviews = trainReviews
        self.trainLabels = trainLabels
        self.evalReviews = evalReviews
        self.evalLabels = evalLabels

def nextBatch(x, y, batchSize):
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]

    numBatches = len(x) // batchSize

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end], dtype="int64")
        batchY = np.array(y[start: end], dtype="float32")

        yield batchX, batchY

def fixedPositionEmbedding(batchSize, sequenceLen):
    embeddedPosition = []
    for batch in range(batchSize):
        x = []
        for step in range(sequenceLen):
            a = np.zeros(sequenceLen)
            a[step] = 1
            x.append(a)
        embeddedPosition.append(x)

    return np.array(embeddedPosition, dtype="float32")

class Transformer(object):
    # #transformer encoder
    def __init__(self, config, wordEmbedding):

        # ռλ
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.int32, [None, config.numClasses], name="inputY")

        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        self.embeddedPosition = tf.placeholder(tf.float32, [None, config.sequenceLength, config.sequenceLength],name="embeddedPosition")
        self.epoch = tf.placeholder_with_default(0.0, shape=())

        self.config = config

        l2Loss = tf.constant(0.0)

        with tf.name_scope("embedding"):
            self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"),name="W")
            self.embedded = tf.nn.embedding_lookup(self.W, self.inputX)
            self.embeddedWords = tf.concat([self.embedded, self.embeddedPosition], -1)

        with tf.name_scope("transformer"):
            for i in range(config.model.numBlocks):
                with tf.name_scope("transformer-{}".format(i + 1)):
                    multiHeadAtt = self._multiheadAttention(rawKeys=self.inputX, queries=self.embeddedWords,keys=self.embeddedWords)
                    self.embeddedWords = self._feedForward(multiHeadAtt,[config.model.filters,config.model.embeddingSize + config.sequenceLength])
            outputs = tf.reshape(self.embeddedWords,[-1, config.sequenceLength * (config.model.embeddingSize + config.sequenceLength)])

        outputSize = outputs.get_shape()[-1].value

        with tf.name_scope("dropout"):
            outputs = tf.nn.dropout(outputs, keep_prob=self.dropoutKeepProb)

        with tf.name_scope("output"):
            outputW = tf.get_variable("outputW",shape=[outputSize, config.numClasses],initializer=tf.contrib.layers.xavier_initializer())
            outputB = tf.Variable(tf.constant(0.1, shape=[config.numClasses]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.logits = tf.nn.xw_plus_b(outputs, outputW, outputB, name="logits")
            values, self.predictions = tf.nn.top_k(self.logits, 10, sorted=True, name="predictions")

        with tf.name_scope("loss"):
            # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)
            losses = ghm(tf.cast(self.logits, 'float32'), tf.cast(self.inputY, 'float32')) #ghm
            # losses = dice(tf.cast(self.logits, 'float32'), tf.cast(self.inputY, 'float32'))#dice
            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss

    def _layerNormalization(self, inputs, scope="layerNorm"):
        epsilon = self.config.model.epsilon
        inputsShape = inputs.get_shape()
        paramsShape = inputsShape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(paramsShape))
        gamma = tf.Variable(tf.ones(paramsShape))
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)
        outputs = gamma * normalized + beta
        return outputs

    def _multiheadAttention(self, rawKeys, queries, keys, numUnits=None, causality=False, scope="multiheadAttention"):
        numHeads = self.config.model.numHeads
        keepProp = self.config.model.keepProp
        if numUnits is None:
            numUnits = queries.get_shape().as_list()[-1]
        Q = tf.layers.dense(queries, numUnits, activation=tf.nn.relu)
        K = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)
        V = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)

        Q_ = tf.concat(tf.split(Q, numHeads, axis=-1), axis=0)
        K_ = tf.concat(tf.split(K, numHeads, axis=-1), axis=0)
        V_ = tf.concat(tf.split(V, numHeads, axis=-1), axis=0)
        similary = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        scaledSimilary = similary / (K_.get_shape().as_list()[-1] ** 0.5)
        keyMasks = tf.tile(rawKeys, [numHeads, 1])
        keyMasks = tf.tile(tf.expand_dims(keyMasks, 1), [1, tf.shape(queries)[1], 1])
        paddings = tf.ones_like(scaledSimilary) * (-2 ** (32 + 1))
        maskedSimilary = tf.where(tf.equal(keyMasks, 0), paddings,scaledSimilary)
        if causality:
            diagVals = tf.ones_like(maskedSimilary[0, :, :])
            tril = tf.contrib.linalg.LinearOperatorTriL(diagVals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0),[tf.shape(maskedSimilary)[0], 1, 1])
            paddings = tf.ones_like(masks) * (-2 ** (32 + 1))
            maskedSimilary = tf.where(tf.equal(masks, 0), paddings,maskedSimilary)
        weights = tf.nn.softmax(maskedSimilary)
        outputs = tf.matmul(weights, V_)
        outputs = tf.concat(tf.split(outputs, numHeads, axis=0), axis=2)
        outputs = tf.nn.dropout(outputs, keep_prob=keepProp)
        outputs += queries
        outputs = self._layerNormalization(outputs)
        return outputs

    def _feedForward(self, inputs, filters, scope="multiheadAttention"):
        params = {"inputs": inputs, "filters": filters[0], "kernel_size": 1,"activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        params = {"inputs": outputs, "filters": filters[1], "kernel_size": 1,"activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs += inputs
        outputs = self._layerNormalization(outputs)
        return outputs

    def _positionEmbedding(self, scope="positionEmbedding"):
        batchSize = self.config.batchSize
        sequenceLen = self.config.sequenceLength
        embeddingSize = self.config.model.embeddingSize
        positionIndex = tf.tile(tf.expand_dims(tf.range(sequenceLen), 0), [batchSize, 1])
        positionEmbedding = np.array([[pos / np.power(10000, (i - i % 2) / embeddingSize) for i in range(embeddingSize)]for pos in range(sequenceLen)])
        positionEmbedding[:, 0::2] = np.sin(positionEmbedding[:, 0::2])
        positionEmbedding[:, 1::2] = np.cos(positionEmbedding[:, 1::2])
        positionEmbedding_ = tf.cast(positionEmbedding, dtype=tf.float32)
        positionEmbedded = tf.nn.embedding_lookup(positionEmbedding_, positionIndex)
        return positionEmbedded

def mean(item: list) -> float:
    res = sum(item) / len(item) if len(item) > 0 else 0
    return res

def accuracy(pred_y, true_y):
    pred_1 = [inst[0].tolist() for inst in pred_y]
    pred_5 = [inst[0:5].tolist() for inst in pred_y]
    pred_10 = pred_y.tolist()
    corr_1 = 0
    corr_5 = 0
    corr_10 = 0
    for num in range(len(true_y)):
        inst = true_y[num]
        for i in range(config.numClasses):
            if inst[i] == 1:
                if i == pred_1[num]:
                    corr_1 += 1
                    corr_5 += 1
                    corr_10 += 1
                else:
                    if i in pred_5[num]:
                        corr_5 += 1
                        corr_10 += 1
                    else:
                        if i in pred_10[num]:
                            corr_10 += 1
                break
    acc_1 = corr_1 / len(pred_y) if len(pred_y) > 0 else 0
    acc_5 = corr_5 / len(pred_y) if len(pred_y) > 0 else 0
    acc_10 = corr_10 / len(pred_y) if len(pred_y) > 0 else 0
    return acc_1, acc_5, acc_10

def means(list):
    sum=0
    for i in list:
        if i >= 0:
            sum=sum+i
    return sum/len(list)

def f_measure(y_true,y_pred,n_classes=100):
    # 获取confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    #print(cm)
    cm = cm.astype(np.float32)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    # Sensitivity, hit rate, recall, or true positive rate
    recall = TP / (TP + FN)#recall
    R=means(recall)
    precision = TP / (TP + FP)#precision
    P=means(precision)
    F = (2 * precision * recall) / (precision + recall)
    F1 = means(F)
    return R,P,F1


config = Config()
data = Dataset(config)
for mode in ['10','50','100','200','500','700','1000']:
    for lang in  ['ruby', 'javascript', 'go', 'python', 'java','php']:
        f = open(lang +'-'+mode+'-ghm.txt', 'w+', encoding='utf-8')
        config.numClasses = int(mode)
        data.dataGen(lang, mode)
        print("{} {} train data shape: {}".format(lang, mode, data.trainReviews.shape))
        print("{} {} train label shape: {}".format(lang, mode, data.trainLabels.shape))
        print("{} {} eval data shape: {}".format(lang, mode, data.evalReviews.shape))

        print("{} {} train data shape: {}".format(lang, mode, data.trainReviews.shape), file=f)
        print("{} {} train label shape: {}".format(lang, mode, data.trainLabels.shape), file=f)
        print("{} {} eval data shape: {}".format(lang, mode, data.evalReviews.shape), file=f)

        trainReviews = data.trainReviews
        trainLabels = data.trainLabels
        evalReviews = data.evalReviews
        evalLabels = data.evalLabels

        wordEmbedding = data.wordEmbedding
        labelList = data.labelList

        embeddedPosition = fixedPositionEmbedding(config.batchSize, config.sequenceLength)


        best_step = 0
        best_f1 = 0
        best_acc = 0
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            print('use：',tf.test.is_gpu_available())
            session_conf.gpu_options.per_process_gpu_memory_fraction = 0.5
            sess = tf.Session(config=session_conf)
            if not os.path.exists('/doc/zsy/trans/model/' + lang):
                os.mkdir('/doc/zsy/trans/model/' + lang)

            with sess.as_default():
                transformer = Transformer(config, wordEmbedding)
                globalStep = tf.Variable(0, name="globalStep", trainable=False)
                learning_rate = tf.train.polynomial_decay(config.training.learningRate, config.training.epoches,
                                                          transformer.epoch, 1e-4, power=1)
                optimizer = tf.train.AdamOptimizer(learning_rate)
                gradsAndVars = optimizer.compute_gradients(transformer.loss)
                trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)
                saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())

                def trainStep(batchX, batchY, i):
                    feed_dict = {
                        transformer.inputX: batchX,
                        transformer.inputY: batchY,
                        transformer.dropoutKeepProb: config.model.dropoutKeepProb,
                        transformer.embeddedPosition: embeddedPosition,
                        transformer.epoch: i + 1
                    }
                    _, step, loss, predictions, epoch = sess.run(
                        [trainOp, globalStep, transformer.loss, transformer.predictions, transformer.epoch],feed_dict)
                    lable=np.argmax(batchY, axis=1).tolist()
                    pred = [inst[0].tolist() for inst in predictions]
                    F1 = f1_score(lable, pred, average='macro')
                    P = precision_score(lable, pred, average='macro')
                    R = recall_score(lable, pred, average='macro')
                    # R,P,F1=f_measure(y_true=lable,y_pred=pred)
                    acc_1, acc_5, acc_10 = accuracy(pred_y=predictions, true_y=batchY)
                    return loss, acc_1, epoch,R,P,F1

                def devStep(batchX, batchY, i):
                    feed_dict = {
                        transformer.inputX: batchX,
                        transformer.inputY: batchY,
                        transformer.dropoutKeepProb: 1.0,
                        transformer.embeddedPosition: embeddedPosition,
                        transformer.epoch: i + 1
                    }
                    step, loss, predictions = sess.run(
                        [globalStep, transformer.loss, transformer.predictions],feed_dict)
                    lable=np.argmax(batchY, axis=1).tolist()
                    pred = [inst[0].tolist() for inst in predictions]
                    F1 = f1_score(lable, pred, average='macro')
                    P = precision_score(lable, pred, average='macro')
                    R = recall_score(lable, pred, average='macro')
                    # R,P,F1=f_measure(y_true=lable,y_pred=pred)
                    acc_1, acc_5, acc_10 = accuracy(pred_y=predictions, true_y=batchY)
                    return loss, acc_1, acc_5, acc_10,R,P,F1

                for i in range(config.training.epoches):
                    print("start training model {} {}".format(lang, mode))
                    print("start training model{} {}".format(lang, mode), file=f)
                    for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize):
                        loss, acc, epoch,R,P,F1 = trainStep(batchTrain[0], batchTrain[1], i)
                        currentStep = tf.train.global_step(sess, globalStep)
                        if currentStep%10 == 0:
                            print("train: step: {},epoch: {}, loss: {}, acc: {}, recall: {}, precision: {}, f-measure: {}".format(currentStep, epoch, loss, acc,R,P,F1))
                            print("train: step: {},epoch: {}, loss: {}, acc: {}, recall: {}, precision: {}, f-measure: {}".format(currentStep, epoch, loss, acc,R,P,F1),file=f)
                    print("\nEvaluation {}:".format(lang))
                    print("\nEvaluation{}:".format(lang), file=f)

                    losses = []
                    accs_1 = []
                    accs_5 = []
                    accs_10 = []
                    Rs= []
                    Ps = []
                    F1s = []

                    for batchEval in nextBatch(evalReviews, evalLabels, config.batchSize):
                        loss, acc_1, acc_5, acc_10,R,P,F1 = devStep(batchEval[0], batchEval[1], i)
                        losses.append(loss)
                        accs_1.append(acc_1)
                        accs_5.append(acc_5)
                        accs_10.append(acc_10)
                        Rs.append(R)
                        Ps.append(P)
                        F1s.append(F1)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}, step: {}, loss: {}, acc_1: {}, acc_5: {}, acc_10: {}, recall: {}, precision: {}, f-measure: {}".format(time_str, currentStep,
                                                                                            mean(losses), mean(accs_1),mean(accs_5),mean(accs_10),mean(Rs),mean(Ps),mean(F1s)))
                    print("{}, step: {}, loss: {}, acc_1: {}, acc_5: {}, acc_10: {}, recall: {}, precision: {}, f-measure: {}".format(time_str, currentStep,
                                                                                            mean(losses), mean(accs_1), mean(accs_5),mean(accs_10),mean(Rs),mean(Ps),mean(F1s)), file=f)
                    if mean(accs_1) >= best_acc:
                        best_acc = mean(accs_1)
                        best_step = currentStep
                    mo='/doc/zsy/trans/model/'+lang+'/model.ckpt'
                    if os.path.exists(mo):
                        os.mkdir(mo)
                    save_path=saver.save(sess,mo)

                print("{} best_step:{},best_acc:{}".format(lang, best_step, best_acc))
                print("{} best_step:{},best_acc:{}".format(lang, best_step, best_acc), file=f)

