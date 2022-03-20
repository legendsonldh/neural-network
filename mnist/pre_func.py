#-*-coding:utf-8-*-

import os
import struct
import numpy as np
import math
from scipy.spatial.distance import pdist,cdist, squareform


## 导入MNIST数据集

def load_mnist(path, kind='train'):
    """

    :param path: 输入数据存放文件夹目录
    :param kind: 输入读取的数据类型：train or test
    :return: 返回一个字典dict：包含"images"和"labels"
    """
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    mnist = {'images':images,
             'labels':labels}
    return mnist

## 独热编码：将0-9转换成类似向量[0,0,0,...,1]

def one_hot_encoding(labels):
    """

    :param labels: 输入一列0-9范围内的向量数据
    :return: 将向量数据扩充到一列x10的矩阵
    """
    w = labels.shape[0]
    blank = np.zeros((w,10))
    for i in range(w):
        blank[i][labels[i]] = 1

    return blank


## 导入数据并进行独热编码
def input_data():
    """
    执行load_mnist和one_hot_encoding两个函数
    :return: MNIST训练集和测试集
    """
    mnist_train = load_mnist('MNIST_data', kind='train')
    print(mnist_train['images'].shape)
    mnist_test = load_mnist('MNIST_data', kind='t10k')
    print(mnist_test['images'].shape)

    mnist_train['labels'] = one_hot_encoding(mnist_train['labels'])
    mnist_test['labels'] = one_hot_encoding(mnist_test['labels'])

    return mnist_train,mnist_test


## 构造并初始化网络的结构、初始化参数矩阵

def init_parameters(X,Y,n_hiddenlayer,pattern):
    """
    :param X:       输入训练集
    :param Y:       结果训练集
    :param n_hiddenlayer: 隐藏层单元数矩阵
    :param pattern:   选取不同模式下的系数矩阵归一化方法
    :return :n_layer  各层单元数矩阵;
            :parameter 权重矩阵，含有W与B 比如有三层，那么权重矩阵中只有W1,W2,W3;B1,B2,B3
    """
    n_layer = [X.shape[1]] + n_hiddenlayer + [Y.shape[1]]

    np.random.seed(1)
    ## 初始化权重矩阵
    parameters = {}
    for l in range(1, len(n_layer)):
        if pattern == "random":
            parameters['W' + str(l)] = np.random.randn(n_layer[l], n_layer[l - 1])*0.0001
            parameters['b' + str(l)] = np.zeros((n_layer[l], 1))
        elif pattern == "He":
            parameters['W' + str(l)] = np.random.randn(n_layer[l], n_layer[l - 1]) * np.sqrt(2/n_layer[l-1])
            parameters['b' + str(l)] = np.zeros((n_layer[l], 1))

    return n_layer,parameters

## 归一化函数
def norm_data(X, pattern):
    """
    :param X: 输入矩阵
    :param pattern: avg1:使用均值和标准差来构建归一化
                    avg2:使用最大和最小值来构建归一化
    :return:
    """
    if pattern == "avg1":
        ## 预处理1:输入数据的归一化(这个归一化并不是在[0-1]之间)
        X_mean = np.mean(X, axis=1, keepdims=True)
        X_std = np.std(X, axis=1, keepdims=True)
        X_norm = (X - X_mean) / X_std
    if pattern == "avg2":
        ## 预处理2：使用最简单归一化
        X_max = np.amax(X, axis=1, keepdims=True)
        X_min = np.amin(X, axis=1, keepdims=True)
        X_norm = (X - X_min) / (X_max - X_min)


    return X_norm

## Pca 降维维度选择
def dim_select(eigVals,percentage):
    sortArray = np.sort(eigVals)
    sortArray = sortArray[-1::-1]
    sumArray  = np.sum(sortArray)

    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum+= i
        num+=1
        if tmpSum >= sumArray*percentage:
            return num


## 1D-PCA
def one_dim_PCA(X_norm,percentage,pattern,main_dim,num):
    """

    :param X_norm:
    :param percentage:
    :param pattern:
    :return:
    """

    if pattern == "train":

        # 协方差矩阵
        cov_mat = np.cov(X_norm, rowvar=False)

        # 求特征值和特征向量
        eigVals, eigVects = np.linalg.eig(cov_mat)

        # 按最大值排序
        eigValIndice = np.argsort(eigVals)

        # 根据权重百分比选择降维的维度
        if num == 0:
            dim_num = dim_select(eigVals,percentage)
        if num == 2:
            dim_num = num
        n_eigValIndice = eigValIndice[-1:-(dim_num + 1):-1]
        n_eigVect = eigVects[:, n_eigValIndice]

    if pattern == "test":

        n_eigVect = main_dim

    # 低维度空间数据
    X_norm_pca = np.real(np.dot(X_norm,n_eigVect))

    return X_norm_pca,n_eigVect

## Kernel-PCA_相似度矩阵核
def one_dim_kernel_PCA(X_norm,X_cache,gamma,percentage,pattern,main_dim,num):

    if pattern == "train":

        K = np.zeros((X_norm.shape[0],X_norm.shape[0]))
        for i in range(X_norm.shape[0]):
            for j in range(X_norm.shape[0]):
                K[i, j] = np.sum(np.exp(-gamma * (X_norm[i, :] - X_norm[j, :]) ** 2))

        # 核均值归一化
        N = K.shape[0]
        one_N = np.ones((N, N)) / N

        K = K - one_N.dot(K) - K.dot(one_N) + one_N.dot(K).dot(one_N)

        # 求特征值和特征向量
        eigVals, eigVects = np.linalg.eig(K)
        # 按最大值排序
        eigValIndice = np.argsort(eigVals)

        # 根据权重百分比选择降维的维度
        if num == 0:
            dim_num = dim_select(eigVals,percentage)
        if num == 1:
            dim_num = num
        n_eigValIndice = eigValIndice[-1:-(dim_num + 1):-1]

        n_eigVect = eigVects[:, n_eigValIndice]

    if pattern == "test":

        n_eigVect = main_dim

        K = np.zeros((X_norm.shape[0],X_cache.shape[0]))
        for i in range(X_norm.shape[0]):
            for j in range(X_cache.shape[0]):
                K[i, j] = np.sum(np.exp(-gamma * (X_norm[i, :] - X_cache[j, :]) ** 2))

        # 核均值归一化
        N = K.shape[0]
        one_N = np.ones((N, N)) / N

        K = K - one_N.dot(K) - K.dot(one_N) + one_N.dot(K).dot(one_N)

    # 低维度空间数据
    X_norm_pca = np.real(np.dot(K,n_eigVect))

    return X_norm_pca,X_norm,n_eigVect

## 2D-2D：PCA，不成熟版本
"""
img_test = X_norm[0].reshape(28,28)
plt.imshow(img_test,cmap='Greys')
plt.show()


A = X_norm[0].reshape(28,28)
cov_row=np.cov(A,rowvar=True)
cov_column=np.cov(A,rowvar= False)

eigVal_row,eigVect_row=np.linalg.eig(np.mat(cov_row))

eigVal_column,eigVect_column=np.linalg.eig(np.mat(cov_column))

## 行向量

eigValIndice = np.argsort(eigVal_row)

n_eigValIndice = eigValIndice[-1:-(16+1):-1]

n_eigVect_row = eigVect_row[:,n_eigValIndice]

## 列向量

eigValIndice = np.argsort(eigVal_column)

n_eigValIndice = eigValIndice[-1:-(16+1):-1]

n_eigVect_column = eigVect_column[:,n_eigValIndice]


## 2D-2D PCA

C = np.dot(np.dot(n_eigVect_column.T,A),n_eigVect_row)

A_= np.dot(np.dot(n_eigVect_column,C),n_eigVect_row.T)

Q = np.dot(np.dot(n_eigVect_row.T,A),n_eigVect_column)

plt.imshow(Q,cmap='Greys')
plt.show()

"""


# 随机mini_batch 化

def random_mini_batches(X_train, mini_batch_size, seed):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    X = X_train['images']
    Y = X_train['labels']

    np.random.seed(seed)
    m = X.shape[0]
    mini_batches = []

    # Step 1: 将训练数据和标签集随机打乱，Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: 按照每一个mini_batch的大小进行划分Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[k * mini_batch_size: (k + 1) * mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: (k + 1) * mini_batch_size,:]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # 把除不尽的部分补上:Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[(k+1) * mini_batch_size: m ,:]
        mini_batch_Y = shuffled_Y[(k+1) * mini_batch_size: m ,:]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches