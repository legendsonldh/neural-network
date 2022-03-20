#-*-coding:utf-8-*-
import matplotlib.pyplot as plt
import numpy as np
from pre_func import init_parameters
from pre_func import random_mini_batches

## 四大函数(Sigmoid & tanh & ReLU & ReLU_update & Softmax)
def activation(X,pattern):
    """
    :param X:
    :param pattern: 4种函数模式：Sigmoid tanh ReLU Softmax
    :return:
    """
    if pattern == "Sigmoid":
        value = 1/(1+np.exp(-X))
    if pattern == "tanh":
        value = np.tanh(X)
    if pattern == "ReLU":
        value = (np.abs(X)+X)/2

    ## 考虑上下溢出的问题
    if pattern == "Softmax":

        e = np.exp(X-np.max(X,axis=0,keepdims=True))

        value = e/np.sum(e,axis=0,keepdims=True)

    return value

## 四大函数的导数
def diff(X,pattern):
    """

    :param X:
    :param pattern: 4种函数模式：Sigmoid tanh ReLU Softmax
    :return:
    """
    g = activation(X, pattern)
    if pattern == "Sigmoid":
        differ = g*(1-g)
    if pattern == "tanh":
        differ = 1-g**2
    if pattern == "ReLU":
        g[g <= 0] = 0
        g[g >  0] = 1
        differ = g
    if pattern == "Softmax":
        differ = g*(1-g)

    return differ

## 构建一个ReLU->ReLU->ReLU->Softmax的Forward Prop模型
def Forwardpro(X,Y,n_layer,parameters) -> object:
    """
    :param X: 已经归一化之后的输入值
    :param Y: 目标值
    :param n_layer: 总网络层数
    :param parameters: 含有W和B矩阵的参数矩阵
    :return: 含有每一层的输入和输出矩阵：Z和A矩阵的cache矩阵
    """
    ## 初始化缓存矩阵
    cache = {}
    ## 将输入赋予第一个A值
    cache['A'+str(0)] = X.T

    for i in range(1,len(n_layer)):
        """
        正向传播公式为：
            Z(i) = W(i)*A(i-1)+B(i)  其中A(0)=X.T
            A(i) = Softmax or ReLU[Z(i)]
        """
        cache['Z'+str(i)] = np.dot(parameters['W'+str(i)],cache['A'+str(i-1)])+parameters['b'+str(i)]
        ## 计算输出层的A与Z矩阵(Softmax)
        if i == len(n_layer)-1:
            cache['A'+str(i)] = activation(cache['Z'+str(i)],'Softmax')
        ## 计算输出层之前的A与Z矩阵
        elif i < len(n_layer)-1:
            cache['A' + str(i)] = activation(cache['Z' + str(i)], 'ReLU')

    ## 计算损失函数
    """
        损失函数：COST = -1/m * sum(Y.T * A(last))
    """
    y_ = cache['A'+str(len(n_layer)-1)]
    y  = Y.T
    m  = y.shape[1]
    Loss = np.multiply(y,np.log(y_))
    cost = -1 / m * np.sum(Loss,keepdims=True)

    ## 将Cost函数转换为Float形式
    cost = np.squeeze(cost)

    return cost,cache

##  反向进行一个ReLU->ReLU->ReLU->Softmax的Borward Prop模型
def Backward_pro(Y, n_layer, cache, parameters):
    """

    :param Y:          输入数据集
    :param n_layer:    网络总层数
    :param cache:      含有前向传播系数Z和A的矩阵
    :param parameters: 含有权重系数W和B的参数矩阵
    :return:           含有dW和db的梯度参数矩阵
    """
    grad = {}
    m = Y.shape[0]
    for l in range(len(n_layer) - 1, 0, -1):
        """
            DZ(LAST) = A(LAST)-Y.T          
            DZ(i) = (W(i+1).T * DZ(i+1)) * diff(Z(i))
        """
        # 最后一个，即Softmax的反向传播
        if l == len(n_layer) - 1:
            dZ = cache['A'+str(l)]-Y.T
        # 之前所有的传播 即ReLU的反向传播
        elif l < len(n_layer) - 1:
            dA = np.dot(parameters['W' + str(l + 1)].T, dZ)
            dZ = dA * diff(cache['Z' + str(l)], "ReLU")

        grad['dW' + str(l)] = 1 / m * np.dot(dZ, cache['A' + str(l - 1)].T)
        grad['db' + str(l)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)

    return grad


## 初始化Adam矩阵
def init_adam(parameters):
    v = {}
    s = {}

    for l in range(len(parameters) // 2):
        v['dW' + str(l + 1)] = np.zeros((parameters['W' + str(l + 1)].shape))
        v['db' + str(l + 1)] = np.zeros((parameters['b' + str(l + 1)].shape))
        s['dW' + str(l + 1)] = np.zeros((parameters['W' + str(l + 1)].shape))
        s['db' + str(l + 1)] = np.zeros((parameters['b' + str(l + 1)].shape))

    return v, s

## 进行参数矩阵的更新,并考虑Adam优化模型
def Update_parameters(parameters,grad,ite,beta1,beta2,learning_rate,epsilon,pattern):

    # 选择Adam优化模型
    if pattern == "Adam":
        # adam 矩阵的初始化
        v, s = init_adam(parameters)
        # 矩阵v,s的校正
        v_corrected = {}
        s_corrected = {}
        for l in range(len(parameters) // 2):

            ## 计算动量估计
            v['dW' + str(l+1)] = beta1 * v['dW' + str(l+1)] + (1 - beta1) * grad['dW' + str(l+1)]
            v['db' + str(l+1)] = beta1 * v['db' + str(l+1)] + (1 - beta1) * grad['db' + str(l+1)]

            v_corrected['dW' + str(l+1)] = v['dW' + str(l+1)] / (1 - np.power(beta1, ite))
            v_corrected['db' + str(l+1)] = v['db' + str(l+1)] / (1 - np.power(beta1, ite))

            ## 计算均方根估计
            s['dW' + str(l + 1)] = beta2 * s['dW' + str(l + 1)] + (1 - beta2) * grad['dW' + str(l + 1)]**2
            s['dW' + str(l + 1)] = beta2 * s['db' + str(l + 1)] + (1 - beta2) * grad['db' + str(l + 1)]**2

            s_corrected['dW' + str(l+1)] = s['dW' + str(l+1)] / (1 - np.power(beta2, ite))
            s_corrected['db' + str(l+1)] = s['db' + str(l+1)] / (1 - np.power(beta2, ite))

            ## 更新参数矩阵
            parameters['W' + str(l+1)] = parameters['W' + str(l+1)] - learning_rate * v_corrected['dW' + str(l+1)] / (
                    np.sqrt(s_corrected['dW' + str(l+1)]+ epsilon))
            parameters['b' + str(l+1)] = parameters['b' + str(l+1)] - learning_rate * v_corrected['db' + str(l+1)] / (
                    np.sqrt(s_corrected['db' + str(l+1)]+ epsilon))

    # 选择普通梯度下降
    if pattern == "Grad":
        for l in range(len(parameters) // 2):
            parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grad['dW' + str(l + 1)]
            parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grad['db' + str(l + 1)]

    return parameters

## 进行预测
def Prediction(X,Y, parameters, n_layer):

    cost,cache = Forwardpro(X,Y,n_layer, parameters)

    prd_a = np.argmax(cache['A' + str(len(n_layer) - 1)], axis=0)
    prd_b = np.argmax(Y.T, axis=0)

    pred = np.sum(prd_a == prd_b) / len(prd_a)

    return pred


## 组装函数nn_model
def nn_model(X_train,X_test,n_hiddenlayer,
             beta1,beta2,learning_rate,epsilon,iteration):

    # 训练集简化
    X = X_train["images"]
    Y = X_train["labels"]

    ## 构造并初始化网络的结构、初始化参数矩阵

    n_layer, parameters = init_parameters(X,Y, n_hiddenlayer, 'He')

    Cost = []
    Pred_train = []
    Pred_test  = []
    for ite in range(0,iteration):
        ## PART2 前向传播

        ## 构建一个ReLU->ReLU->ReLU->Softmax
        cost,cache = Forwardpro(X,Y,n_layer,parameters)

        ## PART3 损失函数
        Cost.append(cost)

        ## PART4 反向传播invalid value encountered in true_divide
        grad = Backward_pro(Y,n_layer,cache,parameters)

        ## PART5 参数矩阵更新
        parameters = Update_parameters(parameters,grad,ite+1,beta1,beta2,learning_rate,epsilon,"Adam")
        #learning_rate = 0.95 * learning_rate

        Pred_train.append(Prediction(X,Y, parameters, n_layer))

        Pred_test.append(Prediction(X_test["images"],X_test["labels"], parameters, n_layer))

        print("Epoch[%s/%s],Cost:%f,Train_accuary = %f,Test_accaury = %f" % (ite+1,iteration,Cost[ite],
                                                                             Pred_train[ite],Pred_test[ite]))
        if abs(Cost[ite]-Cost[ite-1])<10**-5 and ite >=1:

            break

    ## 画出损失函数随迭代变化的曲线
    plt.plot(Cost)
    plt.show()
    plt.plot(Pred_train)
    plt.show()
    plt.plot(Pred_test)
    plt.show()

    return parameters,Cost,Pred_train,Pred_test


## 组装mini_batch函数nn_model
def mini_batch_nn_model(X_train, X_test,mini_batch_size,
                        n_hiddenlayer,beta1, beta2, learning_rate, epsilon, iteration):

    # 训练集简化
    X = X_train["images"]
    Y = X_train["labels"]

    ## 构造并初始化网络的结构、初始化参数矩阵

    n_layer, parameters = init_parameters(X,Y, n_hiddenlayer, 'He')

    # 一些矩阵的
    Cost = []
    Pred_train = []
    Pred_test = []
    seed = 10
    t = 0
    for ite in range(0, iteration):

        seed = seed + 1

        ## mini_batch化
        mini_batches = random_mini_batches(X_train,mini_batch_size,seed)



        for minibatch in mini_batches:

            # 选择一个minibatch
            (minibatch_X, minibatch_Y) = minibatch

            ## PART2 前向传播

            ## 构建一个ReLU->ReLU->ReLU->Softmax
            cost, cache = Forwardpro(minibatch_X, minibatch_Y, n_layer, parameters)

            ## PART3 损失函数
            Cost.append(cost)

            ## PART4 反向传播
            grad = Backward_pro(minibatch_Y, n_layer, cache, parameters)

            ## PART5 参数矩阵更新
            t = t + 1

            parameters = Update_parameters(parameters, grad, t, beta1, beta2, learning_rate, epsilon, "Adam")


        # learning_rate = 0.95 * learning_rate

        Pred_train.append(Prediction(X, Y, parameters, n_layer))

        Pred_test.append(Prediction(X_test["images"], X_test["labels"], parameters, n_layer))
        itt = ite + 1
        print("Epoch[%s/%s],Cost:%f,Train_accuary = %f,Test_accaury = %f" % (round(itt),round(iteration),Cost[ite],
                                                                             Pred_train[ite],Pred_test[ite]))

        if abs(Cost[ite] - Cost[ite - 1]) < 10 ** -5 and ite >= 1:
            break

    ## 画出损失函数随迭代变化的曲线
    plt.plot(Cost)
    plt.show()
    plt.plot(Pred_train)
    plt.show()
    plt.plot(Pred_test)
    plt.show()

    return parameters, Cost, Pred_train, Pred_test