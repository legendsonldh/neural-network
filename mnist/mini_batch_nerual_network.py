#-*-coding:utf-8-*-

# 每次重新运行清空所有变量
# reset
import matplotlib.pyplot as plt
import numpy as np

from pre_func import input_data
from pre_func import norm_data
from pre_func import one_dim_PCA
from nn_func import mini_batch_nn_model

## 超参数
mini_batch_size,n_hiddenlayer,beta1,beta2,learning_rate,epsilon,iteration = 128,[800,400],\
                                                                            0.9,0.99,0.001,10**-8,100

## PART1 导入数据并预处理

## 导入数据
X_train, X_test = input_data()

## 归一化
X_train['images'] = norm_data(X_train['images'], "avg1")
X_test['images'] = norm_data(X_test['images'], "avg1")


## 对每一个mini_batch部分进行PCA

## 归一化
X_train['images'] = norm_data(X_train['images'], "avg1")
X_test['images'] = norm_data(X_test['images'], "avg1")


## 1D_PCA
X_train['images'],main_mat = one_dim_PCA(X_train['images'],0.999,"train",_,0)   # 2 代表重构的维度，0就代表正常操作

X_test['images'],_ = one_dim_PCA(X_test['images'],0.90,"test",main_mat,0)

"""
## 输出PCA 投影到2维的重构
X,main_mat = one_dim_PCA(X_train['images'],0.999,"train",_,2)   # 2 代表重构的维度，0就代表正常操作

plt.figure(figsize=(12, 8))
plt.tick_params(labelsize=18)
plt.scatter(X[:5000, 0], X[:5000, 1],s = 20 ,c=np.argmax(X_train['labels'][:5000,:], 1),cmap="rainbow")
plt.colorbar()
plt.grid(linestyle = "--")
plt.savefig("pca.svg",format = 'svg')
plt.show()
"""

## 模型
parameters,Cost,Pred_train,Pred_test = mini_batch_nn_model(X_train,X_test,mini_batch_size,n_hiddenlayer,
                                                           beta1,beta2,learning_rate,epsilon,iteration)

"""

np.savetxt('Cost_pca',Cost)

np.savetxt('Pred_pca_test',Pred_test)

np.savetxt('Pred_pca_train',Pred_train)

"""