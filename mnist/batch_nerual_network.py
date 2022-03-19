#-*-coding:utf-8-*-

# 每次重新运行清空所有变量
# reset
import matplotlib.pyplot as plt
import pre_func
import nn_func

from pre_func import input_data

from pre_func import norm_data
from pre_func import one_dim_PCA
from pre_func import one_dim_kernel_PCA
from nn_func import nn_model
from nn_func import mini_batch_nn_model

## 超参数
# n_hiddenlayer,beta1,beta2,learning_rate,epsilon,iteration = [500,300],0.9,0.99,0.001,10**-8,200
## 超参数
mini_batch_size,n_hiddenlayer,beta1,beta2,learning_rate,epsilon,iteration = 128,[800,400],\
                                                                            0.9,0.99,0.001,10**-8,20
## PART1 导入数据并预处理

## 导入数据
X_train, X_test = input_data()

## 归一化
X_train['images'] = norm_data(X_train['images'], "avg1")
X_test['images'] = norm_data(X_test['images'], "avg1")

# Kernel-PCA
dim = 5000

X_train['images'],X_cache,X_vetcor = one_dim_kernel_PCA(X_train['images'][:dim,:],_,15,0.9,"train",_,0)
X_test['images'],_,_ = one_dim_kernel_PCA(X_test['images'][:dim,:],X_cache,15,0.9,"test",X_vetcor,0)

X_train['labels'] = X_train['labels'][:dim,:]
X_test['labels'] = X_test['labels'][:dim,:]

## 模型
parameters,Cost,Pred_train,Pred_test = mini_batch_nn_model(X_train,X_test,mini_batch_size,n_hiddenlayer,
                                                           beta1,beta2,learning_rate,epsilon,iteration)


## 模型
# parameters,Cost,Pred_train,Pred_test = nn_model(X_train,X_test,n_hiddenlayer,beta1,beta2,learning_rate,epsilon,iteration)


## 1D_PCA
# X_train['images'],main_mat = one_dim_PCA(X_train['images'],0.999,"train",_,0)
# X_test['images'],_ = one_dim_PCA(X_test['images'],0.90,"test",main_mat,0)

## 必要时才打开
"""

np.savetxt('Cost_pca',Cost)

np.savetxt('Pred_pca_test',Pred_test)

np.savetxt('Pred_pca_train',Pred_train)

"""
"""
## 输出PCA 投影到1维的重构
X_train['images'],X_cache,X_vetcor = one_dim_kernel_PCA(X_train['images'][:dim,:],_,15,0.9,"train",_,1)

plt.figure(figsize=(12, 8))
plt.tick_params(labelsize=18)
plt.scatter(X_train['images'][:dim,0],np.zeros((dim,1)),s = 20 ,c=np.argmax(X_train['labels'][:dim,:], 1),cmap="rainbow")
plt.colorbar()
plt.grid(linestyle = "--")
plt.xlim(-0.04,0)
plt.savefig("kernel_pca.svg",format = 'svg')
plt.show()
"""