from numpy.random import rand
import mnist
from answerTree import *
import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
num_tree = 16     # 树的数量
ratio_data = 0.8   # 采样的数据比例
ratio_feat = 0.75 # 采样的特征比例
hyperparams = {
    "depth":5, 
    "purity_bound":0.2,
    "gainfunc": gainratio
    } # 每颗树的超参数


def buildtrees(X, Y):
    """
    构建随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @return: List of DecisionTrees, 随机森林
    """
    # TODO: YOUR CODE HERE
    # 提示：整体流程包括样本扰动、属性扰动和预测输出
    trees=[]
    # tree=buildTree(X,Y,list(np.int64(np.arange(X.shape[1]*ratio_feat))),**hyperparams)
    # trees.append(tree)
    # return trees
    for i in range(num_tree):
        rand_d=np.random.choice(X.shape[0],int(X.shape[0]*ratio_data),replace=False)
        sort_d=np.sort(rand_d)
        rand_f=np.random.choice(X.shape[1],int(X.shape[1]*ratio_feat),replace=False)
        sort_f=np.sort(rand_f)
        sub_data=X[sort_d,:]
        sub_Y=Y[sort_d]
        unused=list(sort_f)
        tree=buildTree(sub_data,sub_Y,unused,
                       **hyperparams)
        trees.append(tree)

    return trees
    raise NotImplementedError    

def infertrees(trees, X):
    """
    随机森林预测
    @param trees: 随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @return: n, 预测的label
    """
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]