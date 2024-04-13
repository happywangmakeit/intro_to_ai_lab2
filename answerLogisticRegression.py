import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
lr = 0.4  # 学习率
wd = 0.005  # l2正则化项系数


def predict(X, weight, bias):
    """
    使用输入的weight和bias，预测样本X是否为数字0。
    @param X: (n, d) 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @return: (n,) 线性模型的输出，即wx+b
    """
    # TODO: YOUR CODE HERE
    pred_y=X@weight+bias
    return pred_y

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: (n, d) 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @param Y: (n,) 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: (n,) 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: (1,) 由交叉熵损失函数计算得到
        weight: (d,) 更新后的weight参数
        bias: (1,) 更新后的bias参数
    """
    # TODO: YOUR CODE HERE
    pred_y=predict(X,weight,bias)
    # haty=sigmoid(pred_y)
    # exp_y=np.exp(pred_y)
    # exp_sum=np.sum(exp_y,axis=1)
    # score=exp_y/exp_sum[:,np.newaxis]
    exp_para=(sigmoid(Y*pred_y)-1)*Y
    grad_w=np.average(X*exp_para[:,np.newaxis],axis=0)
    grad_b=np.average(exp_para)

    loss=np.average(np.log(1+np.exp(-Y*pred_y))+wd*np.sum(weight*weight))
    new_w=weight-lr*grad_w+2*weight*wd
    new_b=bias-lr*grad_b
    
    return pred_y,loss,new_w,new_b


