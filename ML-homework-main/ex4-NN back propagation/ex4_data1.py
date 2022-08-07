"""
    反向传播计算偏导，要将参数向量化，并使用梯度检测函数
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.metrics import classification_report
import sys

def load_data(path):
    """处理数据"""
    data = loadmat(path)
    X = data['X']
    Y = data['y']
    Y = Y.reshape(Y.shape[0])
    return X, vector_y(Y), Y

def load_weight(path):
    """获取神经网络参数"""
    data = loadmat(path)
    return data['Theta1'], data['Theta2']

def plot_100_img(X):
    """随机画100个灰度图"""
    sz = int(np.sqrt(X.shape[1]))
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)
    sample_img = X[sample_idx]
    fig, axs = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            axs[i, j].matshow(sample_img[10 * i + j].reshape(sz, sz).T, cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()

def vector_y(Y):
    """将结果向量化"""
    res = []
    for i in Y:
        temp = np.zeros(10)
        temp[i-1] = 1
        res.append(temp)
    return np.array(res)

def serialize(a, b):
    """将参数向量化"""
    return np.concatenate((np.ravel(a), np.ravel(b)))

def deserialize(theta):
    """解向量化"""
    return theta[:25*401].reshape(25, 401), theta[25*401:].reshape(10, 26)

def sigmoid(z):
    """sigmoid函数"""
    return (1 / (1 + np.exp(-z)))

def feed_forward(theta, X):
    """前向传播"""
    t1, t2 = deserialize(theta) # t1 (25, 401), t2 (10, 26)
    a1 = X # 5000 * 401
    z2 = a1 @ t1.T # 5000 * 25
    a2 = np.insert(sigmoid(z2), 0, 1, axis=1) # 5000 * 26
    z3 = a2 @ t2.T # 5000 * 10
    res = sigmoid(z3)
    return a1, a2, res

def cost(theta, X, Y):
    """代价函数"""
    h = feed_forward(theta, X)[-1]
    temp = -1 * (Y * np.log(h) + (1-Y) * np.log(1-h))
    return temp.sum() / Y.shape[0]

def regularized_cost(theta, X, Y, l=1):
    """正则化后的代价函数"""
    t1, t2 = deserialize(theta)
    reg1 = np.power(t1[:, 1:], 2).sum()
    reg2 = np.power(t2[:, 1:], 2).sum()
    return cost(theta, X, Y) + (l / (2*X.shape[0])) * (reg1 + reg2)

def gradient(theta, X, Y):
    """计算梯度的函数"""
    t1, t2 = deserialize(theta)
    m = X.shape[0]

    delta1 = np.zeros(t1.shape)  # 25*401
    delta2 = np.zeros(t2.shape)  # 10*26
    a1, a2, a3 = feed_forward(theta, X)

    for i in range(m):
        a1i = a1[i] # 1*401
        a2i = a2[i] # 1*26

        hi = a3[i] # 1*10
        yi = Y[i] # 1*10
        d3i = hi - yi
        d2i = t2.T @ d3i * (a2i * (1-a2i))

        delta2 += np.matrix(d3i).T @ np.matrix(a2i)
        delta1 += np.matrix(d2i[1:]).T @ np.matrix(a1i) # 切片
    return serialize(delta1, delta2)

def regularized_gradient(theta, X, Y, l=1):
    """正则化后的梯度"""
    m = X.shape[0]
    delta1, delta2 = deserialize(gradient(theta, X, Y))
    delta1 /= m
    delta2 /= m

    t1, t2 = deserialize(theta)
    t1[:, 0] = 0
    t2[:, 0] = 0
    
    delta1 += l / m * t1
    delta2 += l / m * t2
    
    return serialize(delta1, delta2)

def expand_array(arr):
    """扩充向量"""
    return np.array(np.matrix(np.ones(arr.shape[0])).T @ np.matrix(arr))

def gradient_checking(theta, X, Y, epsilon, regularized=False):
    """梯度检测函数"""
    m = len(theta)
    def a_numeric_grad(plus, minus, regularized=False):
        if regularized:
            return (regularized_cost(plus, X, Y) - regularized_cost(minus, X, Y)) / (epsilon*2)
        else:
            return (cost(plus, X, Y) - cost(minus, X, Y)) / (epsilon*2)
    
    theta_matrix = expand_array(theta)
    epsilon_matrix = np.identity(m) * epsilon # identity单位矩阵
    plus_matrix = theta_matrix + epsilon_matrix
    minus_matrix = theta_matrix - epsilon_matrix
    
    approx_grad = np.array([a_numeric_grad(plus_matrix[i], minus_matrix[i], regularized) 
                            for i in range(m)])
    analytic_grad = regularized_gradient(theta, X, Y) if regularized else gradient(theta, X, Y)
    diff = np.linalg.norm(approx_grad - analytic_grad) / np.linalg.norm(approx_grad + analytic_grad)
    
    print('If your backpropagation implementation is correct,\nthe relative difference will be smaller than 10e-8 (assume epsilon=0.0001).\nRelative Difference: {}\n'.format(diff))

def random_init(size):
    """随机生成参数"""
    return np.random.uniform(-0.12, 0.12, size)

def nn_training(theta, X, Y, l=1, simple=False):
    """训练参数"""
    init_theta = random_init(len(theta))
    if simple:
        res = opt.minimize(fun=regularized_cost, x0 = init_theta,
                        args=(X, Y, 1), method='TNC',
                        jac=regularized_gradient,
                        options={'maxiter': 400})
        return res.x
    else:
        res = opt.fmin_tnc(func=regularized_cost, x0=init_theta, args=(X, Y, l), fprime=regularized_gradient)
        return res[0]

def gradient_descent(theta, X, Y, alpha, iters, l=1):
    """梯度下降函数"""
    init_theta = random_init(len(theta))
    costs = np.zeros(iters)
    for i in range(iters):
        init_theta -= alpha * regularized_gradient(init_theta, X, Y, l)
        costs[i] = regularized_cost(init_theta, X, Y)
    return init_theta, costs

def main():
    """主函数"""
    data_path = 'ex4-NN back propagation/ex4data1.mat'
    weight_path = 'ex4-NN back propagation/ex4weights.mat'
    X, Y, y_true = load_data(data_path)
    # plot_100_img(X)
    X = np.insert(X, 0, 1, axis=1)
    t1, t2 = load_weight(weight_path)
    theta = serialize(t1, t2)
    l = 1 # 惩罚度
    # gradient_checking(theta, X, Y, epsilon=0.0001, regularized=True) # 梯度检测
    if len(sys.argv)==2:
        if int(sys.argv[1])==1:
            """梯度下降"""
            alpha, iters = 1, 1000
            final_theta, costs = gradient_descent(theta, X, Y, alpha, iters, l)
            h = feed_forward(final_theta, X)[-1]
            y_pred = np.argmax(h, axis=1)+1
            print(classification_report(y_true, y_pred))
            accuracy=np.mean(y_pred==y_true)
            print(f'accuracy = {accuracy}')
            plt.figure(figsize=(6,4))
            plt.plot(range(iters), costs)
            plt.show()
        elif int(sys.argv[1])==2 or int(sys.argv[1])==3:
            """高级优化算法"""
            if int(sys.argv[1])==2:
                simple = True
            else:
                simple = False
            res = nn_training(theta, X, Y, l, simple)
            h = feed_forward(res, X)[-1]
            y_pred = np.argmax(h, axis=1)+1
            print(classification_report(y_true, y_pred))
            accuracy=np.mean(y_pred==y_true)
            print(f'accuracy = {accuracy}')
        else:
            print("请输入正确的参数：\n\t1.\t梯度下降算法\t2.\t高级优化算法（简略）\t3.\t高级优化算法")
    else:
        print("请输入正确的参数：\n\t1.\t梯度下降算法\t2.\t高级优化算法（简略）\t3.\t高级优化算法")

if __name__ == '__main__':
    main()
