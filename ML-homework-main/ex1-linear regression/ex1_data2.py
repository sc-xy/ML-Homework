import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

def cost_fun(X, Y, theta):
    """代价函数"""
    inners = np.power((X * theta.T) - Y, 2)
    return np.sum(inners) / ( 2 * len(X))

def graident_descent(X, Y, alpha, theta, iters):
    """梯度下降函数"""
    temp = np.matrix(np.zeros(theta.shape))
    parms = int(theta.shape[1])
    costs = np.zeros(iters)
    for i in range(iters):
        errors = X * theta.T - Y

        for j in range(parms):
            term = np.multiply(errors, X[:, j])
            temp[0, j] = temp[0, j] - alpha / len(Y) * np.sum(term)

        theta = temp
        costs[i] = cost_fun(X, Y, theta)
    
    return theta, costs

def predict_fun_g(theta,mu, std,  size, bednums):
    """房价预测函数（梯度下降）"""
    n_size = (size - mu[0]) / std[0]
    n_bed = (bednums - mu[1]) / std[1]
    return theta[0, 0] + theta[0, 1] * n_size + theta[0, 2] * n_bed

def predict_fun_n(theta, size, bednums):
    """房价预测函数（正规方程）"""
    return (theta[0, 0] + theta[0, 1] * size + theta[0, 2] * bednums)

def normalnize(X, Y):
    """正规化方程"""
    theta = np.linalg.inv(X.T*X)*X.T*Y
    theta = theta.reshape(1, -1)
    return theta

def data_pre_g():
    """梯度下降（特征值标准化）数据预处理"""
    # 读取数据
    path = "ex1-linear regression/ex1data2.txt"
    data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

    # 处理数据
    cols = data.shape[1] # 取得列数
    X = data.iloc[:, 0:cols-1] 
    Y = data.iloc[:, cols-1:cols]
    mu = X.mean()
    std = X.std()
    X = (X - X.mean()) / X.std() # 标准化X
    X.insert(0, 'ones', 1)
    X = np.matrix(X.values)
    Y = np.matrix(Y.values)
    return X, Y, mu, std

def data_pre_n():
    """正规方程数据预处理"""
    # 读取数据
    path = "ex1-linear regression/ex1data2.txt"
    data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

    # 处理数据
    cols = data.shape[1] # 取得列数
    X = data.iloc[:, 0:cols-1] 
    Y = data.iloc[:, cols-1:cols]
    X.insert(0, 'ones', 1)
    X = np.matrix(X.values)
    Y = np.matrix(Y.values)
    return X, Y

def gradient_pre(X, Y):
    """执行梯度下降"""
    # 设置初值进行梯度下降
    theta = np.matrix(np.array([0, 0, 0])) 
    alpha = 0.3 # 学习速率
    iters = 10000 # 循环次数
    theta, costs = graident_descent(X, Y, alpha, theta, iters)
    return theta, costs, iters

def cost_trend(costs, iters):
    """画出cost图像"""
    plt.figure(figsize=(6, 4))
    plt.title("Error vs Training Epoch")
    plt.xlabel('Iterations', size=14)
    plt.ylabel('Cost', size=14)
    plt.plot(range(iters), costs, c='red')
    plt.show()

def main():
    if len(sys.argv) == 4:
        if int(sys.argv[1]) == 1:
            X, Y, mu, std = data_pre_g()
            theta, costs, iters = gradient_pre(X, Y)
            size = float(sys.argv[2])
            bednums = float(sys.argv[3])
            print(predict_fun_g(theta, mu, std, size, bednums))
            cost_trend(costs, iters)
        elif int(sys.argv[1]) == 2:
            X, Y = data_pre_n()
            theta = normalnize(X, Y)
            size = float(sys.argv[2])
            bednums = float(sys.argv[3])
            print(predict_fun_n(theta, size, bednums))
        else:
            print("请输入三个参数:\n\t1 size bednums \t\t 梯度下降算法; \t\t2 size bednums  \t\t正规方程算法")
    else:
        print("请输入三个参数:\n\t1 size bednums \t\t 梯度下降算法; \t\t2 size bednums  \t\t正规方程算法")

if __name__ == '__main__':
    main()