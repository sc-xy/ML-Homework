import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

def cost_fun(X, Y, theta):
    """代价函数"""
    inner = np.power((X * theta.T) - Y, 2)
    return np.sum(inner) / (2*len(Y))

def gradient_descent(X, Y, alpha, theta, times):
    """梯度下降函数"""
    temp = np.matrix(np.zeros(theta.shape)) # 参数中间变量 -- 矩阵
    parameters = int(theta.shape[1]) # 参数个数
    cost = np.zeros(times) # 创建一维0数组
    
    for i in range(times):
        # 整个函数
        error = X * theta.T - Y # h(x) - y 的矩阵
        
        for j in range(parameters):
            term = np.multiply(error, X[:, j]) # 对应偏导数的矩阵
            temp[0, j] = temp[0, j] - alpha / len(Y) * np.sum(term) # 中间变量存储更改后的参数值
        
        theta = temp # 更新参数
        cost[i] = cost_fun(X, Y, theta) # 记录对应代价
    
    return theta, cost

def normalnize(X, Y):
    """正规方程"""
    theta = np.linalg.inv(X.T@X)@X.T@Y
    theta = theta.reshape(1, -1)
    return theta

def data_pre():
    """数据预处理"""
    # 读取数据
    path = 'ex1-linear regression/ex1data1.txt'
    data = pd.read_csv(path, header=None, names=['Population', 'Profit']) # header=None表示没有标题行，避免第一行数据被当作标题行

    # 处理数据
    data.insert(0, 'one', 1) # 增加一列
    col = data.shape[1] # 取得列数
    X = data.iloc[:, 0: col-1] # 取得对应切片 -- m x 2的矩阵
    Y = data.iloc[:, col-1:] # m x 1的矩阵

    # 取得对应矩阵
    X = np.matrix(X.values)
    Y = np.matrix(Y.values)
    return X, Y, data

def simulate_predict(data, theta):
    """预测画图"""
    # 画出拟合图像
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    f = theta[0, 0] + theta[0, 1]*x
    plt.figure(figsize=(6, 4))
    plt.xlabel('Population', size=14)
    plt.ylabel('Profit', size=14)
    l1 = plt.plot(x, f, label='Prediction', c='red')
    l2 = plt.scatter(data.Population, data.Profit, s=10)
    plt.legend(loc='best')
    plt.title('Predicted Profit vs Population Size')
    # print(theta, costs[-1])
    print(theta)
    plt.show()

def cost_trend(costs, times):
    """画出cost的走势"""
    plt.figure(figsize=(6, 4))
    plt.xlabel('Iterations', size=14)
    plt.ylabel('Cost', size=14)
    plt.title('Error vs Training Epoch')
    plt.plot(np.arange(times), costs, 'r')
    plt.show()

def gradient_pre(X, Y):
    """执行梯度下降"""
    # 梯度下降
    theta = np.matrix(np.array([0, 0])) # 1 x 2 矩阵
    alpha = 0.02
    times = 10000
    theta, costs = gradient_descent(X, Y, alpha, theta, times)
    return theta, costs, times

def main():
    """主函数"""
    X , Y , data = data_pre()
    if len(sys.argv) == 2:
        if int(sys.argv[1]) == 1:
            """执行梯度下降"""
            theta, costs, times = gradient_pre(X, Y)
            simulate_predict(data, theta)
            cost_trend(costs, times)
            print(costs[-1])
        elif int(sys.argv[1]) == 2:
            """执行正规方程"""
            theta = normalnize(X, Y)
            simulate_predict(data, theta)
        else:
            print("请输入一个参数:\n\t1\t\t 梯度下降算法; \t\t2\t\t正规方程算法")
    else:
        print("请输入一个参数:\n\t1\t\t 梯度下降算法; \t\t2\t\t正规方程算法")

if __name__ == '__main__':
    main()