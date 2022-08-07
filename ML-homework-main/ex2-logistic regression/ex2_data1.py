import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sys

def sigmoid(z):
    """sigmoid函数"""
    return 1/(1 + np.exp(-z))

def cost_fun(theta, X, Y):
    """代价函数，传入数组是因为数组才能实现一个函数改变数组脸面所有元素，而且数组具有矩阵的运算性质"""
    first = np.multiply(Y, np.log(sigmoid(X @ theta.T) + 1e-6))
    second = np.multiply((1 - Y), np.log(1 - sigmoid(X @ theta.T) + 1e-6))
    return -1 * np.mean(first + second)

def gradient_descent(X, Y, alpha, theta, iters):
    """梯度下降函数"""
    costs = np.zeros(iters)
    for i in range(iters):
        theta = theta - alpha * gradient(theta, X, Y)
        costs[i] = cost_fun(theta, X, Y)
    return theta, costs

def gradient(theta, X, Y):
    """梯度函数，计算梯度步长"""
    return (1/len(X)) * X.T @ (sigmoid(X @ theta.T) - Y)

def data_pre(flag=0):
    """数据预处理"""
    path = 'ex2-logistic regression/ex2data1.txt'
    data = pd.read_csv(path, header=None, names=['Exam1', 'Exam2', 'Admitted'])
    if flag:
        data = standardlize(data)
    data.insert(0, 'ones', 1)
    X = data.iloc[:, 0:-1]
    Y = data.iloc[:, -1]
    l_min = X.Exam1.min()
    l_max = X.Exam1.max()
    X = X.values
    Y = Y.values
    positived = data.loc[data['Admitted'] == 1]
    negatived = data.loc[data['Admitted'] == 0]
    return X, Y, l_min, l_max, positived, negatived

def standardlize(data):
    """数据标准化"""
    a = data.iloc[:, -1]
    data = (data - data.mean()) / data.std()
    data['Admitted'] = a
    return data

def visualize(theta, l_min, l_max, positived, negatived):
    """实现可视化"""
    x = np.linspace(l_min, l_max, 100)
    f = (- theta[0] - theta[1] * x) / theta[2]
    plt.figure(figsize=(6, 4))
    plt.plot(x, f, 'y', label='prediction')
    plt.scatter(positived['Exam1'], positived['Exam2'], s=30, c='b', marker='o', label='Admitted')
    plt.scatter(negatived['Exam1'], negatived['Exam2'], s=30, c='r', marker='x', label='Not Admitted')
    plt.legend()
    plt.xlabel("Exam1 Score")
    plt.ylabel("Exam2 Score")
    plt.show()

def predict(theta, X):
    """预测函数"""
    probability = sigmoid(X @ theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

def cost_trend(iters, costs):
    """绘制梯度下降算法中代价函数变化"""
    plt.figure(figsize=(6, 4))
    plt.plot(range(iters), costs, 'r')
    plt.show()

def gradient_process(X, Y):
    """执行梯度下降"""
    alpha = 1
    iters = 10000
    theta = np.zeros(3)
    theta, costs = gradient_descent(X, Y, alpha, theta, iters)
    return theta, costs, iters

def predict_fun(theta, X, Y):
    """预测函数，输出拟合结果以及代价"""
    predictions = predict(theta, X)
    correct = [1 if a^b == 0 else 0 for (a, b) in zip(predictions, Y)]
    accuracy = np.mean(correct)
    print(f"accuracy = {accuracy*100}%")
    print(f"cost = {cost_fun(theta, X, Y)}")

def advanced_fun(X, Y):
    """执行高级算法"""
    theta = np.zeros(3)
    result = opt.fmin_tnc(func=cost_fun, x0=theta, fprime=gradient, args=(X, Y))
    theta = result[0]
    return theta

def main():
    if len(sys.argv) == 2:
        if int(sys.argv[1]) == 1:
            """执行梯度下降算法"""
            X, Y, l_min, l_max, positived, negatived = data_pre(1)
            theta, costs, iters = gradient_process(X, Y)
            predict_fun(theta, X, Y)
            visualize(theta, l_min, l_max, positived, negatived)
            cost_trend(iters, costs)
        elif int(sys.argv[1]) == 2:
            """高级算法"""
            X, Y, l_min, l_max, positived, negatived = data_pre()
            theta = advanced_fun(X, Y)
            predict_fun(theta, X, Y)
            visualize(theta, l_min, l_max, positived, negatived)
        else:
            print("请输入正确的参数：\n\t1. 梯度下降算法\t\t\t\t2. 高级优化算法")
    else:
        print("请输入正确的参数：\n\t1. 梯度下降算法\t\t\t\t2. 高级优化算法")
        
if __name__ == '__main__':
    main()