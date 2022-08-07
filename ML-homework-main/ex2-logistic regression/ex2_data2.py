"""
    绘制非线性（多项式）决策边界时，可以用散点图近似表达：
    注意：因为精度限制，并不能找出所有f(x, y) = 0 的点，只能用近似0的部分设置边界，例如本程序中的threshlod = 2 * 10**-3 
"""

import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

def feature_mapping(x1, x2, degree):
    """特征值映射函数"""
    data = pd.DataFrame()
    for i in range(1, degree+1):
        for j in range(i+1):
            data['F'+str(i-j)+str(j)] = np.power(x1, i-j) * np.power(x2, j)
    data.insert(0, 'ones', 1)
    return data

def sigmoid(z):
    """sigmoid函数"""
    return 1 / (1 + np.exp(-z))

def cost(theta, X, Y):
    """代价函数"""
    first = Y * np.log(sigmoid(X @ theta.T))
    second = (1 - Y) * np.log(1- sigmoid(X @ theta.T))
    return -1 * np.mean(first + second)

def regularized_cost(theta, X, Y, l=1):
    """正则化后的的代价函数"""
    theta_ln = theta[1:]
    regularize_term = l / (2 * len(X)) * np.power(theta_ln, 2).sum()
    return cost(theta, X, Y) +regularize_term
    
def gradient(theta, X, Y):
    """计算梯度步长"""
    return (1 / len(X) * X.T @ (sigmoid(X @ theta.T) - Y))

def regularized_gradient(theta, X, Y, l=1): 
    """计算正则化梯度步长"""
    theta_ln = theta[1:]
    regularized_theta = l / len(X) * theta_ln
    regularized_term = np.concatenate([np.array([0]), regularized_theta])
    return gradient(theta, X, Y) + regularized_term 

def gradient_descent(X, Y, alpha, theta, iters, l=1):
    """梯度下降函数"""
    costs = np.zeros(iters)
    for i in range(iters):
        theta -= alpha * regularized_gradient(theta, X, Y, l)
        costs[i] = regularized_cost(theta, X, Y, l)
    return theta, costs
    
def predict_fun(theta, X, Y):
    """预测正确率"""
    temp = sigmoid(X @ theta.T)
    temp = [ 1 if i >= 0.5 else 0 for i in temp]
    correct = [ 1 if a^b == 0 else 0 for (a,b) in zip(temp, Y)]
    accuracy = np.mean(correct)
    print(f'accuracy: {accuracy*100} %')
    print(f'cost = {regularized_cost(theta, X, Y)}')

def find_decision_doundary(density, degree, theta, threshold):
    """找出决策边界"""
    t1 = np.linspace(-1, 1.2, density)
    t2 = np.linspace(-1, 1.2, density)
    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    mapped_cord = feature_mapping(x_cord, y_cord, degree)
    pred = mapped_cord.values @ theta.T
    decision = mapped_cord[ np.abs(pred) <= threshold]
    return decision.F10, decision.F01

def data_pre():
    """数据预处理"""
    path = 'ex2-logistic regression/ex2data2.txt'
    data = pd.read_csv(path, header=None, names=['Test1', 'Test2', 'Accepted'])
    positived = data.loc[data['Accepted'] == 1]
    negatived = data.loc[data['Accepted'] == 0]
    Y = data['Accepted'].values
    x1 = data['Test1']
    x2 = data['Test2']
    degree = 6
    data = feature_mapping(x1, x2, degree)
    theta = np.zeros(data.shape[1])
    X = data.values
    return theta, X, Y, positived, negatived

def visalize(positived, negatived, x, y):
    """可视化决策边界"""
    plt.figure(figsize=(6, 4))
    plt.scatter(positived['Test1'], positived['Test2'], s=10, c='b', marker='o', label='Accepted')
    plt.scatter(negatived['Test1'], negatived['Test2'], s=10, c='r', marker='x', label='Rejected')
    plt.scatter(x, y, s=10, c='g', marker='.', label='Prediction')
    plt.legend()
    plt.xlabel('Test1 Score')
    plt.ylabel("Test2 Score")
    plt.show()

def cost_trend(iters, costs):
    """绘制代价变化趋势"""
    plt.figure(figsize=(6,4))
    plt.plot(range(iters), costs, c='r')
    plt.show()

def main():
    if len(sys.argv) == 2:
        l = 1 # 惩罚力度
        if int(sys.argv[1]) == 1:
            """执行梯度下降算法"""
            theta, X, Y, positived, negatived = data_pre()
            alpha, iters = 1.,1000 # 学习速率，迭代次数
            theta, costs = gradient_descent(X, Y, alpha, theta, iters, l)
            x, y = find_decision_doundary(1000, 6, theta, 2*10**-3)
            predict_fun(theta, X, Y)
            visalize(positived, negatived, x, y)
            cost_trend(iters, costs)
        elif int(sys.argv[1]) == 2:
            """执行高级算法"""
            theta, X, Y, positived, negatived = data_pre()
            result = opt.fmin_tnc(func=regularized_cost, x0=theta, fprime=regularized_gradient, args=(X, Y, l))
            theta = result[0]
            x, y = find_decision_doundary(1000, 6, theta, 2*10**-3)
            predict_fun(theta, X, Y)
            visalize(positived, negatived, x, y)
        else:
            print("请输入正确的参数：\n\t1.\t执行梯度下降算法\t\t\t2.\t执行高级算法")
    else:
        print("请输入正确的参数：\n\t1.\t执行梯度下降算法\t\t\t2.\t执行高级算法")

if __name__ == '__main__':
    main()