import numpy as np
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.metrics import classification_report
import sys

def load_data(path):
    """逻辑回归算法的数据"""
    data = loadmat(path)
    raw_x = data['X']
    raw_y = data['y']
    raw_y = raw_y.reshape(raw_y.shape[0])
    X = np.insert(raw_x, 0, 1, axis=1)
    Y = []
    for i in range(1, 11):
        Y.append([ 1 if k==i else 0 for k in raw_y])
    # Y = np.array([Y[-1]] + Y[:-1])
    return X, np.array(Y), raw_y
   
def sigmoid(z):
    """sigmoid函数"""
    return 1 / (1 + np.exp(-z))

def cost(theta, X, Y):
    """代价函数"""
    first = Y * np.log(sigmoid(X @ theta.T))
    second = (1 - Y) * np.log(sigmoid(1 - (X @ theta.T)))
    return -np.mean(first + second)

def regularized_cost(theta, X, Y, l):
    """正则化后代价函数"""
    reg = l / (2 * len(X)) * (theta[1:] **2).sum()
    return cost(theta, X, Y) + reg

def gradient(theta, X, Y):
    """计算梯度的函数"""
    return 1 / len(X) * X.T @ (sigmoid(X @ theta.T) - Y)

def regularize_gradient(theta, X, Y, l):
    """计算正则化后梯度的函数"""
    reg = theta / len(X)
    reg[0] = 0
    return gradient(theta, X, Y) + reg

def regression_pre(X, Y, l=1):
    """执行逻辑回归"""
    theta = np.zeros(X.shape[1])
    res = opt.fmin_tnc(func=regularized_cost, x0=theta, args=(X, Y, l), fprime=regularize_gradient)
    return res[0]

def predict_fun(theta, X):
    """预测函数"""
    prob = sigmoid(X @ theta.T)
    return [ 1 if i >= 0.5 else 0 for i in prob]

def gradient_descent(X, Y, alpha, theta, iters, l=1):
    """梯度下降函数"""
    for i in range(iters):
        theta -= alpha * regularize_gradient(theta, X, Y, l)
    return theta

def load_weight(path):
    """"获取神经网络算法的参数"""
    data = loadmat(path)
    return data['Theta1'], data['Theta2']

def main():
    """主函数"""
    if len(sys.argv) == 2:
        path = 'ex3-neural network/ex3data1.mat'
        X, Y, y_true = load_data(path)
        if int(sys.argv[1]) == 1:
            """梯度下降算法"""
            theta_k = np.array([gradient_descent(X, Y[k], 1, np.zeros(X.shape[1]), 3000) for k in range(10)])
            pred_matrix = sigmoid(X @ theta_k.T)
            y_pred = np.argmax(pred_matrix, axis=1)
            y_pred = np.array([i+1 for i in y_pred])
            print(classification_report(y_true, y_pred))
        elif int(sys.argv[1]) == 2:
            """高级优化算法"""
            theta_k = np.array([regression_pre(X, Y[k]) for k in range(10)])
            pred_matrix = sigmoid(X @ theta_k.T)
            y_pred = np.argmax(pred_matrix, axis=1)
            y_pred = np.array([i+1 for i in y_pred])
            print(classification_report(y_true, y_pred))
        elif int(sys.argv[1]) == 3:
            """神经网络算法"""
            weight_path = 'ex3-neural network/ex3weights.mat'
            theta_1, theta_2 = load_weight(weight_path)
            a1 = X
            z2 = a1 @ theta_1.T
            a2 = sigmoid(z2)
            a2 = np.insert(a2, 0, 1, axis=1)
            z3 = a2 @ theta_2.T
            a3 = sigmoid(z3)
            y_pred = np.argmax(a3, axis=1)+1
            print(classification_report(y_true, y_pred))
        else:
            print("请输入正确的参数：\n\t1.\t执行梯度下降算法\t\t\t2.\t执行高级算法\t\t3.\t执行神经网络算法")
    else:
        print("请输入正确的参数：\n\t1.\t执行梯度下降算法\t\t\t2.\t执行高级算法\t\t3.\t执行神经网络算法")

if __name__ == '__main__':
    main()
