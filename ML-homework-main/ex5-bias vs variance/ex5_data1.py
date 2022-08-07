import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import  scipy.optimize as opt
import sys

def cost(theta, X, Y):
    """代价函数"""
    return np.sum(np.power((X @ theta.T) - Y, 2)) / (2 * X.shape[0])

def regularized_cost(theta, X, Y, l=1):
    """正则化后的代价函数"""
    temp = np.power(theta, 2)
    temp[0] = 0
    return cost(theta, X, Y) + np.sum(temp) * l / (2 * X.shape[0])

def gradient(theta, X, Y):
    """梯度函数"""
    m = X.shape[0]
    temp = X.T @ ((X @ theta.T) - Y)
    return temp / m

def regularized_gradient(theta, X, Y, l=1):
    """正则化后的梯度"""
    m = X.shape[0]
    temp = theta.copy()
    temp[0] = 0
    return gradient(theta, X, Y) + temp * l / m

def plot_data(theta, X, Y):
    """绘制数据"""
    b = theta[0]
    a = theta[1]
    x = X[:, 1:]
    plt.figure(figsize=(6, 4))
    plt.scatter(x, Y, marker='x', c='r',s=50, label='Training Data')
    plt.plot(x, a*x+b, c='b', label='Prediction')
    plt.legend()
    plt.show()

def advanced_pre(X, Y, l=1):
    """高级优化算法"""
    theta = np.ones(X.shape[1])
    res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, Y, l), jac=regularized_gradient, method='TNC')
    return res.x

def load_data(path):
    """处理数据"""
    data = sio.loadmat(path)
    return data['X'].reshape(-1,), data['y'].reshape(-1,), data['Xval'].reshape(-1,), data['yval'].reshape(-1,), data['Xtest'].reshape(-1,), data['ytest'].reshape(-1,)    

def normalize_feature(df):
    """数据标准化"""
    return df.apply(lambda col: (col - col.mean()) / col.std())

def poly_feature(x, power):
    """扩展特征值"""
    data = {f'f{i}': np.power(x, i) for i in range(1, power+1)}
    df = pd.DataFrame(data)
    return df

def data_poly_pre(*args, power):
    """扩展特征值并标准化"""
    def prepare(data):
        df = poly_feature(data, power)
        df = normalize_feature(df).values
        return np.insert(df, 0, 1, axis=1)
    return [prepare(i) for i in args]
    
def plot_learning_curve(X, Y, Xval, Yval, l=0):
    """画出学习曲线"""
    if l==0:
        titl = '过拟合'
    elif l==1:
        titl = '拟合'
    else:
        titl = '欠拟合'

    m = X.shape[0]
    training_cost, cv_cost = [], []
    for i in range(1, m+1):
        res = advanced_pre(X[:i, :], Y[:i], l)
        tr = regularized_cost(res, X[:i, :], Y[:i], 0)
        cv = regularized_cost(res, Xval, Yval, 0)
        training_cost.append(tr)
        cv_cost.append(cv)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, m+1), training_cost, label='Training Cost')
    plt.plot(range(1, m+1), cv_cost, label='CV Cost')
    plt.legend(loc='best', title=r'$\lambda={}$'.format(l) + titl)
    plt.show()

def select_lambda(X, Y, Xval, Yval, Xtest, Ytest):
    """绘制曲线选择lambda"""
    candidate_l = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    training_cost, cv_cost, te_cost = [], [], []

    for l in candidate_l:
        theta = advanced_pre(X, Y, l)
        tc = regularized_cost(theta, X, Y, 0)
        cv = regularized_cost(theta, Xval, Yval, 0)
        test_c = regularized_cost(theta, Xtest, Ytest, 0)

        training_cost.append(tc)
        cv_cost.append(cv)
        te_cost.append(test_c)
    print(f"验证集选择的lambda:{candidate_l[np.argmin(cv_cost)]}")
    print(f"测试集选择的lambda:{candidate_l[np.argmin(te_cost)]}")
    plt.figure(figsize=(6, 4))
    plt.plot(candidate_l, training_cost, label='Training Cost')
    plt.plot(candidate_l, cv_cost, label='CV Cost')
    plt.plot(candidate_l, te_cost, label='Test Cost')
    plt.legend()
    plt.xlabel(r'$\lambda$')
    plt.ylabel('cost')
    plt.show()

def main():
    """主函数"""
    path = 'ex5-bias vs variance/ex5data1.mat'
    X, Y, Xval, Yval, Xtest, Ytest = load_data(path)

    X, Xval, Xtest = data_poly_pre(X, Xval, Xtest, power=8)

    if len(sys.argv)==2:
        if int(sys.argv[1])==1:
            pred_l = [0, 1, 100]
            for l in pred_l:
                plot_learning_curve(X, Y, Xval, Yval, l)
        elif int(sys.argv[1])==2:
            select_lambda(X, Y, Xval, Yval, Xtest, Ytest)
        else:
            print("请输入正确的参数\n\t1.\t三种拟合情况\t\t2.\t不同lambda下代价变化")
    else:
        print("请输入正确的参数\n\t1.\t三种拟合情况\t\t2.\t不同lambda下代价变化")

if __name__ == '__main__':
    main()