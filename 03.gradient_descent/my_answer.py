import math

import numpy as np
import pandas as pd
import seaborn as sns



def loss_f(X, Y, m_curr: float, b_curr: float):
    n = len(X)
    return (1/n)*( sum( (Y - (m_curr * X + b_curr)) ** 2 ) )



def deriv(X, Y, m_curr: float, b_curr: float):
    n = len(X)
    d_m = -(2 / n) * sum( X * (Y - (m_curr * X + b_curr)))
    d_b = -(2 / n) * sum(Y - (m_curr * X + b_curr))
    return d_m, d_b


def grad(X, Y):
    N = X.size
    m_curr = 0
    b_curr = 0
    iteration = 1000
    learning_rate = 0.0003
    l_list = []
    for i in range(iteration):

        l_f = loss_f(X, Y, m_curr, b_curr)
        l_list.append(l_f)
        d_m, d_b = deriv(X, Y, m_curr, b_curr)
        m_curr = m_curr - (learning_rate * d_m)
        b_curr = b_curr - (learning_rate * d_b)
        print(f'i:{i},m_curr: {m_curr},b_curr:{b_curr},loss_f:{l_f}')
        if len(l_list) > 1:
            if math.isclose(l_f, l_list[-2], rel_tol=1e-20) :
                break





# x = np.array([1, 2, 3, 4, 5])
# y = np.array([5, 7, 9, 11, 13])
# grad(x, y)


df = pd.read_csv('test_scores.csv')
x = np.array(df.loc[:, 'math'])
y = np.array(df.loc[:, 'cs'])
grad(x, y)
