import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# 导入模型相关的库
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import data_processing as dp


def linear_regression():
    pass

def ridge_regression(model, X_train, y_train):
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=5))
    return (rmse)
    pass

if __name__ == '__main__':

    data = pd.read_csv("data.csv")
    feature_data, target_data = dp.missing_processing(data)
    X_train, X_test, y_train, y_test = train_test_split(feature_data, target_data, test_size=0.3)

    #导入ridge模型
    model_ridge = Ridge()

    # 对超参数取值进行猜测和验证
    alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
    cv_ridge = [ridge_regression(Ridge(alpha = alpha), X_train, y_train).mean() for alpha in alphas]

    # 画图查看不同超参数的模型的分数
    cv_ridge = pd.Series(cv_ridge, index=alphas)
    cv_ridge.plot(title="Validation - Just Do It")
    plt.xlabel("alpha")
    plt.ylabel("rmse")
    print(cv_ridge)
    plt.show()



