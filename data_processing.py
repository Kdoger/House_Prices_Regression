'''
数据预处理

'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from statsmodels.formula.api import ols
from statsmodels.sandbox.regression.predstd import wls_prediction_std
# 导入模型相关的库
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score



if __name__ == '__main__':
    data = pd.read_csv("data.csv")

    # isnull() boolean, isnull().sum()统计所有缺失值的个数
    # isnull().count()统计所有项个数（包括缺失值和非缺失值），.count()统计所有非缺失值个数
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
    # pd.concat() axis=0 index,axis=1 column, keys列名
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.head(20))

    # 处理缺失值，将含缺失值的整列剔除
    data1 = data.drop(missing_data[missing_data['Total'] > 1].index, axis=1)
    # 由于特征Electrical只有一个缺失值，故只需删除该行即可
    data2 = data1.drop(data1.loc[data1['Electrical'].isnull()].index)
    # 检查缺失值数量
    print(data2.isnull().sum().max())

    feature_data = data2.drop(['SalePrice'], axis=1)
    target_data = data2['SalePrice']

    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(feature_data, target_data, test_size=0.3)

    df_train = pd.concat([X_train, y_train], axis=1)
    # ols("target~feature+C(feature)", data=data
    # C(feature)表示这个特征为分类特征category
    lr_model = ols("SalePrice~C(OverallQual)+GrLivArea+C(GarageCars)+TotalBsmtSF+C(FullBath)+YearBuilt",
                   data=df_train).fit()
    print(lr_model.summary())

    # 预测测试集
    lr_model.predict(X_test)

    # prstd为标准方差，iv_l为置信区间下限，iv_u为置信区间上限
    prstd, iv_l, iv_u = wls_prediction_std(lr_model, alpha=0.05)
    # lr_model.predict()为训练集的预测值
    predict_low_upper = pd.DataFrame([lr_model.predict(), iv_l, iv_u], index=['PredictSalePrice', 'iv_l', 'iv_u']).T
    predict_low_upper.plot(kind='hist', alpha=0.4)
    plt.savefig('chart/PredictSalePrices.png')  # 保存图片

    # 岭回归
    def rmse_cv(model):
        rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=5))
        return (rmse)


    # 导入ridge模型
    model_ridge = Ridge()

    # 对超参数取值进行猜测和验证
    alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
    cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean() for alpha in alphas]

    # 画图查看不同超参数的模型的分数
    cv_ridge = pd.Series(cv_ridge, index=alphas)
    cv_ridge.plot(title="Validation - Just Do It")
    plt.xlabel("alpha")
    plt.ylabel("rmse")
    cv_ridge

    # 线性回归

    # 二分类-罗辑回归

    plt.show()





