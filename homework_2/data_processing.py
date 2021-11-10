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

def missing_processing(data):
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

    return feature_data, target_data

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

    print(feature_data)

    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(feature_data, target_data, test_size=0.3)

    df_train = pd.concat([X_train, y_train], axis=1)
    # ols("target~feature+C(feature)", data=data
    # C(feature)表示这个特征为分类特征category
    # lr_model = ols("SalePrice~C(OverallQual)+GrLivArea+C(GarageCars)+TotalBsmtSF+C(FullBath)+YearBuilt", data=df_train).fit()
    lr_model = ols("SalePrice~GrLivArea+TotalBsmtSF+YearBuilt", data=df_train).fit()
    print(lr_model.summary())
    print(lr_model.params)

    # 预测测试集
    print(lr_model.predict(X_test))
    print(X_train)
    print(y_train)
    print(X_test)

    plt.scatter(X_train['YearBuilt'], y_train, alpha=0.3)
    plt.xlabel('GrLivArea')
    plt.ylabel('SalePrice')
    plt.plot(X_test['TotalBsmtSF'], lr_model.predict(X_test), 'r', alpha=0.9)
    #plt.show()

    '''# prstd为标准方差，iv_l为置信区间下限，iv_u为置信区间上限
    prstd, iv_l, iv_u = wls_prediction_std(lr_model, alpha=0.05)
    # lr_model.predict()为训练集的预测值
    predict_low_upper = pd.DataFrame([lr_model.predict(), iv_l, iv_u], index=['PredictSalePrice', 'iv_l', 'iv_u']).T
    predict_low_upper.plot(kind='hist', alpha=0.4)
    #plt.savefig('chart/PredictSalePrices.png')  # 保存图片'''

    plt.show()





