import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
from statsmodels.sandbox.regression.predstd import wls_prediction_std
# 导入模型相关的库
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
# 导入模型相关的库
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

# 数据分析：查看各特征之间的相关性
def data_analysis(data):
    sns.distplot(data['SalePrice'])
    plt.savefig('chart/SalePrice_distplot.png')  # 保存图片

    # 挑选特征：皮尔逊相关系数
    sns.jointplot(x='1stFlrSF', y='SalePrice', data=data)
    plt.savefig('chart/analysis/1stFlrSF-SalePrice.png')  # 保存图片

    sns.jointplot(x='GrLivArea', y='SalePrice', data=data)
    plt.savefig('chart/analysis/GrLivArea-SalePrice.png')  # 保存图片

    sns.jointplot(x='PoolArea', y='SalePrice', data=data)
    plt.savefig('chart/analysis/PoolArea-SalePrice.png')  # 保存图片

    sns.lmplot(x='GrLivArea', y='SalePrice', data=data)
    plt.savefig('chart/analysis/GrLivArea-SalePrice-lmplot.png')  # 保存图片

    # 针对分类变量，无法使用皮尔逊相关系数，可以通过观察每个分类值上目标变量的变化程度来查看相关性，通常来说，在不同值上数据范围变化较大，两变量相关性较大。
    sns.boxplot(x='OverallQual', y='SalePrice', data=data)
    plt.savefig('chart/analysis/OverallQual-SalePrice.png')  # 保存图片

    sns.boxplot(x='YearBuilt', y='SalePrice', data=data)
    plt.savefig('chart/analysis/YearBuilt-SalePrice.png')  # 保存图片

    grouped = data.groupby('OverallQual')
    g1 = grouped['SalePrice'].mean().reset_index('OverallQual')
    sns.barplot(x='OverallQual', y='SalePrice', data=g1)
    plt.savefig('chart/analysis/OverallQual-SalePrice_barplot.png')  # 保存图片

    '''# 计算热力图：上两种分析都是针对单个特征与目标变量逐一分析，这种方法非常耗时繁琐，下面介绍一种系统性分析特征与目标变量相关性的方法，通过对数据集整体特征（数值型数据）进行分析，来找出最佳特征
    # 设置图幅大小
    plt.rcParams['figure.figsize'] = (15, 10)
    # 计算相关系数
    corrmatrix = data.corr()
    # 绘制热力图，热力图横纵坐标分别是data的index/column,vmax/vmin设置热力图颜色标识上下限，center显示颜色标识中心位置，cmap颜色标识颜色设置
    heatmap_fig_1 = sns.heatmap(corrmatrix, square=True, vmax=1, vmin=-1, center=0.0, cmap='coolwarm')
    fig = heatmap_fig_1.get_figure()
    fig.savefig('chart/analysis/heatmap.png')  # 保存图片'''

    # 取相关性前10的特征
    corrmatrix_1 = data.corr()
    k = 10
    # data.nlargest(k, 'target')在data中取‘target'列值排前十的行
    # cols为排前十的行的index,在本例中即为与’SalePrice‘相关性最大的前十个特征名
    cols = corrmatrix_1.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(data[cols].values.T)
    # data[cols].values.T
    # 设置坐标轴字体大小
    sns.set(font_scale=1.25)
    # sns.heatmap() cbar是否显示颜色条，默认是；cmap显示颜色；annot是否显示每个值，默认不显示；
    # square是否正方形方框，默认为False,fmt当显示annotate时annot的格式；annot_kws为annot设置格式
    # yticklabels为Y轴刻度标签值，xticklabels为X轴刻度标签值
    hm = sns.heatmap(cm, cmap='RdPu', annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                     yticklabels=cols.values, xticklabels=cols.values)
    fig_1 = hm.get_figure()
    fig_1.savefig('chart/analysis/heatmap_valid.png', bbox_inches='tight')  # 保存图片

    '''
    由上图可以看出：

    1. 'OverallQual', 'GrLivArea'这两个变量与'SalePrice'有很强的线性关系;
    2. 'GarageCars', 'GarageArea'与'SalePrice'也有很强的线性关系，但'GarageCars', 'GarageArea'相关性0.88，有很强的共线性，只取其一即可，取与目标变量关系更强的'GarageCars'；
    3. 同样地，'TotalBsmtSF', '1stFlrSF'也有很强的共线性，只取其一即可，取'TotalBsmtSF'；
    4. 因此，选取的特征有：'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt'。

    '''

    cols1 = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd',
             'YearBuilt']
    sns.pairplot(data[cols1], size=2.5)
    plt.savefig('chart/analysis/pairplot.png')  # 保存图片

    plt.show()

# 数据预处理：填补缺省值，删除无用列，划分训练集和测试集
def data_processing(data):
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

    return X_train, X_test, y_train, y_test, df_train

def linear_regression(X_train, X_test, y_train, y_test, df_train):
    # ols("target~feature+C(feature)", data=data
    # C(feature)表示这个特征为分类特征category

    # 使用最小二乘法计算回归模型参数
    # 一元线性回归
    lr_model_GrLivArea = ols("SalePrice~GrLivArea", data=df_train).fit()
    lr_model_TotalBsmtSF = ols("SalePrice~TotalBsmtSF", data=df_train).fit()
    lr_model_YearBuilt = ols("SalePrice~YearBuilt", data=df_train).fit()

    # 预测测试集
    y_hat_GrLivArea = lr_model_GrLivArea(X_test)
    y_hat_TotalBsmtSF = lr_model_TotalBsmtSF(X_test)
    y_hat_YearBuilt = lr_model_YearBuilt(X_test)

    plt.scatter(X_train['GrLivArea'], y_train, alpha=0.3)
    plt.xlabel('GrLivArea')
    plt.ylabel('SalePrice')
    plt.plot(X_test['GrLivArea'], y_hat_GrLivArea, 'r', alpha=0.9)
    plt.savefig('chart/regression/SalePrice_GrLivArea.png')  # 保存图片

    plt.scatter(X_train['TotalBsmtSF'], y_train, alpha=0.3)
    plt.xlabel('TotalBsmtSF')
    plt.ylabel('SalePrice')
    plt.plot(X_test['TotalBsmtSF'], y_hat_TotalBsmtSF, 'r', alpha=0.9)
    plt.savefig('chart/regression/SalePrice_TotalBsmtSF.png')  # 保存图片

    plt.scatter(X_train['YearBuilt'], y_train, alpha=0.3)
    plt.xlabel('YearBuilt')
    plt.ylabel('SalePrice')
    plt.plot(X_test['YearBuilt'], y_hat_YearBuilt, 'r', alpha=0.9)
    plt.savefig('chart/regression/SalePrice_YearBuilt.png')  # 保存图片

    plt.show()

    # 多元线性回归
    # lr_model = ols("SalePrice~C(OverallQual)+GrLivArea+C(GarageCars)+TotalBsmtSF+C(FullBath)+YearBuilt", data=df_train).fit()
    lr_model = ols("SalePrice~GrLivArea+TotalBsmtSF+YearBuilt", data=df_train).fit()
    print(lr_model.summary())
    print('预测结果：')
    print(lr_model.predict(X_test))

def ridge_regression():
    pass

if __name__ == '__main__':
    data = pd.read_csv("data.csv")

    # 数据分析
    data_analysis(data)

    # 数据预处理：填补缺省值，删除无用列，划分训练集和测试集
    X_train, X_test, y_train, y_test, df_train = data_processing(data)

    # 建立回归模型
    # 线性回归模型：一元回归、多元回归
    linear_regression(X_train, X_test, y_train, y_test, df_train)

    # 岭回归模型
    ridge_regression()