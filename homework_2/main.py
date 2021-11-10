import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import skew
from scipy.special import boxcox1p
from statsmodels.formula.api import ols

from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

    '''grouped = data.groupby('OverallQual')
    g1 = grouped['SalePrice'].mean().reset_index('OverallQual')
    sns.barplot(x='OverallQual', y='SalePrice', data=g1)
    plt.savefig('chart/analysis/OverallQual-SalePrice_barplot.png')  # 保存图片'''

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
    # print(missing_data.head(20))

    # 处理缺失值，将含缺失值的整列剔除
    data1 = data.drop(missing_data[missing_data['Total'] > 1].index, axis=1)
    # 由于特征Electrical只有一个缺失值，故只需删除该行即可
    data2 = data1.drop(data1.loc[data1['Electrical'].isnull()].index)
    # 检查缺失值数量
    # print(data2.isnull().sum().max())

    feature_data = data2.drop(['SalePrice'], axis=1)
    target_data = data2['SalePrice']

    # print(feature_data)

    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(feature_data, target_data, test_size=0.3)

    df_train = pd.concat([X_train, y_train], axis=1)

    return X_train, X_test, y_train, y_test, df_train

# 线性回归模型：一元回归、多元回归
def linear_regression(X_train, X_test, y_train, y_test, df_train):
    # ols("target~feature+C(feature)", data=data
    # C(feature)表示这个特征为分类特征category

    # 使用最小二乘法计算回归模型参数
    # 一元线性回归
    lr_model_GrLivArea = ols("SalePrice~GrLivArea", data=df_train).fit()
    lr_model_TotalBsmtSF = ols("SalePrice~TotalBsmtSF", data=df_train).fit()
    lr_model_YearBuilt = ols("SalePrice~YearBuilt", data=df_train).fit()

    # 预测测试集
    y_hat_GrLivArea = lr_model_GrLivArea.predict(X_test)
    y_hat_TotalBsmtSF = lr_model_TotalBsmtSF.predict(X_test)
    y_hat_YearBuilt = lr_model_YearBuilt.predict(X_test)

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
    print('线性回归预测结果：')
    print(lr_model.predict(X_test))

# 岭回归模型
def ridge_regression(data):
    feature_data = data
    target_data = data['SalePrice']

    train, test, y_train, y_test = train_test_split(feature_data, target_data, test_size=0.3)

    # print(train.head(5))

    # 1.首先对train进行处理，删除可视化中出现的异常值
    train = train.drop(train[(train['TotalBsmtSF'] > 5000) & (train['SalePrice'] < 200000)].index)
    train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000)].index)

    # 这里采用对数变换的方法使其符合正态分布
    train["SalePrice"] = np.log(train["SalePrice"])

    # 2.将train和test联合起来一起进行数据处理
    train_id = train['Id']
    test_id = test['Id']
    ntrain = train.shape[0]
    ntest = test.shape[0]
    y_train = train.SalePrice.values
    all_data = pd.concat((train, test)).reset_index(drop=True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)
    # print("all_data size is : {}".format(all_data.shape))  # all_data size is : (2917, 80)

    # 由于ID对预测没有作用，删除ID字段
    all_data.drop(['Id'], axis=1, inplace=True)

    # 查看缺失值比率
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    # print(all_data_na)
    missing_data = pd.DataFrame({'missing_data': all_data_na})
    # print(missing_data.head(20))

    # 对于缺失率在80%以上的特征删除
    all_data = all_data.drop('PoolQC', axis=1)
    all_data = all_data.drop('MiscFeature', axis=1)
    all_data = all_data.drop('Alley', axis=1)
    all_data = all_data.drop('Fence', axis=1)
    # print(all_data.shape)  # (2917, 75)

    # 对于其他缺失值进行处理, 壁炉为空可能是没有，用none填充
    all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('none')

    # LotFrontage代表房屋前街道的长度, 房屋前街道的长度应该和一个街区的房屋相同，可以取同一个街区房屋的街道长度的平均值
    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    # 对于Garage类的4个特征，缺失率一致，一起处理，可能是没有车库，用none填充
    for c in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        all_data[c] = all_data[c].fillna('none')

    # 对于garage，同样猜测缺失值缺失的原因可能是因为房屋没有车库，连续型变量用0填充
    for c in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all_data[c] = all_data[c].fillna(0)

    # 对于地下室相关的连续变量，缺失同样认为房屋可能是没有地下室，用0填充
    for c in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[c] = all_data[c].fillna(0)

    # 地下室相关离散变量，同理用None填充
    for c in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        all_data[c] = all_data[c].fillna('None')

    # Mas为砖石结构相关变量，缺失值我们同样认为是没有砖石结构，用0和none填补缺失值
    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

    # MSZoning代表房屋所处的用地类型，先看下不同取值
    all_data.groupby('MSZoning')['MasVnrType'].count().reset_index()
    # 由于业务上房屋类型是必须的，一般都有，考虑用众数填充
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

    # 由于数据Functional缺失即为Typ，所以进行填充Typ
    all_data["Functional"] = all_data["Functional"].fillna("Typ")

    # 对于Utilities,观察到除了一个“NoSeWa”和2个NA之外，所有记录都是“AllPub”，对于房价预测用处很小，删除这个特征
    all_data.drop(['Utilities'], axis=1, inplace=True)

    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    # print(all_data_na)
    missing_data = pd.DataFrame({'missing_data': all_data_na})
    # print(missing_data)

    # 填充剩余的缺失值
    '''for i in missing_data.index:
        print(all_data[i].head())'''  # 未展示

    for i in ('SaleType', 'KitchenQual', 'Electrical', 'Exterior2nd', 'Exterior1st'):
        all_data[i] = all_data[i].fillna(all_data[i].mode()[0])

    # 查看缺失值的比率，发现已经处理完毕，all_data里已经没有缺失值
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    # print(all_data_na)
    missing_data = pd.DataFrame({'missing_data': all_data_na})
    # print(missing_data)

    # 对于一些数值型特征，数值并不表示大小，将其值转换为字符型
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

    all_data['OverallCond'] = all_data['OverallCond'].astype(str)

    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)

    # 将地下室面积、1楼面积、2楼面积相加得到总面积特征
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    # 由前面的可视化将房子建造时间做一个划分，以1990进行划分，1990前为0,1990后为1
    all_data['YearBuilt_cut'] = all_data['YearBuilt'].apply(lambda x: 1 if x > 1990 else 0)

    all_data['Total_sqr_footage'] = (all_data['BsmtFinSF1'] + all_data['BsmtFinSF2'] +
                                     all_data['1stFlrSF'] + all_data['2ndFlrSF'])

    all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +
                                   all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))

    # print(all_data.shape)  # (2917, 79)

    # 将all_data分开为训练集与测试集两部分，查看新特征与房价的相关性
    new_train = all_data[:ntrain]
    new_test = all_data[ntrain:]
    new_train['SalePrice'] = y_train

    # 观察到Total_Bathrooms等于5或6时都只有一行，且对应房价较为异常，删除这两个值
    new_train.loc[:, 'Total_Bathrooms'].value_counts()
    new_train = new_train.drop(new_train[new_train['Total_Bathrooms'] >= 5.0].index)

    # 将new_train与new_test重新组合成all_data进行数据的统一处理
    ntrain = new_train.shape[0]
    ntest = new_test.shape[0]
    y_train = new_train.SalePrice.values
    all_data = pd.concat((new_train, new_test)).reset_index(drop=True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)
    # print("all_data size is : {}".format(all_data.shape))  # all_data size is : (2915, 78)

    # 对有序性离散变量使用label encoder 进行编码

    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
            'ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual', 'BsmtFinType1',
            'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond',
            'YrSold', 'MoSold')
    for c in cols:
        lbe = LabelEncoder()
        lbe.fit(list(all_data[c].values))
        all_data[c] = lbe.transform(list(all_data[c].values))
    # print(all_data.shape)  # (2915, 78)
    # print(all_data.head(5))

    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    # 查看所有数字特征的偏度
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_feats})
    # print(skewness.head(10))

    # 查看有多少特征的偏度不符合要求，并进行转换
    skewness = skewness[abs(skewness) > 0.75]
    # print("有{}个特征需要转换 ".format(skewness.shape[0]))
    # 有59个特征需要转换
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        # all_data[feat] += 1
        all_data[feat] = boxcox1p(all_data[feat], lam)

    # 将无序型离散变量转化为哑变量（one-hot编码）
    all_data = pd.get_dummies(all_data)
    shreshold = 0.9
    corr_all_data = all_data.corr().abs()
    # 取矩阵的上三角部分，判断系数大于0.9的并删除
    data_up = corr_all_data.where(np.triu(np.ones(corr_all_data.shape), k=1).astype(np.bool))

    drop_col = [column for column in data_up.columns if any(data_up[column] > 0.9)]
    all_data = all_data.drop(columns=drop_col)
    # print(all_data.shape)  # (2915, 207)

    # 将训练集与测试集分开，用于建模与测试
    train = all_data[:ntrain]
    test = all_data[ntrain:]
    # print(train.head())

    def rmse_cv(model):
        rmse = np.sqrt(-cross_val_score(model, train, y_train, scoring="neg_mean_squared_error", cv=5))
        return (rmse)

    # 导入ridge模型
    model_ridge = Ridge()

    # 对超参数取值进行猜测和验证
    alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
    cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean() for alpha in alphas]

    # 画图查看不同超参数的模型的分数
    cv_ridge = pd.Series(cv_ridge, index=alphas)
    cv_ridge.plot(title="alpha VS rmse")
    plt.xlabel("alpha")
    plt.ylabel("rmse")
    plt.savefig('chart/regression/ridge_regression.png')   # 保存图片
    # print(cv_ridge)

    plt.show()

    # alpha参数用我们之前验证过的10,然后用训练集对模型进行训练
    clf = Ridge(alpha=10)
    clf.fit(train, y_train)
    # 输出 Ridge(alpha=10, copy_X=True, fit_intercept=True, max_iter=None, normalize=False,random_state=None, solver='auto', tol=0.001)

    # 对测试集进行预测，并导出结果
    predict = clf.predict(test)
    test_pre = pd.DataFrame()
    test_pre['ID'] = test_id
    test_pre['SalePrice'] = np.exp(predict)
    test_pre.to_csv('prediction.csv', index=False)
    print('岭回归模型预测结果：')
    print(test_pre.head())

    pass

# 逻辑回归模型
def logistic_regression(data):
    # 求房价中位数
    median = data['SalePrice'].median()
    # 选取的特征有：'GrLivArea', 'TotalBsmtSF', 'YearBuilt'。
    subdata = {
        'GrLivArea': data['GrLivArea'],
        'YearBuilt': data['YearBuilt'],
        'TotalBsmtSF': data['TotalBsmtSF']
    }
    subdata = pd.DataFrame(subdata)
    train, test, y_train, y_test = train_test_split(subdata, data['SalePrice'], test_size=0.3)

    # 标准化
    std = StandardScaler()
    train = std.fit_transform(train)
    test = std.transform(test)

    # 建立模型
    lr_model = LogisticRegression(C=1.0)
    lr_model.fit(train, y_train)

    y_predict = lr_model.predict(test)
    # print(y_predict)
    # print(lr_model.coef_)
    # print(lr_model.intercept_)
    print('逻辑回归模型预测准确率：%.2f ' %(lr_model.score(test, y_test) * 100) + '%')

    '''xcord_1 = []
    xcord_2 = []
    ycord_1 = []
    ycord_2 = []
    for i in range(data.shape[0]):
        if data['SalePrice'][i] >= median:   # 高房价
            xcord_1.append(data['YearBuilt'][i])
            ycord_1.append(data['GrLivArea'][i])
        else:
            xcord_2.append(data['YearBuilt'][i])
            ycord_2.append(data['GrLivArea'][i])

    plt.scatter(xcord_1, ycord_1, s=30, c='red', marker='s')
    plt.scatter(xcord_2, ycord_2, s=30, c='green')

    plt.show()'''

if __name__ == '__main__':
    data = pd.read_csv("data.csv")

    # 数据分析
    data_analysis(data)

    # 数据预处理：填补缺省值，删除无用列，划分训练集和测试集
    X_train, X_test, y_train, y_test, df_train = data_processing(data)

    # 建立回归模型
    # 线性回归模型：一元回归、多元回归
    linear_regression(X_train, X_test, y_train, y_test, df_train)
    print('-----分割线-----')

    # 岭回归模型
    ridge_regression(data)
    print('-----分割线-----')

    # 逻辑回归模型
    logistic_regression(data)