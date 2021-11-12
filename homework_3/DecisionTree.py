import pandas as pd
from sklearn import tree #导入需要的模块
from sklearn.model_selection import train_test_split

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
    # data1.to_csv('data1.csv')
    median = data1['SalePrice'].median()
    # print('房价中位数：', median)
    data1.loc[data1['SalePrice'] >= median, 'SalePrice'] = median + 1     # 1 广义的代表高房价
    # print(data1)
    data1.loc[data1['SalePrice'] < median, 'SalePrice'] = median - 1     # median-1 广义的代表低房价
    # print(data1)

    # 由于特征Electrical只有一个缺失值，故只需删除该行即可
    data2 = data1.drop(data1.loc[data1['Electrical'].isnull()].index)
    # 检查缺失值数量
    # print('缺失值数量：', data2.isnull().sum().max())

    feature_data = data2.drop(['SalePrice'], axis=1)
    target_data = data2['SalePrice'].astype(str)     # 将分类特征转化为字符型

    # print(feature_data.shape)
    # print(target_data.shape)

    # 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt'。
    sub_feature_data = {
        'OverallQual': feature_data['OverallQual'],  # 分类特征
        'GrLivArea': feature_data['GrLivArea'],
        'GarageCars': feature_data['GarageCars'],
        'TotalBsmtSF': feature_data['TotalBsmtSF'],
        'FullBath': feature_data['FullBath'],
        'TotRmsAbvGrd': feature_data['TotRmsAbvGrd'],
        'YearBuilt': feature_data['YearBuilt']
    }
    sub_feature_data = pd.DataFrame(sub_feature_data)
    sub_feature_data['OverallQual'] = sub_feature_data['OverallQual'].astype(str)  # 将分类特征转化为字符型

    # print(sub_feature_data)
    # print(target_data)

    # print(sub_feature_data.shape)
    # print(target_data)
    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(sub_feature_data, target_data, test_size=0.3)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':

    data = pd.read_csv('data.csv')
    X_train, X_test, y_train, y_test = data_processing(data)

    clf = tree.DecisionTreeClassifier() #实例化算法对象，需要使用参数
    clf = clf.fit(X_train,y_train) #用训练集数据训练模型
    result = clf.score(X_test,y_test) #打分，导入测试集，从接口中调用需要的信息

    print('结果：', result)
    # print(y_test)
    # print(clf.predict(X_test))