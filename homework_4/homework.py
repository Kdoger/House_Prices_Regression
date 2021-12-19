import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

def data_load():
    data = pd.read_csv('winequality-red.csv')

    return data

'''主成分分析'''
def data_PCA(dataMat, topNfeat=11):
    meanValues = np.mean(dataMat, axis=0)  # 竖着求平均值，数据格式是m×n

    meanRemoved = dataMat - meanValues  # 0均值化  m×n维
    covMat = np.cov(meanRemoved, rowvar=0)  # 每一列作为一个独立变量求协方差  n×n维
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 求特征值和特征向量  eigVects是n×n维
    eigValInd = np.argsort(-eigVals)  # 特征值由大到小排序，eigValInd是个arrary数组 1×n维
    eigValInd = eigValInd[:topNfeat]  # 选取前topNfeat个特征值的序号  1×r维
    redEigVects = eigVects[:, eigValInd]  # 把符合条件的几列特征筛选出来组成P  n×r维  matrix类型

    ''' * 操作对于多维array不适用，最好是np.matmul()或者np.dot(),这两个函数矩阵乘法'''
    lowDDataMat = np.matmul(meanRemoved.values,redEigVects)  # 矩阵点乘筛选的特征向量矩阵  m×r维 公式Y=X*P，

    reconMat = np.matmul(lowDDataMat , redEigVects.T)   # 转换新空间的数据  m×n维
    reconMat = pd.DataFrame(reconMat)
    reconMat.columns = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
                        'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
    reconMat = reconMat + meanValues    # matrix类型和series类型不能直接进行算术运算


    return lowDDataMat, reconMat

'''聚类分析——K-means'''
def cluster_K_means(data):  # 默认有几个样本，分几类

    # 加载数据
    df = data.drop(columns=['quality'])

    # print(df.info())   #可以查看数据类型  都是float64,没有空值

    # 由于是非监督学习，不使用label
    x = np.array(df.astype(float))
    # 将每一列特征标准化为标准正太分布，注意，标准化是针对每一列而言的
    x = preprocessing.scale(x)

    clf = KMeans(n_clusters=data.groupby(['quality']).size().shape[0])   # 原始数据共有几类
    clf.fit(x)
    # 上面已把数据分成两组

    # 下面计算分组准确率是多少
    y = np.array(data['quality'])

    correct = 0
    for i in range(len(x)):
        predict_data = np.array(x[i].astype(float))
        predict_data = predict_data.reshape(-1, len(predict_data))
        predict = clf.predict(predict_data)
        # print(predict[0], y[i])
        if predict[0] == y[i] - 3:
            correct += 1

    score = (correct * 1.0 / len(x)) * 100
    print('K-means聚类：：')
    print('正确率：%.2f' % score + '%')

    pass

'''聚类分析——层次聚类'''
'''
六、层次聚类的优缺点

优点：
1，距离和规则的相似度容易定义，限制少；
2，不需要预先制定聚类数；
3，可以发现类的层次关系；
4，可以聚类成其它形状

缺点：
1，计算复杂度太高；
2，奇异值也能产生很大影响；
3，算法很可能聚类成链状
'''
def hierarchy_cluster(data):

    clustering = AgglomerativeClustering(n_clusters=data.groupby(['quality']).size().shape[0]).fit(data.drop(columns=['quality']))
    print('类别列表：', clustering.labels_)
    print('详细信息如下：')
    print(clustering.children_)

if __name__ == '__main__':

    data = data_load()
    data_no_quality = data.drop(columns=['quality'])

    '''主成分分析'''
    proccess_data, reconMat = data_PCA(data_no_quality, 10)    # 降维后的数据和重构的原始数据
    proccess_data = pd.DataFrame(proccess_data)
    print(proccess_data)
    print(reconMat)

    '''聚类分析'''
    cluster_K_means(data)   # K-means
    '''
    hierarchy_cluster()函数输出说明：
    类别列表：指的是类别名称构成的list，类别名称分别是0，1，2，3，4，5；共6类
    详细信息：指的是哪两个样本构成一类，二者之间的差值代表类别名称
    '''
    hierarchy_cluster(data)


