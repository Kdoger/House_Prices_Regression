'''
支持向量机

'''
import pandas as pd
from sklearn import svm #导入需要的模块
from sklearn.model_selection import train_test_split

from homework_3.data_processing import Data_Processing

if __name__ == '__main__':

    data = pd.read_csv('data.csv')
    dp = Data_Processing(data)

    X_train, X_test, y_train, y_test = dp.data_processing()

    clf = svm.SVC() #实例化算法对象，需要使用参数
    clf.fit(X_train,y_train) #用训练集数据训练模型
    result = clf.score(X_test,y_test) #打分，导入测试集，从接口中调用需要的信息

    print('SVM模型：')
    print('正确率：%.2f' % (result * 100) + '%')
    # print(y_test)
    # print(clf.predict(X_test))