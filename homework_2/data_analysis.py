'''
对数据进行预处理

'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = pd.read_csv("data.csv")

    #print(data['SalePrice'].describe())  # 查看数据

    sns.distplot(data['SalePrice'])
    plt.savefig('chart/SalePrice_distplot.png')  # 保存图片

    # 挑选特征：皮尔逊相关系数
    sns.jointplot(x='1stFlrSF', y='SalePrice', data=data)
    plt.savefig('chart/1stFlrSF-SalePrice.png')  # 保存图片

    sns.jointplot(x='GrLivArea', y='SalePrice', data=data)
    plt.savefig('chart/GrLivArea-SalePrice.png')  # 保存图片

    sns.jointplot(x='PoolArea', y='SalePrice', data=data)
    plt.savefig('chart/PoolArea-SalePrice.png')  # 保存图片

    sns.lmplot(x='GrLivArea', y='SalePrice', data=data)
    plt.savefig('chart/GrLivArea-SalePrice-lmplot.png')  # 保存图片

    # 针对分类变量，无法使用皮尔逊相关系数，可以通过观察每个分类值上目标变量的变化程度来查看相关性，通常来说，在不同值上数据范围变化较大，两变量相关性较大。
    sns.boxplot(x='OverallQual', y='SalePrice', data=data)
    plt.savefig('chart/OverallQual-SalePrice.png')  # 保存图片

    sns.boxplot(x='YearBuilt', y='SalePrice', data=data)
    plt.savefig('chart/YearBuilt-SalePrice.png')  # 保存图片'''

    '''grouped = data.groupby('OverallQual')
    g1 = grouped['SalePrice'].mean().reset_index('OverallQual')
    sns.barplot(x='OverallQual', y='SalePrice', data=g1)
    plt.savefig('chart/OverallQual-SalePrice_barplot.png')  # 保存图片'''

    '''# 计算热力图：上两种分析都是针对单个特征与目标变量逐一分析，这种方法非常耗时繁琐，下面介绍一种系统性分析特征与目标变量相关性的方法，通过对数据集整体特征（数值型数据）进行分析，来找出最佳特征
    # 设置图幅大小
    plt.rcParams['figure.figsize'] = (15, 10)
    # 计算相关系数
    corrmatrix = data.corr()
    # 绘制热力图，热力图横纵坐标分别是data的index/column,vmax/vmin设置热力图颜色标识上下限，center显示颜色标识中心位置，cmap颜色标识颜色设置
    heatmap_fig_1 = sns.heatmap(corrmatrix, square=True, vmax=1, vmin=-1, center=0.0, cmap='coolwarm')
    fig = heatmap_fig_1.get_figure()
    fig.savefig('chart/heatmap.png')  # 保存图片'''

    # 取相关性前10的特征
    corrmatrix = data.corr()
    k = 10
    # data.nlargest(k, 'target')在data中取‘target'列值排前十的行
    # cols为排前十的行的index,在本例中即为与’SalePrice‘相关性最大的前十个特征名
    cols = corrmatrix.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(data[cols].values.T)
    # data[cols].values.T
    # 设置坐标轴字体大小
    sns.set(font_scale=1.25)
    # sns.heatmap() cbar是否显示颜色条，默认是；cmap显示颜色；annot是否显示每个值，默认不显示；
    # square是否正方形方框，默认为False,fmt当显示annotate时annot的格式；annot_kws为annot设置格式
    # yticklabels为Y轴刻度标签值，xticklabels为X轴刻度标签值
    hm = sns.heatmap(cm, cmap='RdPu', annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                     yticklabels=cols.values, xticklabels=cols.values)
    plt.savefig('chart/heatmap_valid.png', bbox_inches='tight')  # 保存图片

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
    plt.savefig('chart/pairplot.png')  # 保存图片

    plt.show()
    #print(data)