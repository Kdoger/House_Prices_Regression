import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim import models



'''对 （发表时间、摘要） 进行处理'''
'''读取数据集'''
def data_load():
    data = pd.read_csv('cnki_data_1.csv')

    data = data.drop(columns=['论文题目','院校','学位'])   # 只保留发表时间和摘要字段

    return data

'''分词'''
def word_split(data):
    pass

'''将Series对象转换成string类型，用于生成词云'''
def to_String(data, name):
    word_string = ''
    for i in range(len(data)):
        if isinstance(data[i],str):    # 确保是str类型才可以生成词云
            word_string = word_string + data[i] + ' '  # 需要加空格将每个词分开

    with open(name + '.txt','w') as f:
        for i in range(len(data)):
            if isinstance(data[i], str):
                f.write(data[i] + '\n')     # 一行一条记录
    f.close()

    return word_string

'''生成词云'''
def word_cloud(data, name):
    # WordCloud的默认字体不支持中文，指定了字体文件
    wordcloud = WordCloud(background_color="white",\
                    width=800,\
                   height=800,
                   font_path='SimHei.ttf',
                   ).generate(data)
    wordcloud.to_file(name + '.png')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

'''LDA建模'''
def lda_model(data):
    text = []
    for i in range(len(data)):
        text.append([])
        if isinstance(data[i], str):
            text[i].append(data[i])
    text = np.array(text)
    return text


if __name__ == '__main__':

    data = data_load()
    # text = lda_model(data['摘要'])
    #dct = Dictionary(text)
    #print(text)
    publish_time = to_String(data['摘要'], 'abstract')   # to_String函数的第二个参数用来指定文本的名字 ——> publish_time.txt  abstract.txt
    # word_cloud(publish_time,'abstract')     # word_cloud函数的第二个参数用来指定词云图片的名字 ——> publish_time.png  abstract.png