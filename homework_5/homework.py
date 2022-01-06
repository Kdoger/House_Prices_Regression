import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim import models
# gensim安装的是3.8版本，最新版本中没有wrappers
from gensim.models.wrappers import DtmModel

'''
对 （发表时间、摘要） 进行处理
思路：
1. 对数据不做处理，先绘制词云图，根据词云图可以看出哪些词出现的次数多（字体越大数量越多）
2. 由步骤1可以得到一些频率高但无意义的词，将这些词过滤掉
3. 由步骤2得到的文本数据用于建模
'''

'''读取数据集'''
def data_load():
    data = pd.read_csv('cnki_data_1.csv')

    data = data.drop(columns=['论文题目','院校','学位'])   # 只保留发表时间和摘要字段

    return data

'''将Series对象转换成txt文本，用于生成词云'''
def to_save_txt(data, name):
    # word_string = ''

    '''
    for i in range(len(data)):
        if isinstance(data[i],str):    # 确保是str类型才可以生成词云
            word_string = word_string + data[i] + ' '  # 需要加空格将每个词分开
    '''
    with open(name + '.txt','w') as f:
        for i in range(len(data)):
            if isinstance(data[i], str):
                f.write(data[i] + '\n')     # 一行一条记录
    f.close()

'''用于生成发表时间词云'''
def word_cloud_publish_time(data):
    word_string = ''

    for i in range(len(data)):
        if isinstance(data[i],str):    # 确保是str类型才可以生成词云
            word_string = word_string + data[i] + ' '  # 需要加空格将每个词分开

    wordcloud = WordCloud(background_color="white", \
                          width=1920, \
                          height=1080,
                          font_path='SimHei.ttf',
                          ).generate(word_string)
    wordcloud.to_file('publish_time.png')

'''分词'''
def word_split(name):

    with open(name + '.txt', 'r', encoding="utf-8") as f:
        word_text = f.read()
    stop_word_text = 'UUA0 U 的 和 与 了'    # 空格隔开的str
    print(word_text)
    print(type(word_text))

    # jieba 分词
    word_text_cut = jieba.cut(word_text)
    stop_word_text_cut = jieba.cut(stop_word_text)

    cut_text = " ".join(word_text_cut)
    stop_words_text = " ".join(stop_word_text_cut)

    return cut_text, stop_words_text

def histgram(data):
    # 解决中文显示问题
    #plt.rcParams['font.sans-serif'] = ['SFNSRounded']
    #plt.rcParams['axes.unicode_minus'] = False

    publish_time = []
    for i in range(len(data)):
        if data[i] == '2010年':
            publish_time.append(2010)
        if data[i] == '2011年':
            publish_time.append(2011)
        if data[i] == '2012年':
            publish_time.append(2012)
        if data[i] == '2013年':
            publish_time.append(2013)
        if data[i] == '2010年':
            publish_time.append(2010)
        if data[i] == '2014年':
            publish_time.append(2014)
        if data[i] == '2015年':
            publish_time.append(2015)
        if data[i] == '2016年':
            publish_time.append(2016)
        if data[i] == '2010年':
            publish_time.append(2010)
        if data[i] == '2017年':
            publish_time.append(2017)
        if data[i] == '2018年':
            publish_time.append(2018)
        if data[i] == '2019年':
            publish_time.append(2019)

    plt.hist(x=publish_time, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)

    plt.grid(axis='y', alpha=0.75)  # 是否有网格
    plt.xlabel('value')  # 横轴标签
    plt.ylabel('frequency')  # 纵轴标签
    plt.title('publish_time_distribution')  # 设置title
    plt.savefig('publish_time_distribution.png')  # 保存图片
    # plt.show()  # 显示图片

'''生成词云'''
def word_cloud(words, stop_words, name):
    # WordCloud的默认字体不支持中文，指定了字体文件
    wordcloud = WordCloud(background_color="white",\
                          width=1920,\
                          height=1080,
                          font_path='SimHei.ttf',
                          stopwords=stop_words,
                          ).generate(words)
    wordcloud.to_file(name + '.png')
    '''
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    '''

'''LDA建模'''
def lda_model(data):
    stop_word_list = [',', '。', ':','、', '-','.']
    texts = []

    for i in range(len(data)):
        texts.append([])
        if isinstance(data[i], str):
            data[i] = jieba.cut(data[i])
            for word in data[i]:
                if word not in stop_word_list:
                    texts[i].append(word)

    texts = np.array(texts)

    dct = Dictionary(texts)
    # 每个字在corpus中的id 和 词频  相当于tf
    corpus = [dct.doc2bow(text) for text in texts]

    # 相当于求出了tfidf
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    # 这里传进去的数据既可以是tf 也可以是idf
    lda = LdaModel(corpus_tfidf, num_topics=10, id2word=dct)
    print('模型输出：')
    for result in lda.print_topics(num_topics=10, num_words=5):
        print(result)

'''DTM建模'''
def dtm_model(data):
    stop_word_list = [',', '。', ':', '、', '-', '.']
    texts = []

    for i in range(len(data)):
        texts.append([])
        if isinstance(data[i], str):
            data[i] = jieba.cut(data[i])
            for word in data[i]:
                if word not in stop_word_list:
                    texts[i].append(word)

    texts = np.array(texts)

    dct = Dictionary(texts)
    # 每个字在corpus中的id 和 词频  相当于tf
    corpus = [dct.doc2bow(text) for text in texts]

    time_slices = [1] * len(corpus)

    model = DtmModel('dtm-win64.exe', corpus=corpus, time_slices=time_slices,
                     id2word=dct, num_topics=10, mode='fit')

    model.show_topics(num_topics=10, times=1)

    pass

if __name__ == '__main__':

    name = 'abstract'
    data = data_load()   # 加载原始数据集

    to_save_txt(data['摘要'], name)  # 生成摘要文本
    abstract_cut_texts, abstract_stop_words_text = word_split(name)   # 分词
    print(type(abstract_cut_texts))

    histgram(data['发表时间'])
    word_cloud_publish_time(data['发表时间'])
    word_cloud(abstract_cut_texts, abstract_stop_words_text, name)  # 绘制词云
    lda_model(data['摘要'])   # lad 建模

    dtm_model(data['摘要'])   # dtm建模


