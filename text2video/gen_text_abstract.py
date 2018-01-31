# -*- coding:utf-8 -*-
import jieba
import jieba.posseg as pseg
import os
import sys
import fnmatch
import numpy as np
from tqdm import tqdm
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# 计算所有词的tfidf权值矩阵
cut_num = 1000


def find_files(directory, pattern='*.txt'):
    '''Recursively finds all files matching the pattern.'''
    filenames = []
    for _, dirnames, files in os.walk(directory):
        for file in fnmatch.filter(files, pattern):
            filenames.append(file)
    return filenames


corpus = []

#path = "./fasttext/THUCNews/"
path=r'D:\Program Files\THUCNews\THUCNews'
files = os.listdir(path)
print files
n = 0  # num of all files
num_files = [0 for i in range(len(files))]
dir = -1
for file in files:
    print 'file is ', file
    filenames = find_files(os.path.join(r'D:\Program Files\THUCNews\THUCNews', file))
    dir = dir + 1
    # for i,filename in tqdm(enumerate(filenames)):
    i = 0
    for filename in filenames:
        # print 'filename is',filename
        num_files[dir] = num_files[dir] + 1
        path = os.path.join(r'D:\Program Files\THUCNews\THUCNews',file,filename)
        with open(path, "r") as f:
            # print 'path',path
            lines = f.read()

            content = lines
            # print 'content',content
            if content != "":
                seg_content = jieba.cut(content)
                seg_content = ' '.join(seg_content)

                corpus.append(seg_content)
                n = n + 1

        i = i + 1
        if i == cut_num:
            print '%d fils ok' % cut_num
            # world = "World"
            # print "Hello %s !" % world
            break
# print 'corpus',len(corpus)
vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
freq_matrix = vectorizer.fit_transform(corpus)
tfidf = transformer.fit_transform(freq_matrix)  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
tfidf = tfidf.tocoo()
weight = tfidf.toarray()
# print np.shape(weight)


# -*- coding:utf-8 -*-
# 文本摘要方法有很多，主要分为抽取式和生成式，应用比较多的是抽取式，也比较简单，就是从文本中抽取重要的句子或段落。本方法主要是利用句子中的关键词的距离，主要思想和参考来自阮一峰的网络日志http://www.ruanyifeng.com/blog/2013/03/automatic_summarization.html

# !/user/bin/python
# coding:utf-8
__author__ = 'hj'
import nltk
import numpy
import jieba
import codecs
# 计算关键词
import operator

# k = 2#第2个文件
# #print corpus[k]

# rows = tfidf.row
# #print 'rows',type(rows),rows,len(rows)#每个词属于的文件,1D
# cols = tfidf.col
# #print type(cols),cols,len(cols)#每个词的词号
# weights = tfidf.data
# row_index = np.where(rows==k)#第2个文件中的所有词在所有词中的位置 (1,390)
# #print 'row_index',np.shape(row_index[0]),type(row_index[0]),row_index[0]
# keywords = dict()
# for index in row_index[0]:
#     col_index = cols[index]#词号
#     keyword = word[col_index]#词
#     keywords[keyword] = weights[index]

# keywords = sorted(keywords.items(), key=operator.itemgetter(1), reverse=True)
# #print keywords
# for keyword, weight in keywords[:10]:
#     print keyword

# k = 2#第2个文件
# print corpus[k]
all_keywords = []
dir = 0
pri_num_files = 0
all_keywords.append([])
for k in range(n):

    rows = tfidf.row
    # print 'rows',type(rows),rows,len(rows)#每个词属于的文件,1D
    cols = tfidf.col
    # print type(cols),cols,len(cols)#每个词的词号
    weights = tfidf.data
    row_index = np.where(rows == k)  # 第2个文件中的所有词在所有词中的位置 (1,390)
    # print 'row_index',np.shape(row_index[0]),type(row_index[0]),row_index[0]
    keywords = dict()
    for index in row_index[0]:
        col_index = cols[index]  # 词号
        keyword = word[col_index]  # 词
        keywords[keyword] = weights[index]

    keywords = sorted(keywords.items(), key=operator.itemgetter(1), reverse=True)
    # print keywords
    for keyword, weight in keywords[:10]:
        print keyword
    # print 'k-pri_num_files',k-pri_num_files #2
    # print num_files[dir] #3
    if (k - pri_num_files) == num_files[dir]:
        # print 'k',k
        # print 'pri_num_files',pri_num_files
        all_keywords.append([])
        dir = dir + 1
        all_keywords[dir].append(keywords)
        pri_num_files = pri_num_files + num_files[dir]

        print '---------------'
    else:
        all_keywords[dir].append(keywords)

# print 'dir',dir #13

# 抽取法提取文本摘要
N = 5  # 关键词数量
CLUSTER_THRESHOLD = 5  # 单词间的距离
TOP_SENTENCES = 2  # 返回的top n句子


def find_files(directory, pattern='*.txt'):
    '''Recursively finds all files matching the pattern.'''
    filenames = []
    for _, dirnames, files in os.walk(directory):
        for file in fnmatch.filter(files, pattern):
            filenames.append(file)
    return filenames


# 分句
def sent_tokenizer(path):
    with open(path, 'r')as f:
        texts = f.read().decode('utf8')
        print 'texts:', texts

        start = 0
        i = 0  # 每个字符的位置
        sentences = []
        punt_list = '.!?。！？'.decode('utf8') + '\n' + ' '  # ',.!?:;~，。！？：；～'.decode('utf8')
        token = texts[1]  # add in
        for text in texts:
            if text in punt_list and token not in punt_list:  # 检查标点符号下一个字符是否还是标点
                sentences.append(texts[start:i + 1])  # 当前标点符号位置
                start = i + 1  # start标记到下一句的开头
                i += 1
            else:
                i += 1  # 若不是标点符号，则字符位置继续前移
                token = list(texts[start:i + 2]).pop()  # 取下一个字符
        if start < len(texts):
            sentences.append(texts[start:])  # 这是为了处理文本末尾没有标点符号的情况
        # print 'sentences',sentences
    return sentences


# 停用词
def load_stopwordslist(path):
    print('load stopwords...')
    stoplist = [line.strip() for line in codecs.open(path, 'r', encoding='utf8').readlines()]
    stopwrods = {}.fromkeys(stoplist)
    return stopwrods


# 摘要
def summarize(text, keywords):
    #stopwords = load_stopwordslist('./Chinese/stopwords.dat')
    stopwords = load_stopwordslist('D:\search\stopwords.dat')
    sentences = sent_tokenizer(text)
    # words=[w for sentence in sentences for w in jieba.cut(sentence)
    #       if w not in stopwords if len(w)>1 and w!='\t']
    # wordfre=nltk.FreqDist(words)#每个单词的出现次数
    # topn_words=[w[0] for w in sorted(wordfre.items(),key=lambda d:d[1],reverse=True)][:N]
    # for i in range(len(topn_words)):
    #    print topn_words[i].encode('utf8')
    topn_words = []
    for keyword, weight in keywords[:3]:
        topn_words.append(keyword)
    # print 'topn_words',topn_words
    scored_sentences = _score_sentences(sentences, topn_words)
    # print 'scored_sentences',scored_sentences
    # approach 1,利用均值和标准差过滤非重要句子
    avg = numpy.mean([s[1] for s in scored_sentences])  # 均值
    std = numpy.std([s[1] for s in scored_sentences])  # 标准差
    mean_scored = [(sent_idx, score) for (sent_idx, score) in scored_sentences
                   if score > (avg + 0.5 * std)]
    # approach 2，返回top n句子
    # print 'mean_scored',mean_scored
    top_n_scored = sorted(scored_sentences, key=lambda s: s[1])[-TOP_SENTENCES:]
    top_n_scored = sorted(top_n_scored, key=lambda s: s[0])
    return dict(top_n_summary=[sentences[idx] for (idx, score) in top_n_scored],
                mean_scored_summary=[sentences[idx] for (idx, score) in mean_scored])


# 句子得分
def _score_sentences(sentences, topn_words):
    scores = []
    sentence_idx = -1
    for s in [list(jieba.cut(s)) for s in sentences]:
        # print 's',s
        sentence_idx += 1
        word_idx = []
        for w in topn_words:
            try:
                word_idx.append(s.index(w))  # 关键词出现在该句子中的索引位置
            except ValueError:  # w不在句子中
                pass
        word_idx.sort()
        # print 'word_idx',word_idx
        if len(word_idx) == 0:
            continue
        # 对于两个连续的单词，利用单词位置索引，通过距离阀值计算族
        # 一个句子划分为若干族
        clusters = []
        cluster = [word_idx[0]]
        i = 1
        # print 'cluster',cluster
        while i < len(word_idx):
            if word_idx[i] - word_idx[i - 1] < CLUSTER_THRESHOLD:
                cluster.append(word_idx[i])
            else:
                clusters.append(cluster[:])
                cluster = [word_idx[i]]
            i += 1
        # clusters.append(cluster)
        # print 'clusters',clusters
        # 对每个族打分，每个族类的最大分数是对句子的打分
        max_cluster_score = 0
        for c in clusters:
            significant_words_in_cluster = len(c)
            total_words_in_cluster = c[-1] - c[0] + 1
            score = 1.0 * significant_words_in_cluster * significant_words_in_cluster / total_words_in_cluster
            if score > max_cluster_score:
                max_cluster_score = score
        scores.append((sentence_idx, max_cluster_score))
    return scores;


# if __name__=='__main__':
#     dict=summarize('./fasttext/THUCNews/tiyu/0.txt',keywords)
#     print('-----------approach 1-------------')
#     for sent in dict['mean_scored_summary']:
#         print(sent)
#     print('-----------approach 2-------------')
#     for sent in dict['top_n_summary']:
#         print(sent)

# if __name__=='__main__':
#     for_search=[]
#     dict=summarize('./fasttext/THUCNews/tiyu/2.txt',keywords)
#     print('-----------approach 1-------------')
#     for sent in dict['mean_scored_summary']:
#         print(sent)
#     print('-----------approach 2-------------')
#     i=0
#     for sent in dict['top_n_summary']:
#         print(sent)
#         for_search.append([])
#         for w in jieba.cut(sent):
#             for word,weight in keywords[:10]:
#                 #print 'w',w
#                 #print 'word',word

#                 if w==word:
#                     print w
#                     for_search[i].append(w)
#         i=i+1
#     print for_search

if __name__ == '__main__':
    for_search = []
    abstract = []
    n = -1
    print 'all_keywords', np.shape(all_keywords), len(all_keywords[13])
    # all_keywords (14, 999) 999
    for file in files:
        n = n + 1
        filenames = find_files(os.path.join(r'D:\Program Files\THUCNews\THUCNews', file))
        for_search.append([])
        abstract.append([])
        for j, filename in tqdm(enumerate(filenames)):

            for_search[n].append([])
            abstract[n].append([])
            #path = "./fasttext/THUCNews/" + file + '/' + filename
            path = os.path.join(r'D:\Program Files\THUCNews\THUCNews',file ,filename)
            print 'n', n
            print 'j', j
            dict2 = summarize(path, all_keywords[n][j])
            print('-----------approach 1-------------')
            for sent in dict2['mean_scored_summary']:
                print(sent)
            print('-----------approach 2-------------')
            i = 0
            for sent in dict2['top_n_summary']:
                print(sent)
                for_search[n][j].append([])
                abstract[n][j].append(sent)
                for w in jieba.cut(sent):
                    for word, weight in all_keywords[n][j][:10]:
                        # print 'w',w
                        # print 'word',word

                        if w == word:
                            print w
                            for_search[n][j][i].append(w)
                i = i + 1
            # print for_search
            if j == cut_num - 1:
                break
    with open(r'D:\search\for_search.txt', 'w')as f:
        for n in range(len(files)):
            for j in range(num_files[n]):
                for i in range(len(for_search[n][j])):
                    for w in range(len(for_search[n][j][i])):
                        f.write(for_search[n][j][i][w].encode('utf8'))
                        f.write(' ')
                        if w == 1:
                            break

                    f.write('\n')
                f.write('file end!\n')

    with open(r'D:\search\abstract.txt', 'w')as f:
        for n in range(len(files)):
            for j in range(num_files[n]):
                for i in range(len(abstract[n][j])):
                    f.write(abstract[n][j][i].encode('utf8').replace('\xe3\x80\x80\xe3\x80\x80', '').strip())
                    # .replace('\n','').replace('\t','').replace(' ',''))
                    f.write('\n')

                f.write('file end!\n')

    print 'abstract', abstract[0][0][1][0]
    print 'finished'

# if __name__=='__main__':
#     dict=summarize(u'腾讯科技讯（刘亚澜）10月22日消息，'
#         u'前优酷土豆技术副总裁黄冬已于日前正式加盟芒果TV，出任CTO一职。'
#         u'资料显示，黄冬历任土豆网技术副总裁、优酷土豆集团产品技术副总裁等职务，'
#         u'曾主持设计、运营过优酷土豆多个大型高容量产品和系统。'
#         u'此番加入芒果TV或与芒果TV计划自主研发智能硬件OS有关。'
#         u'今年3月，芒果TV对外公布其全平台日均独立用户突破3000万，日均VV突破1亿，'
#         u'但挥之不去的是业内对其技术能力能否匹配发展速度的质疑，'
#         u'亟须招揽技术人才提升整体技术能力。'
#         u'芒果TV是国内互联网电视七大牌照方之一，之前采取的是“封闭模式”与硬件厂商预装合作，'
#         u'而现在是“开放下载”+“厂商预装”。'
#         u'黄冬在加盟土豆网之前曾是国内FreeBSD（开源OS）社区发起者之一，'
#         u'是研究并使用开源OS的技术专家，离开优酷土豆集团后其加盟果壳电子，'
#         u'涉足智能硬件行业，将开源OS与硬件结合，创办魔豆智能路由器。'
#         u'未来黄冬可能会整合其在开源OS、智能硬件上的经验，结合芒果的牌照及资源优势，'
#         u'在智能硬件或OS领域发力。'
#         u'公开信息显示，芒果TV在今年6月对外宣布完成A轮5亿人民币融资，估值70亿。'
#         u'据芒果TV控股方芒果传媒的消息人士透露，芒果TV即将启动B轮融资。')
#     print('-----------approach 1-------------')
#     for sent in dict['top_n_summary']:
#         print(sent)
#     print('-----------approach 2-------------')
#     for sent in dict['mean_scored_summary']:
#         print(sent)
