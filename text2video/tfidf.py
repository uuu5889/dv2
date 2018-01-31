# -*- coding:utf-8 -*- 
import jieba  
import jieba.posseg as pseg  
import os  
import sys
import fnmatch
import operator
import pickle
from tqdm import tqdm
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer

def find_files(directory, pattern='*.txt'):
    '''Recursively finds all files matching the pattern.'''
    filenames = []
    for _,dirnames, files in os.walk(directory):
        for file in fnmatch.filter(files, pattern):
            filenames.append(file)
    return filenames

def train(corpus_path):
    '''Train and dump the model.'''
    filenames = find_files(corpus_path)
    corpus = []
    for filename in tqdm(filenames):
        path = corpus_path + filename
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line[:9] == "<content>":
                    content = line[9:-11]
                    if content != "":
                        seg_content = jieba.cut(content)
                        seg_content = ' '.join(seg_content)
                        corpus.append(seg_content)
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    freq_matrix = vectorizer.fit_transform(corpus)
    tfidf = transformer.fit_transform(freq_matrix)
    with open("./vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("./transformer.pkl","wb") as f:
        pickle.dump(transformer, f)
        
def tf_idf(text_path, vectorizer, transformer):
    '''Calculate the tfidf value of a new text. '''
    content = " "
    with open(text_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line != "":
                seg_line = jieba.cut(line)
                seg_line = ' '.join(seg_line)
                content = content + ' ' + seg_line 
    with open(vectorizer, "r") as f:
        vectorizer = pickle.load(f)
    with open(transformer,"r") as f:
        transformer = pickle.load(f)
    word_bag = vectorizer.get_feature_names()
    freq_matrix = vectorizer.transform([content])
    tfidf = transformer.transform(freq_matrix)
    return tfidf, word_bag

def get_keywords(text_path, vectorizer, transformer, keyword_num=5):
    '''Get the key words of a new text.'''    
    tfidf, word_bag = tf_idf(text_path, vectorizer, transformer)
    tfidf = tfidf.toarray()[0]
    word_list = zip(word_bag, tfidf) 
    word_list.sort(key=operator.itemgetter(1))
    keywords = [word for word, _ in word_list[:keyword_num]]
    return keywords
    
    
    
    
    
    
    
    
    
    
    