import sys
import os
import requests
import numpy as np
import tempfile
import pickle
import sys
from prepare_chinese_data import *
from tqdm import *
batch_size=50
single_size=218
double_size=47524
sys.path.append("..")
phoneme_label={}
n=0
target_indices=[[]for k in range(10000/batch_size)]#
target_values=[[]for k in range(10000/batch_size)]
dense_shape=[[]for k in range(10000/batch_size)]
len_lines=[[]for k in range(10000/batch_size)]
max_phoneme_label_len=[0 for k in range(10000/batch_size)]
k=0
for line in open('/home/deepvoice/deepvoice2_chinese/data/test/phone.txt','r').readlines():
    
    file_name='/home/deepvoice/deepvoice2_chinese/data/test/phonemes/'+line[:7]+'.txt'
    k=k+1
    if k==1:
        print file_name
    f=open(file_name,'w')
    f.write(line[8:])
    f.close()