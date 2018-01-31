import requests
import numpy as np
import tempfile
import pickle
import sys
import fnmatch
import librosa
sys.path.append("..")
from scipy.io import wavfile
from tqdm import *
import os
files_len=2
file_len_train=np.load('file_len_train.npy')
file_len_dev=np.load('file_len_dev.npy')
index_model4_pre=np.load('index_model4.npy')
voiced_target_pre=np.load('voiced_target.npy')
_,channels=np.shape(index_model4_pre)
index_model4=[[0 for j in range(channels)] for i in range(files_len)]
voiced_target=[[0 for j in range(channels)] for i in range(files_len)]
len_index_model4_i=[0 for i in range(files_len)]
for i in tqdm(range(files_len)):
    for j in range(channels):
        index_model4[i][j]=index_model4_pre[i][j]
        voiced_target[i][j]=voiced_target_pre[i][j]
    len_index_model4_i[i]=len(index_model4[i])
len_index_model4=max(len_index_model4_i)

len_f0=[0 for i in range(files_len)]

def find_files(directory, pattern='*.txt'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    nn=0
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            nn=nn+1
            if nn>2:
                break
            files.append(os.path.join(root, filename))
            
    return files
files=find_files("/home/chinese_data/f0_voiced")


#load f0
f1=[[]for i in range(files_len)]
f2=[[]for i in range(files_len)]
len1=[[]for i in range(files_len)]
for i in tqdm(range(files_len)):
    f = open(files[i], 'r')
    sourceInLines = f.readlines()  
    f.close() 
    new = []
 
    for line in sourceInLines: 
        temp1 = line.strip('\n')
        temp2 = temp1.split(' ')     
        new.append(temp2)
    for t in range(len(new)):
        f2[i].append(float(new[t][0]))
        new[t].pop(0)
        new[t][0]=float(new[t][0])
        f1[i].append(new[t][0])
        
    len1[i]=len(f1[i])
    
max_len=max(len1)
for i in tqdm(range(files_len)):
    for j in range(len(f1[i]),max_len):
        f1[i].append(0)
        f2[i].append(0)
#print np.shape(f1)
#print f1,np.shape(f1[0]),np.shape(f1[1])

#load f0 end

max_file_len_train=max(file_len_train)
max_file_len_train=len_index_model4
print len_index_model4
print max_len

#for i in tqdm(range(max_len)): 
print len(index_model4[0]),len(index_model4[1])
if max_file_len_train>max_len:
    cha=max_file_len_train-max_len
    #print cha
    for j in range(cha):
        #print j
        for k in range(files_len):
            f1[k].append(0)
            f2[k].append(0)
else:
    cha=max_len-max_file_len_train
    print cha
    for j in range(cha):
        print j
        for k in range(files_len):
            index_model4[k].append(0)
            voiced_target[k].append(0)

#print np.shape(f1)
#print np.shape(index_model4)
#print np.shape(voiced_target)
print len(index_model4[0]),len(index_model4[1])
print len(f1[0]),len(f1[1])
print len(f2[0]),len(f2[1])
np.save('index_model4.npy', np.array(index_model4))
np.save('voiced_target.npy', np.array(voiced_target))
np.save('f1.npy', np.array(f1))
np.save('f2.npy', np.array(f2))
     
        
    