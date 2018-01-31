import sys
import os
import requests
import numpy as np
import tempfile
import pickle
import sys

from tqdm import tqdm
import fnmatch
f1_p=np.load('f1.npy')
f2_p=np.load('f2.npy')
files_len=2
index_model4=np.load('index_model4.npy')
print 'index_model4',np.shape(index_model4)
index_model42=[[0 for j in range(len(index_model4[i]))]for i in range(files_len)]
f1=[[0 for j in range(len(f1_p[i]))]for i in range(files_len)]
f2=[[0 for j in range(len(f2_p[i]))]for i in range(files_len)]
#print len(index_model4[0]),len(index_model4[1])
#print len(f1[0]),len(f1[1])
#print len(f2[0]),len(f2[1])
for i in range(files_len):
    for j in range(len(index_model4[i])):
        index_model42[i][j]=index_model4[i][j]
        f1[i][j]=f1_p[i][j]
        f2[i][j]=f2_p[i][j]



n=0

label5=[[]for i in range(files_len)]
for i in range(files_len):
    index_model42[i].insert(0,0)
    index_model42[i].insert(0,0)
    index_model42[i].append(0)
    index_model42[i].append(0)
    for j in range(2,len(index_model42[i])-2):
        label5[i].append([index_model42[i][j-2],index_model42[i][j-1],
                         index_model42[i][j],index_model42[i][j+1],
                         index_model42[i][j+2],])
#print label5
for i in range(files_len):#len(files)
    j=0
    for j in range(len(index_model42[i])-4): 
        #print i,j
        if n>1:
            break
        label5[i][j].append(f2[i][j])
        label5[i][j].append(f1[i][j])
        #label5[i][j].append(float(line.strip('\n').strip('\r').split(' ')[0]))
        #label5[i][j].append(float(line.strip('\n').strip('\r').split(' ')[1]))
        j=j+1    

#print label5[0][0],len(label5)
label55=[[[]for j in range(len(label5[i]))] for i in range(files_len)]
ide1=np.identity(219,dtype=np.float)
for i in range(files_len):
    for j in range(len(label5[i])):
        for k in range(len(label5[i][j])-2):
            #print label5[i][j][k]
            label55[i][j]=label55[i][j]+list(ide1[label5[i][j][k]])
        label55[i][j].append(f2[i][j])
        label55[i][j].append(f1[i][j])
#print label55

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
files=find_files("/home/chinese_data/_2_f0_voiced")
for i in range(len(files)):
    with open(files[i],'w') as f:
        for j in range(len(label55[i])):
            for k in range(len(label55[i][j])):
                f.write(str(label55[i][j][k])+' ')
            f.write('\n')

