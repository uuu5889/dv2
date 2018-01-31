import sys
sys.path.append("..")
sys.path.append("/home/VCTK-Corpus/wav48/p225")



import os

import tensorflow as tf
import numpy as np
from scipy.io import wavfile

import itertools 
import re 

def postprocess(encoder_output,audio_label_pre2,beamsize):
    q=encoder_output
    print "encoder_output",np.shape(encoder_output)
    x,y,z=np.shape(encoder_output)
    #print x,y,z

    index3=[[[0 for i in range(beamsize)] for j in range(y)]for k in range(x)]
    BEAM3=[[[0 for i in range(beamsize)] for j in range(y)]for k in range(x)]
    #print np.shape(index3)
    #p=np.array([[5,2,4,6],[3,7,4,5],[9,1,10,3]])
    #p=np.random.rand(3,4)
    for t in range(x):
        #print "t is:",t
        p=q[t]
        b,c=y,z

        index=[[0 for i in range(beamsize)] for j in range(b)]
        mask=[[0 for i in range(beamsize)] for j in range(b)]
        index2=[[0 for i in range(beamsize)] for j in range(b)]

        states_1=p[0,:]#embedding_size
        index_states_1=np.argsort(states_1)
        BEAM=[]
        BEAM2=[[0 for i in range(beamsize)] for j in range(b)]#b,beamsize
        for i in range(beamsize):
            BEAM.append(states_1[index_states_1[len(states_1)-1-i]])#(beamsize)

            index[0][i]=index_states_1[len(states_1)-1-i]
        for i in range(1,b):#b:max_seq_len
            SET=p[i,:]#(c,)
            sum=[]
            for j in range(c):#c:embedding_size
                for m in BEAM:
                    sum.append(SET[j]+m)


            sort_sum=np.argsort(sum)
            kk=0
            for k in range(len(sum)):
                        #print "i is",i,"k is",k

                if kk==beamsize:
                    break
                if k==0:
                    BEAM=[]
                                #print "len(sum)",len(sum),"k",k,"shape_sort_sum",np.shape(sort_sum)
                index_sum=sort_sum[len(sum)-1-k]

                aa=(index_sum+1)/beamsize#5 6/2=3
                bb=(index_sum+1)%beamsize
                if bb!=0:

                    if aa!=30 and index[i-1][bb-1]!=30 and abs(index[i-1][bb-1]-aa)>1:
                        continue
                    else:
                        index[i][kk]=aa
                        mask[i][kk]=bb-1
                        kk=kk+1
                        BEAM.append(sum[sort_sum[len(sum)-1-k]])
                                            #print "k:",k,"kk:",kk,"shape_BEAM:",np.shape(BEAM)
                if bb==0:
                    if (aa-1)!=30 and index[i-1][beamsize-1]!=30 and abs(index[i-1][beamsize-1]-aa+1)>1:
                        continue
                    else:
                        index[i][kk]=aa-1
                        mask[i][kk]=beamsize-1
                        kk=kk+1
                        BEAM.append(sum[sort_sum[len(sum)-1-k]])

            BEAM2[i]=BEAM

        index2=index
        for k in range(beamsize): 
            index2[b-2][k]=index[b-2][mask[b-1][k]]
            mask[b-2][k]=mask[b-2][mask[b-1][k]]
        for i in range(b-2):
            for k in range(beamsize):
                index2[b-3-i][k]=index[b-3-i][mask[b-2-i][k]]
                mask[b-3-i][k]=mask[b-3-i][mask[b-1-i][k]]
                #print "index2 is",index2#path

        index3[t]=index2
        BEAM3[t]=BEAM2
    yinsu_size=218
    #print "index3 is",index3,"BEAM3 is",BEAM3
        #print "shape_index3:",np.shape(index3),"shape_BEAM3:",np.shape(BEAM3)
    
    index_final=np.array(index3)[:,:,0]
    BEAM_final=np.array(BEAM3)[:,:,0]
    #add
    index_model4=index_final
    voiced_target=[[0 for j in range(y)]for i in range(x)]
    for i in range(x):
        for j in range(y):
            if index_model4[i][j]==z-1:
                index_model4[i][j]=0
            else:
                #index_model4[i][j] += 1
                index_model4[i][j]=index_final[i][j]+1
                voiced_target[i][j]=1
    print "index_model4:",np.shape(index_model4),index_model4
    #print "voiced_target:",type(voiced_target),voiced_target



    #add end
    #print index_final,np.shape(index_final)
    yinsu=[[]for t in range(x)]
    duration=[[]for t in range(x)]
    len_yinsu=[[]for t in range(x)]
    for t in range(x):
        yinsu[t].append(index_final[t][0])
    d=1
    for t in range(x):
        for j in range(1,y):
            if index_final[t][j]!=index_final[t][j-1]:
                yinsu[t].append(index_final[t][j])
                duration[t].append(d)
                d=1
            else:
                d=d+1
        duration[t].append(d)

    for t in range(x):
        len_yinsu[t]=len(yinsu[t])
    for t in range(x):
        for j in range(len_yinsu[t]):
            if yinsu[t][j]==z-1:
                yinsu[t].remove(z-1)
                duration[t].remove(duration[t][j])

    leijia=0

    num_buckets=10
    #mean_duration=np.mean(duration,1)
    max_duration=[0 for t in range(x)]
    min_duration=[0 for t in range(x)]
    for t in range(x):
        max_duration[t]=np.max(duration[t])
        min_duration[t]=np.min(duration[t])
    max_duration2=np.max(max_duration)
    min_duration2=np.min(min_duration)
    #print max_duration2,min_duration2
    buckets=[0 for i in range(num_buckets)]

    interval=(max_duration2-min_duration2)/num_buckets
    for i in range(num_buckets):
        buckets[i]=buckets[i]+i*interval
    #print "buckets is",buckets,"shape_buckets is",np.shape(buckets)
    len_yinsu=[0 for t in range(x)]
    for t in range(x):
        len_yinsu[t]=len(yinsu[t])
    duration_buckets=[[0 for i in range(len_yinsu[t])]for t in range(x)]
    for t in range(x):
        for i in range(len_yinsu[t]):
            for k in range(num_buckets):
                if duration[t][i]>=buckets[num_buckets-k-1]:
                    
                    duration_buckets[t][i]=num_buckets-k#jia le 1
                    break
    print "duration_buckets is",duration_buckets,"shape_duration_buckets is",np.shape(duration_buckets)
    max_yinsu_seq_len=np.max(len_yinsu)
    for t in range(x):
        for i in range(max_yinsu_seq_len-len(yinsu[t])):
            yinsu[t].append(0)
            duration_buckets[t].append(0)
            
    audio_label=[0 for i in range(len(audio_label_pre2))]
    for i in range(len(audio_label_pre2)):
        audio_label[i]=audio_label_pre2[i][0][0]
    return audio_label,yinsu,len_yinsu,duration_buckets,index_model4,voiced_target

                                                        #print "yinsu is ",yinsu,"yinsu_seq_len is ",yinsu_seq_len,"duration_buckets is ",duration_buckets
                                                        #np.save('yinsu_train.npy', np.array(yinsu))
                                                        #np.save('yinsu_seq_len_train.npy', np.array(len_yinsu))
                                                        #np.save('duration_buckets_train.npy', np.array(duration_buckets))
                                                        #np.save('index_model4_train.npy', np.array(index_model4))
                                                        #np.save('voiced_target_train.npy', np.array(voiced_target))
                                                        #this one 
                                                        #np.save('yinsu.npy', np.array(yinsu))
                                                        #np.save('yinsu_seq_len.npy', np.array(len_yinsu))
                                                        #np.save('duration_buckets.npy', np.array(duration_buckets))
                                                        #np.save('index_model4.npy', np.array(index_model4))
                                                        #np.save('voiced_target.npy', np.array(voiced_target))

                                                        #print "voiced_target:",type(voiced_target),voiced_target
                                                        #get_ipython().magic(u'store yinsu')
                                                        #get_ipython().magic(u'store yinsu_seq_len')
                                                        #get_ipython().magic(u'store duration_buckets')
                                                        #get_ipython().magic(u'store index_model4')
                                                        #get_ipython().magic(u'store voiced_target')

                                                        #2,30
                                                        #coord.request_stop()
                                                        #coord.join(threads)



                                                        # In[ ]:




                                                        # In[ ]:'''
                                                        