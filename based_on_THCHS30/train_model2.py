
# coding: utf-8

# In[1]:

import sys
sys.path.append("..")
sys.path.append("/home/VCTK-Corpus/wav48/p225")


# In[2]:

import os
#from models.segmentation_model import SegmentationModel
from segmentation_model22 import SegmentationModel
import tensorflow as tf
import numpy as np
from scipy.io import wavfile
import fnmatch
import librosa
import itertools 
import re 
#from _2model2 import *
from tqdm import *
from _2_model2_thread import *
from postprocess import *
import time

#melM=np.load('melM.npy')
#melM_dev=np.load('melM_dev.npy')
#file_len_train=np.load('file_len_train.npy')
#file_len_dev=np.load('file_len_dev.npy')

#melM=melM[0:10]

# # Define parametes

# In[ ]:
#Use following commands on multiple terminals to start the training.
#python example.py --ps_hosts=10.200.12.230:2222 --worker_hosts=10.200.12.230:2224,10.200.12.230:2225 --job_name=ps --task_index=0
#python example.py --ps_hosts=10.200.12.230:2222 --worker_hosts=10.200.12.230:2224,10.200.12.230:2225 --job_name=worker --task_index=0
#python example.py --ps_hosts=10.200.12.230:2222 --worker_hosts=10.200.12.230:2224,10.200.12.230:2225 --job_name=worker --task_index=1


train_parameters = {
    "lr": 0.0001,#0.001
    "decay_steps": 500,#400,
    "decay_rate": 0.95,#0.95,
    "dropout_prob": 0.85,#0.05
}

model_parameters = {
    "speaker_embedding_size": 37,#37
    "num_conv_layers": 4,#2,
    "conv_num_filters": 64,#8,
    "conv_kernel_size": [9,5],#[2, 2],
    "num_bidirectional_units":1024,#16 ,#16
    "num_bidirectional_layers": 4,#2,
}

output_vocab_size=219 #47525
num_speakers = 37 #10
num_steps = 2
save_energy = 2
n_mels = 40
num_beams = 5
batch_size=8
batch_size2=8
sample_rate=16000
audio_dir="/home/data_thchs30/train"
phoneme_dir="/home/deepvoice/deepvoice2_chinese/data/train/phonemes"
audio_dir2="/home/data_thchs30/test"
phoneme_dir2="/home/deepvoice/deepvoice2_chinese/data/test/phonemes"
# # Train model

# In[ ]:





with tf.Session() as sess:

    model = SegmentationModel(
        output_vocab_size, num_speakers, model_parameters,
    )
    
    frequencies = tf.placeholder(tf.float32, [None, None, n_mels])
    frequencies_seq_len = tf.placeholder(tf.int32, [None])
    speaker_ids = tf.placeholder(tf.int32, [None])
    phonemes = tf.sparse_placeholder(tf.int32)
    
   

    #The tuple of concatenated tensors that was dequeued.
    train_op_tf, loss_tf, global_step_tf, summary_tf = model.build_train_operations(
        frequencies, frequencies_seq_len,
        speaker_ids,
        phonemes,
        train_parameters
    )
    
    predictor_frequencies = tf.placeholder(tf.float32, [None, None, n_mels])
    predictor_frequencies_seq_len = tf.placeholder(tf.int32, [None])
    predictor_speaker_ids = tf.placeholder(tf.int32, [None])
    
    greedy_predictor_tf = model.build_greedy_predictor(
        predictor_frequencies, predictor_frequencies_seq_len, predictor_speaker_ids, True
    )
    
    encoder_output_tf,beam_predictor_tf = model.build_beam_search_predictor(
        predictor_frequencies, predictor_frequencies_seq_len, predictor_speaker_ids,
        num_beams, True
    )

    train_writer = tf.summary.FileWriter('log/train_grapheme_to_phoneme_model_notebook/train', sess.graph)
    
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    reader=AudioPhonemeReader(audio_dir,
                              phoneme_dir,
                              sample_rate,
                              coord)

    audio_batch=reader.dequeue(batch_size)
    file_len_batch=reader.dequeue_wl(batch_size)
    phoneme_batch = reader.dequeue_p(batch_size)
    
    
    #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #reader.start_threads(sess)
    
    reader2=AudioPhonemeReader(audio_dir2,
                              phoneme_dir2,
                              sample_rate,
                              coord)

    audio_batch_pre=reader2.dequeue(batch_size2)
    file_len_batch_pre=reader2.dequeue_wl(batch_size2)
    audio_label_pre=reader2.dequeue_label(batch_size2)
    
    
    #threads2 = tf.train.start_queue_runners(sess=sess, coord=coord)
    #reader2.start_threads(sess)
    

    saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=3)
    #get_ipython().magic(u'store -r target_indices')
    #get_ipython().magic(u'store -r target_values')
    #get_ipython().magic(u'store -r dense_shape')
    
    #print np.shape(melM)
    #print np.shape(np.array([file_len[i]for i in range(files_len)],dtype=np.float))
    #print np.shape(files_len)
    #print np.shape()
    #saver.restore(sess, '../weights/train_grapheme_to_phoneme_model_notebook/-400')
    graph = tf.get_default_graph()
    train=1
    if train==0:
        kk=0
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        reader.start_threads(sess)
        
        for _ in tqdm(xrange(num_steps)):
            kk=kk+1
            target_indices_batch=[]#
            target_values_batch=[]
            phoneme_len=[]
            k=0
            #for i in tqdm(range(n/batch_size)):
            
            audio_batch2=sess.run(audio_batch)
            file_len_batch2=sess.run(file_len_batch)
            phoneme_batch2=sess.run(phoneme_batch)
            #print 'audio_batch2',np.shape(audio_batch2),audio_batch2
            #print 'file_len_batch2',np.shape(file_len_batch2),file_len_batch2
            #print 'phoneme_batch2',np.shape(phoneme_batch2),phoneme_batch2
            #print kk
            #phoneme_batch2=list(phoneme_batch2)
            phoneme_batch3=[[[ 0 for k in range(1)]for j in range(len(phoneme_batch2[i]))]for i in range(batch_size)]
            for i in range(batch_size):
                for j in range(len(phoneme_batch2[i])):
                    
                    phoneme_batch3[i][j][0]=phoneme_batch2[i][j][0]

            
            for i in range(batch_size):
                for j in range(len(phoneme_batch2[i])):
                    if phoneme_batch3[i][-1][0]==0:
                        phoneme_batch3[i]=phoneme_batch3[i][:-1]
                    else:
                        phoneme_len.append(len(phoneme_batch3[i]))
                        break
                    
            file_len_batch3=[]
            for i in range(batch_size):
                file_len_batch3.append(file_len_batch2[i][0][0])
            #target start
            for i in range(batch_size):
                for j in range(0,len(phoneme_batch3[i])):
                    indices_phoneme=phoneme_batch3[i][j][0]
                    
                    target_indices_batch.append([i,j])
                    target_values_batch.append(indices_phoneme)
                    

                    #if i>1:
                    #target_values[k].append(label[i-2]*(single_size-1)+indices_phoneme)
                    #target_indices[k].pop(-1)
            max_phoneme_len=max(phoneme_len)       
            dense_shape_batch=[batch_size,max_phoneme_len]
            #print 'target_indices_batch',len(target_indices_batch),target_indices_batch
            #print 'target_values_batch',len(target_values_batch),target_values_batch
            #print 'dense_shape_batch',dense_shape_batch
            
            out = sess.run([
                train_op_tf,
                global_step_tf,
                loss_tf,
                summary_tf],
                feed_dict={
                    frequencies:audio_batch2,
                    frequencies_seq_len: file_len_batch3,
                    speaker_ids: 1 * np.ones(batch_size),
                    phonemes:(target_indices_batch,target_values_batch, dense_shape_batch)
                })

            _, global_step, loss, summary  = out

            print global_step
            print loss

            train_writer.add_summary(summary, global_step)

                    # detect gradient explosion
            if loss > 1e8 and global_step > 500:
                print('loss exploded')
                break

            if global_step % save_energy == 0 and global_step != 0:

                print('saving weights')
                if not os.path.exists('weights/train_grapheme_to_phoneme_model_notebook/'):
                    os.makedirs('weights/train_grapheme_to_phoneme_model_notebook/')
                saver.save(sess, 'weights/train_grapheme_to_phoneme_model_notebook/model2',
                           global_step=global_step)

        coord.request_stop()
        coord.join(threads)
    else:
        threads2 = tf.train.start_queue_runners(sess=sess, coord=coord)
        reader2.start_threads2(sess)
        #batch_size=10
        saver.restore(sess, 'weights/train_grapheme_to_phoneme_model_notebook/model2-2')
        graph = tf.get_default_graph()
        
        encoder_output=[[]for i in range(10000/batch_size2)]
        yinsu=[[]for i in range(10000/batch_size2)]
        len_yinsu=[[]for i in range(10000/batch_size2)]
        duration_buckets=[[]for i in range(10000/batch_size2)]
        index_model4=[[]for i in range(10000/batch_size2)]
        voiced_target=[[]for i in range(10000/batch_size2)]
        audio_label=[[] for i in range(10000/batch_size2)]
        for _ in tqdm(range(10000/batch_size2)):
            audio_batch_pre2=sess.run(audio_batch_pre)
            file_len_batch_pre2=sess.run(file_len_batch_pre)
            file_len_batch_pre3=[]
            audio_label_pre2=sess.run(audio_label_pre)
            print 'audio_label_pre2',np.shape(audio_label_pre2),audio_label_pre2
            for i in range(batch_size2):
                file_len_batch_pre3.append(file_len_batch_pre2[i][0][0])
        #batch_size=2
            
            
            encoder_output[i],_=sess.run([encoder_output_tf,beam_predictor_tf],
                                         feed_dict={
                                             predictor_frequencies:audio_batch_pre2,
                                             predictor_frequencies_seq_len:file_len_batch_pre3,
                                             predictor_speaker_ids:1 * np.ones((batch_size2))
                                         })
            #print np.shape(encoder_output),encoder_output,type(encoder_output[0]),np.shape(encoder_output[0])
            ###############
            ###############
            
            audio_label[i],yinsu[i],len_yinsu[i],duration_buckets[i],index_model4[i],voiced_target[i]=postprocess(encoder_output[i],audio_label_pre2,beamsize=num_beams)
            #print yinsu[0],np.shape(yinsu[0]),type(yinsu[0])
        coord.request_stop()
        coord.join(threads)
        


