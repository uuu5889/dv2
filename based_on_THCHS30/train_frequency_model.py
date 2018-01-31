
# coding: utf-8

# In[1]:

import sys
sys.path.append("..")


# In[2]:

import os
from models.frequency_model import FrequencyModel
import tensorflow as tf
import numpy as np


# # Define parametes

# In[3]:

train_parameters = {
    "lr": 0.0001,
    "decay_steps": 300,
    "decay_rate": 0.9,
    "dropout_prob": 0.2
}

model_parameters = {
    "phonemes_embedding_size": 219,#76,
    "speaker_embedding_size": 16,
    "num_bidirectional_layers": 2,
    "num_bidirectional_units": 16,
    "conv_widths": [2, 2],
    "output_dimension": 2    
}

input_vocab_size = 219#76
num_speakers = 10

num_steps = 2

save_energy = 2


# # Train model

# In[4]:

#get_ipython().magic(u'store -r index_model4')
#get_ipython().magic(u'store -r voiced_target')
#get_ipython().magic(u'store -r duration_buckets')
#get_ipython().magic(u'store -r f1')
index_model4=np.load('index_model4.npy')
voiced_target=np.load('voiced_target.npy')
file_len=np.load('file_len_train.npy')
file_len_dev=np.load('file_len_dev.npy')
f1=np.load('f1.npy')
files_len=2

#print "index_model4",type(index_model4),index_model4
#print "voiced_target2",type(voiced_target),voiced_target

voiced_target1=voiced_target
#file_len=[193,381]
with tf.Session() as sess:
    phonemes = tf.placeholder(tf.int32, [None, None])
    phonemes_seq_len = tf.placeholder(tf.int32, [None])
    speaker_ids = tf.placeholder(tf.int32, [None])
    voiced_target = tf.placeholder(tf.int32, [None, None])
    frequency_target = tf.placeholder(tf.float32, [None, None])
    
    prediction_phonemes = tf.placeholder(tf.int32, [None, None])
    prediction_phonemes_seq_len = tf.placeholder(tf.int32, [None])
    prediction_speaker_ids = tf.placeholder(tf.int32, [None])

    model = FrequencyModel(
        input_vocab_size, num_speakers,
        model_parameters
    )
    
    voice_tf,fzero_tf,train_op_tf, loss_tf, global_step_tf, summary_tf = model.build_train_operations(
        phonemes, phonemes_seq_len, speaker_ids, voiced_target, frequency_target, train_parameters
    )
    
    prediction_voiced_tf, prediction_frequencies_tf = model.build_prediction(
        prediction_phonemes, prediction_phonemes_seq_len, prediction_speaker_ids, True
    )

    train_writer = tf.summary.FileWriter('../log/train_grapheme_to_phoneme_model_notebook/train', sess.graph)
    
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=3)
    train=1
    if train==0:
    
    
        for _ in xrange(num_steps):
            
            out = sess.run([
                voice_tf,
                fzero_tf,
                train_op_tf,
                loss_tf,
                global_step_tf,
                summary_tf
            ], feed_dict={
                phonemes:index_model4, #index_model4,
                phonemes_seq_len:np.array([file_len_dev[i] for i in range(files_len)],dtype=np.float32),
                speaker_ids: 2 * np.ones((files_len)),
                voiced_target: voiced_target1, #voiced_target,
                frequency_target: f1
            })
            voice,f_zero,_, loss, global_step, summary  = out

            print global_step
            print loss


            train_writer.add_summary(summary, global_step)

            # detect gradient explosion
            if loss > 1e8 and global_step > 500:
                print('loss exploded')
                break

            if global_step % save_energy == 0 and global_step != 0:

                print('saving weights')
                print "voice",voice
                print "voice_shape",np.shape(voice)
                print "f_zero",f_zero
                print "shape_f_zero",np.shape(f_zero)
                if not os.path.exists('../weights/train_grapheme_to_phoneme_model_notebook/'):
                    os.makedirs('../weights/train_grapheme_to_phoneme_model_notebook/')
                saver.save(sess, '../weights/train_grapheme_to_phoneme_model_notebook/model4', global_step=global_step)

        coord.request_stop()
        coord.join(threads)
    else:
        saver.restore(sess, '../weights/train_grapheme_to_phoneme_model_notebook/model4-2')
        graph = tf.get_default_graph()
        
        
        prediction_voiced, prediction_frequencies=sess.run([prediction_voiced_tf,
                                                           prediction_frequencies_tf] ,feed_dict={
                                                               prediction_phonemes:index_model4,
                                                               prediction_phonemes_seq_len:np.array([file_len_dev[i] for i in range(2)],dtype=np.float32),
                                                               prediction_speaker_ids: 2 * np.ones((2)),
    
                                                           })
        print 'shape_prediction_voiced:',np.shape(prediction_voiced)
        print 'prediction_voiced:',prediction_voiced
        print 'shape_prediction_frequencies:',np.shape(prediction_frequencies)
        print 'prediction_frequencies:',prediction_frequencies
        max_prediction_voiced=[[0 for j in range(len(prediction_voiced[i]))]for i in range(len(prediction_voiced))]
        for i in range(len(prediction_voiced)):
            for j in range(len(prediction_voiced[i])):
                max_prediction_voiced[i][j]=np.argsort(prediction_voiced[i][j])[0]#less to greater
        max_prediction_voiced=np.squeeze(np.array(max_prediction_voiced))
        print 'shape_max_prediction_voiced:',np.shape(max_prediction_voiced)
        print 'max_prediction_voiced:',max_prediction_voiced
        np.save('prediction_voiced.npy',np.array(prediction_voiced))#
        np.save('prediction_frequencies.npy',np.array(prediction_frequencies))#
        #get_ipython().magic(u'store prediction_voiced')
        #get_ipython().magic(u'store prediction_frequencies')
        coord.request_stop()
        coord.join(threads)


# In[ ]:




# In[ ]:



