
# coding: utf-8

# In[1]:

import sys
sys.path.append("..")


# In[2]:

import os
from models.duration_model import DurationModel
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
    "phonemes_embedding_size": 219,#76
    "speaker_embedding_size": 16,#16
    "num_dense_layers": 2,
    "dense_layers_units": 16,
    "num_bidirectional_layers": 2,
    "num_bidirectional_units": 16
}

input_vocab_size = 219#76 
num_speakers = 10
num_buckets =11#10+1

num_steps = 2

save_energy = 2


# # Train model

# In[4]:


#get_ipython().magic(u'store -r yinsu')
#get_ipython().magic(u'store -r yinsu_seq_len')
#get_ipython().magic(u'store -r duration_buckets')
#get_ipython().magic(u'store -r files_len')
files_len=2
file_len_train=np.load('file_len_train.npy')
yinsu=np.load('yinsu.npy')
yinsu_seq_len=np.load('yinsu_seq_len.npy')
duration_buckets=np.load('duration_buckets.npy')
        


#2,8
#[8,6]
#2,8
print yinsu
print yinsu_seq_len
print duration_buckets


with tf.Session() as sess:
    phonemes = tf.placeholder(tf.int32, [None, None])
    phonemes_seq_len = tf.placeholder(tf.int32, [None])
    speaker_ids = tf.placeholder(tf.int32, [None])
    durations = tf.placeholder(tf.int32, [None, None])
    
    prediction_phonemes = tf.placeholder(tf.int32, [None, None])
    prediction_phonemes_seq_len = tf.placeholder(tf.int32, [None])
    prediction_speaker_ids = tf.placeholder(tf.int32, [None])
    prediction_durations=tf.placeholder(tf.int32,[None,None])
    model = DurationModel(
        input_vocab_size, num_speakers,
        num_buckets, model_parameters
    )
    
    train_op_tf, loss_tf, global_step_tf, summary_tf, logits_tf, transition_params_tf = model.build_train_operations(
        phonemes, phonemes_seq_len, speaker_ids, durations, train_parameters
    )
    
    viterbi= model.build_prediction_logits(
        prediction_phonemes, prediction_phonemes_seq_len, prediction_speaker_ids,prediction_durations
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
                train_op_tf,
                loss_tf,
                global_step_tf,
                summary_tf,
                logits_tf,
                transition_params_tf
            ], feed_dict={
                phonemes: np.array(yinsu),
                phonemes_seq_len: np.array(yinsu_seq_len),
                speaker_ids: 2 * np.ones((files_len)),
                durations: np.array(duration_buckets)
#                 phonemes: np.ones((2, 200)),#t(2,30);np.ones((2, 200))
#                 phonemes_seq_len: np.ones(2),#t_seq_len,
#                 speaker_ids: 2 * np.ones((2)),
#                 durations: np.zeros((2, 200))#model2_to_model3
            })
            _, loss, global_step, summary, logits, transition_params  = out

            print "global_step:",global_step
            print "logits:",np.shape(logits),logits
            print "transition_params:",np.shape(transition_params),transition_params
            print "loss:",np.shape(loss),loss



            train_writer.add_summary(summary, global_step)

            # detect gradient explosion
            if loss > 1e8 and global_step > 500:
                print('loss exploded')
                break

            if global_step % save_energy == 0 and global_step != 0:

                print('saving weights')
                if not os.path.exists('../weights/train_grapheme_to_phoneme_model_notebook/'):
                    os.makedirs('../weights/train_grapheme_to_phoneme_model_notebook/')
                saver.save(sess, '../weights/train_grapheme_to_phoneme_model_notebook/model3', global_step=global_step)

        coord.request_stop()
        coord.join(threads)
    else:
        saver.restore(sess, '../weights/train_grapheme_to_phoneme_model_notebook/model3-2')
        graph = tf.get_default_graph()
        
        
        viterbi=sess.run(viterbi ,feed_dict={
                prediction_phonemes: yinsu,#t(2,30);np.ones((2, 200))
                prediction_phonemes_seq_len: yinsu_seq_len,#t_seq_len,
                prediction_speaker_ids: 2 * np.ones((2)),
                prediction_durations: duration_buckets#model2_to_model3
        })
        print np.shape(viterbi),viterbi
        coord.request_stop()
        coord.join(threads)
        


# In[ ]:



