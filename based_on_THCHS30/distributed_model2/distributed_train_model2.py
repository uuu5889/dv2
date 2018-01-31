import sys
import os
from segmentation_model2 import SegmentationModel
import tensorflow as tf
import model2_input
import pickle
import numpy as np
import time

#Use following commands on multiple terminals to start the training.
#python distributed_train_model22.py --ps_hosts=10.200.12.232:2222 --worker_hosts=10.200.12.230:2222,10.200.12.230:2223,10.200.12.230:2224,10.200.12.230:2225,10.200.12.231:2222,10.200.12.231:2223,10.200.12.231:2224,10.200.12.231:2225 --job_name=ps --task_index=0
#python example.py --ps_hosts=10.200.12.230:2222 --worker_hosts=10.200.12.230:2224,10.200.12.230:2225 --job_name=worker --task_index=0
#python example.py --ps_hosts=10.200.12.230:2222 --worker_hosts=10.200.12.230:2224,10.200.12.230:2225 --job_name=worker --task_index=1

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('steps_to_validate', 40,
                     'Steps to validate and print loss')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")


train_parameters = {
    "lr": 0.0001, #0.001,
    "decay_steps": 1000,#10000
    "decay_rate": 0.95,
    "dropout_prob": 0.85
}

model_parameters = {
    "speaker_embedding_size": 37, #38,
    "num_conv_layers": 4, #2
    "conv_num_filters": 64, #5,
    "conv_kernel_size": [9,5], #[3, 3],
    "num_bidirectional_units": 1024,#256
    "num_bidirectional_layers": 4,#2
}

output_vocab_size= 1422
num_speakers = 37
num_steps = 100000
save_energy = 2
n_mels = 40#40 20
num_beams = 5
batch_size=8

def main(_):
  
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)
    
    

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
                        worker_device="/job:worker/task:%d" % FLAGS.task_index,
                        cluster=cluster)):
            
            train_inputs, dev_inputs, indexes = model2_input.get_inputs()
            model = SegmentationModel(
                output_vocab_size, num_speakers, model_parameters,
            )

            frequencies = tf.placeholder(tf.float32, [None, None, n_mels])
            frequencies_seq_len = tf.placeholder(tf.int32, [None])
            speaker_ids = tf.placeholder(tf.int32, [None])
            phonemes = tf.sparse_placeholder(tf.int32)


            train_op_tf, loss_tf, global_step_tf, summary_tf = model.build_train_operations(
                frequencies, frequencies_seq_len,
                speaker_ids,
                phonemes,
                train_parameters
            )
            
            

            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=3)

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                init_op=init_op,
                                global_step=global_step_tf,
                                logdir='./checkpoints/')
        
        with sv.managed_session(server.target) as sess: 
            train_writer = tf.summary.FileWriter('./log/train_model_segmentation/train', sess.graph)
            
            min_loss = 100000.
            step=0
            while step<100000:
                start = time.time()
                # randomly pick an sample
                #add begin
                #add begin
                sample =[]
                freq=[]
                freq_seq_len=[]
                spk_ids=[]
                target_indices =[]
                target_values =[]
                len_target_indices=[]
                for i in range(batch_size):
                    picked = np.random.randint(750)
                    curr_sample=train_inputs[picked]
                    sample.append(curr_sample)
                    n1=np.random.randint(len(curr_sample["frequencies"]))
                    target_values=target_values+curr_sample["target_values"][n1]
                    len_target_indices.append(len(curr_sample["target_values"][n1]))

                    for j in range(0,len(curr_sample["target_values"][n1])):

                        target_indices=target_indices+[[i,j]]
                    print 'target_values',np.shape(np.array(curr_sample["target_values"]))
                            #target_values = curr_sample["target_values"]
                            #dense_shape = curr_sample["dense_shape"]


                            #add end

                    freq=freq+[curr_sample["frequencies"][n1]]
                    freq_seq_len=freq_seq_len+[curr_sample["frequencies_seq_len"][n1]]
                    spk_ids=spk_ids+[curr_sample["speaker_ids"][n1]]      
                max_target_indices_len=max(len_target_indices)       
                dense_shape=[batch_size,max_target_indices_len]
                #print 'target_indices',target_indices
                #print 'target_values',np.shape(np.array(target_values))#,target_values
                #print 'dense_shape',dense_shape
                    #print 'freq',np.shape(np.array(freq)),freq
                    #print 'freq_seq_len',freq_seq_len
                    #print 'spk_ids',spk_ids
                    #add end
                           

                out = sess.run([
                    train_op_tf,
                    global_step_tf,
                    loss_tf,
                    summary_tf
                ], feed_dict={
                    frequencies: freq, #np.random.rand(2, 10, 20),
                    frequencies_seq_len: freq_seq_len, #8* np.ones(2),
                    speaker_ids: spk_ids, #2 * np.ones((2)),
                    phonemes:(target_indices,target_values, dense_shape)
                })

                train_op, step, loss, summary  = out
                end = time.time()
                #print "Step:",step,". Training loss:",loss
                train_writer.add_summary(summary, step)
                #print("step: %d, time: %f" %(step,end-start))
                if step%10 == 0:
                    
                    print "Step:",step,". Training loss:",loss
                    # validation
                    #picked = np.random.randint(535)
                    #curr_sample = dev_inputs[0]
                    
                    #add begin
                    #add begin
                    sample =[]
                    freq=[]
                    freq_seq_len=[]
                    spk_ids=[]
                    target_indices =[]
                    target_values =[]
                    len_target_indices=[]
                    for i in range(batch_size):
                        picked = np.random.randint(535)
                        curr_sample=dev_inputs[picked]
                        sample.append(curr_sample)
                        n1=np.random.randint(len(curr_sample["frequencies"]))
                        target_values=target_values+curr_sample["target_values"][n1]
                        len_target_indices.append(len(curr_sample["target_values"][n1]))

                        for j in range(0,len(curr_sample["target_values"][n1])):

                            target_indices=target_indices+[[i,j]]
                        #print 'target_values',np.shape(np.array(curr_sample["target_values"]))
                                #target_values = curr_sample["target_values"]
                                #dense_shape = curr_sample["dense_shape"]


                                #add end

                        freq=freq+[curr_sample["frequencies"][n1]]
                        freq_seq_len=freq_seq_len+[curr_sample["frequencies_seq_len"][n1]]
                        spk_ids=spk_ids+[curr_sample["speaker_ids"][n1]]      
                    max_target_indices_len=max(len_target_indices)       
                    dense_shape=[batch_size,max_target_indices_len]
                    print 'data preparing finished'
                    '''print 'target_indices',target_indices
                    print 'target_values',np.shape(np.array(target_values))#,target_values
                    print 'dense_shape',dense_shape'''
                        #print 'freq',np.shape(np.array(freq)),freq
                        #print 'freq_seq_len',freq_seq_len
                        #print 'spk_ids',spk_ids
                        #add end
                    #add end
                    '''target_indices = curr_sample["target_indices"]
                    target_values = curr_sample["target_values"]
                    dense_shape = curr_sample["dense_shape"]
                    freq = curr_sample["frequencies"]
                    freq_seq_len = curr_sample["frequencies_seq_len"]
                    spk_ids = curr_sample["speaker_ids"]'''     

                    out = sess.run([
                        loss_tf,
                    ], feed_dict={
                        frequencies: freq, 
                        frequencies_seq_len: freq_seq_len, 
                        speaker_ids: spk_ids,
                        phonemes:(target_indices,target_values, dense_shape)
                    })
                    
                    curr_loss = out[0]
                    print "Validation loss:",out[0]

                    

                    if curr_loss < min_loss:
                        print('saving weights...')
                        if not os.path.exists('../weights/train_grapheme_to_phoneme_model_notebook/'):
                            os.makedirs('../weights/train_grapheme_to_phoneme_model_notebook/')
                            saver.save(sess, '../weights/train_grapheme_to_phoneme_model_notebook/', global_step=step)
                            min_loss = curr_loss
            
        sv.stop()
        print("Done.")
        
if __name__ == "__main__":
  tf.app.run()
