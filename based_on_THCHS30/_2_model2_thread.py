import fnmatch
import os
import random
import re
import threading
import csv
import librosa
import numpy as np
import tensorflow as tf
from prepare_chinese_data import *
FILE_PATTERN = r'p([0-9]+)_([0-9]+)\.wav'

def get_category_cardinality(files):
    '''return the min id and the max id of all the speakers'''
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]
        id, recording_id = [int(id_) for id_ in matches]
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id
    return min_id, max_id


def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def find_files(audio_dir, phoneme_dir):
    '''Recursively finds all files matching the pattern.'''
    audio_files = []
    phoneme_files = []
    for _,dirnames, filenames in os.walk(audio_dir):
        for filename in fnmatch.filter(filenames, '*.wav'):
            audio_files.append(os.path.join(audio_dir, filename))
            #filename = filename[:-3]+"txt"
            #phoneme_files.append(os.path.join(phoneme_dir, filename))
    for _,dirnames2, filenames2 in os.walk(phoneme_dir):
        for filename in fnmatch.filter(filenames2,'*.txt'):
            phoneme_files.append(os.path.join(phoneme_dir, filename))
    return audio_files, phoneme_files

def find_files2(audio_dir):
    '''Recursively finds all files matching the pattern.'''
    audio_files = []
    for _,dirnames, filenames in os.walk(audio_dir):
        for filename in fnmatch.filter(filenames, '*.wav'):
            audio_files.append(os.path.join(audio_dir, filename))
            #filename = filename[:-3]+"txt"

    return audio_files

def load_audio_info(audio_dir, phoneme_dir, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    audio_files, phoneme_files = find_files(audio_dir,phoneme_dir)
    files = zip(audio_files, phoneme_files)
    id_reg_exp = re.compile(FILE_PATTERN)
    print("files length: {}".format(len(files)))
    randomized_files = randomize_files(files)
    #for filename, phoneme_filename in files:
    for filename, phoneme_filename in randomized_files:
        ids = id_reg_exp.findall(filename)                
        if not ids:
            # The file name does not match the pattern containing ids, so
            # there is no id.
            category_id = None
        else:
            # The file name matches the pattern for containing ids.
            category_id = int(ids[0][0])
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        #audio = audio.reshape(-1, 1)
        phoneme = load_phoneme(phoneme_filename)
        phoneme = np.resize(phoneme,(-1,1))
        yield audio, filename, category_id, phoneme
        
def load_audio_info2(audio_dir, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    audio_files= find_files2(audio_dir)
    #files = zip(audio_files, phoneme_files)
    id_reg_exp = re.compile(FILE_PATTERN)
    print("files length: {}".format(len(audio_files)))
    #randomized_files = randomize_files(files)
    k=1
    for filename in audio_files:
        ids = id_reg_exp.findall(filename)                
        if not ids:
            # The file name does not match the pattern containing ids, so
            # there is no id.
            category_id = None
        else:
            # The file name matches the pattern for containing ids.
            category_id = int(ids[0][0])
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        #audio = audio.reshape(-1, 1)
        audio_label=[[k]]
        k=k+1
        #print ("audio_label: {}".format(audio_label))
        yield audio, audio_label,filename, category_id

def load_phoneme(filename):
    for line in open(filename,'r').readlines():
    #if n > 9:
    #    break
    
        len_line=len(list(line.strip().split(' ')))
        label=[]
        for i in range(1,len_line):
            indices_phoneme=char2id[list(line.strip().split(' '))[i]]
            label.append(indices_phoneme)
        
    return label

    
def trim_silence(audio, threshold, frame_length=2048):
    '''Removes silence at the beginning and end of a sample.'''
    if audio.size < frame_length:
        frame_length = audio.size
    energy = librosa.feature.rmse(audio, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]
    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

def not_all_have_id(files):
    ''' Return true iff any of the filenames does not conform to the pattern
        we require for determining the category id.'''
    id_reg_exp = re.compile(FILE_PATTERN)
    for file in files:
        ids = id_reg_exp.findall(file)
        if not ids:
            return True
    return False


class AudioPhonemeReader(object):    
    def __init__(self,
                 audio_dir,
                 phoneme_dir,
                 sample_rate,
                 coord,
                 sample_size=None,
                 silence_threshold=0.3,
                 gc_enabled=None,
                 queue_size=10):
        self.audio_dir = audio_dir
        self.phoneme_dir = phoneme_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.silence_threshold = silence_threshold
        self.gc_enabled = gc_enabled
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,40])
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[(None, 40)])
        # the inputs' shape may vary along the given dimension, and dequeue_many will pad the given dimension
        # with zeros up to the maximum shape of all elements in the given batch.
        self.enqueue = self.queue.enqueue([self.sample_placeholder])
        
        self.p_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.wl_placeholder = tf.placeholder(dtype=tf.float32, shape=None)

        self.p_queue = tf.PaddingFIFOQueue(queue_size, ['float32'], shapes=[(None,1)])
        self.wl_queue = tf.PaddingFIFOQueue(queue_size, ['float32'], shapes=[(None,1)])

        self.p_enqueue = self.p_queue.enqueue([self.p_placeholder])
        self.wl_enqueue = self.wl_queue.enqueue([self.wl_placeholder])
        self.label_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.label_queue = tf.PaddingFIFOQueue(queue_size, ['float32'], shapes=[(None,1)])
        self.label_enqueue = self.label_queue.enqueue([self.label_placeholder])
        
        
    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output
    
    def dequeue_p(self, num_elements):
        return self.p_queue.dequeue_many(num_elements)
    
    def dequeue_wl(self, num_elements):
        return self.wl_queue.dequeue_many(num_elements)
    
    def dequeue_label(self, num_elements):
        return self.label_queue.dequeue_many(num_elements)
       
    def dequeue_gc(self, num_elements):
        return self.gc_queue.dequeue_many(num_elements)

    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_audio_info(self.audio_dir,self.phoneme_dir,self.sample_rate)
            for audio, filename, category_id, phoneme in iterator:
                if self.coord.should_stop():
                    stop = True
                    break
                if self.silence_threshold is not None:
                    # Remove silence
                    audio, _ = librosa.effects.trim(audio, top_db=10)
                    
                    #audio = trim_silence(audio[:, 0], self.silence_threshold)
                    #audio = audio.reshape(-1, 1)
                    if audio.size == 0:
                        print("Warning: {} was ignored as it contains only "
                              "silence. Consider decreasing trim_silence "
                              "threshold, or adjust volume of the audio."
                              .format(filename))
                #start
                #melM=[]
        
                melM=librosa.feature.mfcc(audio,sr=self.sample_rate,n_mfcc=40)
                

                #files_len=len(files)
                #files_len=len(files)
                #print melM
                #print len(melM[0])
                #print len(melM[0][0])

                #file_len = np.array([len(melM[i][0]) for i in range(files_len)], dtype=np.int32)
                print np.shape(melM)
                file_len = [[len(melM[0])]]#1,20,740
                #file_len=np.resize(file_len,(-1,1))
                #np.save('file_len_test.npy',file_len)

                #max_file_len = np.amax(file_len)
                #print max_file_len
                #melM = [[list(np.pad(np.array(melM[i][j], dtype=np.float32),
                #                     (0, max_file_len - file_len[i]), 
                #                     'constant', constant_values=(0, 0)))
                #          for j in range(20)] for i in range(files_len)]

                #         for j in range(20)] for i in range(files_len)]

                melM=np.array(melM).transpose(1,0)
                
                
                #audio = np.pad(audio, [[self.receptive_field, 0], [0, 0]],
                #              'constant')
                # receptive field length is 5117
                # audios with length smaller than 5117 will be padded to 5117
                if self.sample_size:
                    # Cut samples into pieces of size receptive_field +
                    # sample_size with receptive_field overlap
                    while len(audio) > self.receptive_field:
                        # the maximum length limitation
                        piece = audio[:(self.receptive_field +
                                        self.sample_size), :]
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece})
                        sess.run(self.p_enqueue,
                                 feed_dict={self.p_placeholder: phoneme})
                        audio = audio[self.sample_size:, :]
                        if self.gc_enabled:
                            sess.run(self.gc_enqueue, feed_dict={
                                self.id_placeholder: category_id})
                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: melM})
                    sess.run(self.wl_enqueue,
                             feed_dict={self.wl_placeholder:file_len})
                    sess.run(self.p_enqueue,
                             feed_dict={self.p_placeholder:phoneme})
                    if self.gc_enabled:
                        sess.run(self.gc_enqueue,
                                 feed_dict={self.id_placeholder: category_id})
                        
    
    def thread_main2(self, sess):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_audio_info2(self.audio_dir,self.sample_rate)
            for audio,audio_label, filename, category_id in iterator:
                #print ("audio_label: {}".format(audio_label))

                if self.coord.should_stop():
                    stop = True
                    break
                if self.silence_threshold is not None:
                    # Remove silence
                    audio, _ = librosa.effects.trim(audio, top_db=10)
                    
                    #audio = trim_silence(audio[:, 0], self.silence_threshold)
                    #audio = audio.reshape(-1, 1)
                    if audio.size == 0:
                        print("Warning: {} was ignored as it contains only "
                              "silence. Consider decreasing trim_silence "
                              "threshold, or adjust volume of the audio."
                              .format(filename))
                #start
                #melM=[]
        
                melM=librosa.feature.mfcc(audio,sr=self.sample_rate,n_mfcc=40)
                

                #files_len=len(files)
                #files_len=len(files)
                #print melM
                #print len(melM[0])
                #print len(melM[0][0])

                #file_len = np.array([len(melM[i][0]) for i in range(files_len)], dtype=np.int32)
                #print np.shape(melM)
                file_len = [[len(melM[0])]]#1,20,740
                #audio_label=[[audio_label]]
                #print audio_label,type(audio_label)
                #file_len=np.resize(file_len,(-1,1))
                #np.save('file_len_test.npy',file_len)

                #max_file_len = np.amax(file_len)
                #print max_file_len
                #melM = [[list(np.pad(np.array(melM[i][j], dtype=np.float32),
                #                     (0, max_file_len - file_len[i]), 
                #                     'constant', constant_values=(0, 0)))
                #          for j in range(20)] for i in range(files_len)]

                #         for j in range(20)] for i in range(files_len)]

                melM=np.array(melM).transpose(1,0)
                
                
                #audio = np.pad(audio, [[self.receptive_field, 0], [0, 0]],
                #              'constant')
                # receptive field length is 5117
                # audios with length smaller than 5117 will be padded to 5117
                if self.sample_size:
                    # Cut samples into pieces of size receptive_field +
                    # sample_size with receptive_field overlap
                    while len(audio) > self.receptive_field:
                        # the maximum length limitation
                        piece = audio[:(self.receptive_field +
                                        self.sample_size), :]
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece})
                        sess.run(self.wl_enqueue,
                             feed_dict={self.wl_placeholder:file_len})
                        sess.run(self.label_enqueue,
                                 feed_dict={self.label_placeholder: audio_label})
                        audio = audio[self.sample_size:, :]
                        if self.gc_enabled:
                            sess.run(self.gc_enqueue, feed_dict={
                                self.id_placeholder: category_id})
                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: melM})
                    sess.run(self.wl_enqueue,
                             feed_dict={self.wl_placeholder:file_len})
                    sess.run(self.label_enqueue,
                                 feed_dict={self.label_placeholder: audio_label})
                    #sess.run(self.label_enqueue,
                    #         feed_dict={self.label_placeholder:audio_label})
                    if self.gc_enabled:
                        sess.run(self.gc_enqueue,
                                 feed_dict={self.id_placeholder: category_id})

    def start_threads(self, sess, n_threads=2):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
    
    def start_threads2(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main2, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
    
    