import numpy as np
import sys
import os
import numpy as np
import pickle
from tqdm import *
import json
import re
import fnmatch
import csv
import librosa
from python_speech_features import mfcc
import scipy.io.wavfile as wav


def find_files(directory, pattern='*.trn'):
    '''Recursively finds all files matching the pattern.'''
    filenames = []
    for _,dirnames, files in os.walk(directory):
        for file in fnmatch.filter(files, pattern):
            filenames.append(file)
    return filenames

def load_text(file_path, filenames,output_path, create_dict=False):
    pinyin = []
    phoneme = []
    melM = []
    for i,filename in tqdm(enumerate(filenames)):
        
        f_path = file_path + filename
        curr_path = f_path
        if create_dict==False:
            with open(f_path, "r") as f:
                curr_path = f.readline()[1:-1]
                
                print 'curr_path',i,curr_path
        with open(curr_path, "r") as f:
            reader = csv.reader(f, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            
            for i,row in enumerate(reader):
                if i==1:
                    pinyin.append(row)
                if i==2:
                    phoneme.append(row)
      
    '''with open(output_path, "w+b") as f:
        writer = csv.writer(f, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in phoneme:
            writer.writerow(row)'''

def load_audio(file_path, filenames,output_path):
    melM = []
    for i,filename in tqdm(enumerate(filenames)):        
        f_path = file_path + filename[:-4]
        fs, audio = wav.read(f_path)
        mel = mfcc(audio, samplerate=fs, numcep=40)#20
        mel = np.asarray(mel[np.newaxis, :])
        mel = (mel - np.mean(mel))/np.std(mel)
        melM.extend(mel)
    np.save(output_path, melM)
    return melM        

def create_dict(txt_path):
    word_dict = {}
    with open(txt_path, "r") as f:
        reader = csv.reader(f, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            for item in row:
                if item not in word_dict:
                    word_dict[item] = 1
                else:
                    word_dict[item] += 1

    return word_dict

def shengdiao_test(shengdiao):
    try:
        s = int(shengdiao)
        return True
    except:
        return False

def shengyun_dict(txt_path):
    shengmu = dict() 
    yunmu = dict() 
    shengmu_count = 0
    yunmu_count = 0
    with open(txt_path, "r") as f:
        reader = csv.reader(f, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            for i, item in enumerate(row):
                has_shengdiao = shengdiao_test(item[-1])
                if i%2==0:
                    if not has_shengdiao:
                        if not item in shengmu.keys():
                            shengmu[item] = shengmu_count
                            shengmu_count += 1
                    else:
                        print "error: shengmu should not include", item                    
                else:
                    if has_shengdiao:
                        if not item in yunmu.keys():
                            yunmu[item] = yunmu_count
                            yunmu_count += 1
                    else:
                        print "error: yunmu should not include", item  
        
    return shengmu, yunmu 


def get_target_indices(phoneme_path):
    target_indices = []
    max_phoneme_num = 0
    file_num = 0
    with open(phoneme_path, "r") as f:
        reader = csv.reader(f, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i, row in tqdm(enumerate(reader)):
            phoneme_num = len(row)
            if phoneme_num > max_phoneme_num:
                max_phoneme_num = phoneme_num
            col1 = [i] * phoneme_num
            col2 = range(phoneme_num)
            indices = zip(col1,col2)
            target_indices.extend(indices)
            file_num = i
        dense_shape = [file_num + 1, max_phoneme_num + 1]
    return target_indices, dense_shape

def create_pair_index(phoneme_path):
    pair_index = {}
    index_count = 0
    with open(phoneme_path, "r") as f:
        reader = csv.reader(f, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for line in reader:
            prev = line
            latter = line[1:]
            pairs = zip(prev, latter)
            for pair in pairs:
                if not pair in pair_index:
                    pair_index[pair] = index_count
                    index_count += 1
    """
    for shengmu in shengmu_dict.keys():
        for yunmu in yunmu_dict.keys():
            pair_index[(shengmu, yunmu)] = index_count
            index_count += 1
    """
    return pair_index

def get_pair_index(phoneme_path, index_dict):
    pair_indexes = []
    with open(phoneme_path, "r") as f:
        reader = csv.reader(f, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for line in reader:
            prev = line
            latter = line[1:]
            pairs = zip(line, latter)
            for pair in pairs:
                if not pair in index_dict:
                    print "error: This pair doesn't recorded in the phoneme index dict. "
                else:
                    pair_indexes.append(index_dict[pair])
    return pair_indexes


def get_pair_index_with_id(phoneme_path, index_dict):
    pair_indexes_dict = {}
    with open(phoneme_path, "r") as f:
        reader = csv.reader(f, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i,line in enumerate(reader):
            pair_indexes = []
            prev = line
            latter = line[1:]
            pairs = zip(prev, latter)
            for pair in pairs:
                if not pair in index_dict:
                    print "error: This pair doesn't recorded in the phoneme index dict. "
                else:
                    pair_indexes.append(index_dict[pair])
            pair_indexes_dict[i] = pair_indexes
    return pair_indexes_dict


def create_file_dicts(filenames, pair_indexes_dict, melM, padded_melM):
    file_list = list()
    '''return the min id and the max id of all the speakers'''
    file_pattern = r'[A-Z]([0-9]+)_([0-9]+)\.wav'
    id_reg_expression = re.compile(file_pattern)
    
    for i, filename in enumerate(filenames):
        curr = dict()
        matches = id_reg_expression.findall(filename)[0]
        speaker_id, recording_id = [int(id_) for id_ in matches]
        #print 'speaker_id/n',speaker_id
        #print 'recording_id',recording_id
        curr["id"] = i
        curr["speaker_id"] = speaker_id
        curr["recording_id"] = recording_id
        
        target_value = pair_indexes_dict[i]
        curr["target_value"] = target_value
        curr["phoneme_num"] = len(target_value) + 1
        
        curr["melM"] = padded_melM[i]   
        curr["freq_length"] = len(melM[i])
        file_list.append(curr)
        
    return file_list


def create_input_dict(file_list):
    # classify dicts according to recording_id
    input_dict = {}
    for f in file_list:
        recording_id = f["recording_id"]
        if not recording_id in input_dict:
            input_dict[recording_id] = []
            input_dict[recording_id].append(f)
        else:
            input_dict[recording_id].append(f)
        
    input_dict_list = []   
    for recording in input_dict:
        curr_input = dict()
        file_list = input_dict[recording]
        speaker_ids = []
        mels = []
        mel_lens = []
        target_values = []
        target_indices = []
        f_count = 0
        max_phoneme_len = 0
        for i, f in enumerate(file_list):
            speaker_ids.append(f["speaker_id"])
            mels.append(f["melM"])
            mel_lens.append(f["freq_length"])
            target_values.append(f["target_value"])#target_values.extend(f["target_value"])
            phoneme_num = f["phoneme_num"]
            indices = zip([i]*(phoneme_num - 1), range(phoneme_num - 1))
            target_indices.append(indices)#target_indices.extend(indices) #file_list zhong wen jian ge shu 
            
            #print 'target_indices',target_indices
            if phoneme_num > max_phoneme_len:
                max_phoneme_len = phoneme_num
            f_count += 1
        dense_shape = [f_count, max_phoneme_len-1]
        
        curr_input["frequencies"] = mels
        curr_input["frequencies_seq_len"] = mel_lens
        curr_input["speaker_ids"] = speaker_ids
        curr_input["target_indices"] = target_indices
        curr_input["target_values"] = target_values
        curr_input["dense_shape"] = dense_shape
        input_dict_list.append(curr_input)
        
    return input_dict_list


def get_inputs():
    filenames = find_files("./data/")
    #load_text("./data/", filenames, "./phoneme_pairs/all_phoneme.csv", create_dict=True) 
    #shengmu, yunmu = shengyun_dict("./phoneme_pairs/all_phoneme.csv")
    index_dict = create_pair_index("./phoneme_pairs/blanked_all_phoneme.csv" )
    print len(index_dict), "detected."
    
    print "Preprocessing training data..."
    train_filenames = find_files("./train/")
    #load_text("./train/", train_filenames, "./phoneme_pairs/train_phoneme.csv")     
    train_melM = np.load("./phoneme_pairs/train_melM.npy")
    padded_train_melM = np.load("./phoneme_pairs/padded_train_melM.npy")
    train_pair_indexes_dict = get_pair_index_with_id("./phoneme_pairs/blanked_train_phoneme.csv", index_dict)
    train_file_list = create_file_dicts(train_filenames, train_pair_indexes_dict, train_melM, padded_train_melM)
    train_inputs = create_input_dict(train_file_list)
    
    print "Preprocessing validation data..."
    dev_filenames = find_files("./dev/")
    #load_text("./dev/", dev_filenames, "./phoneme_pairs/dev_phoneme.csv")     
    dev_melM = np.load("./phoneme_pairs/dev_melM.npy")
    padded_dev_melM = np.load("./phoneme_pairs/padded_dev_melM.npy")
    dev_pair_indexes_dict = get_pair_index_with_id("./phoneme_pairs/blanked_dev_phoneme.csv", index_dict)
    dev_file_list = create_file_dicts(dev_filenames, dev_pair_indexes_dict, dev_melM, padded_dev_melM)
    dev_inputs = create_input_dict(dev_file_list)
    
    print "Finished preprocessing." 
    return train_inputs, dev_inputs, index_dict

if __name__=="__main__":
    filenames=find_files("./train/")
    load_audio("./train/", filenames,"./phoneme_pairs/train_melM.npy")
    filenames=find_files("./dev/")
    load_audio("./dev/", filenames,"./phoneme_pairs/padded_dev_melM.npy")
    