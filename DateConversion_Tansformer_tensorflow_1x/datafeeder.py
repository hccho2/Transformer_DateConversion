# coding: utf-8

import tensorflow as tf
import numpy as np
import os
import pandas as pd
import pickle
import errno
from tensorflow.python.keras.preprocessing.text import Tokenizer
GO_TOKEN = '<sos>'
END_TOKEN = '<eos>'
PAD = '<pad>'



def load_data(hp):
    
    train_data = pd.read_csv(hp.data_filename, header=None, delimiter='_' )
    inputs, targets = train_data[0].tolist(), train_data[1].tolist()

    # TF 1.10에서는 소문자 Tokenizer의 소문자 변경에 문제가 있음. 그 이후 버전에서는 해결
    inputs = [[c for c in x.lower()] for x in inputs]
    targets = [[GO_TOKEN] + [c for c in x.lower()] + [END_TOKEN] for x in targets]
    
    tokenizer = Tokenizer(lower=True)  # 소문자로 안 바뀜(tensorflow 1.10에서만) 다른 버전에서는 잘됨.   Filter 기능이 포함되어 있다.
    tokenizer.fit_on_texts(inputs+targets) # 전체 character 분석   
    
    inputs = tokenizer.texts_to_sequences(inputs)  # character를 숫자로 변환
    targets = tokenizer.texts_to_sequences(targets)
    
    
    word_to_index = tokenizer.word_index  # 0에는 아무것도 할당하지 않는다.
    word_to_index[PAD] = 0
    index_to_word = dict(map(reversed, word_to_index.items()))
    
    
    
    VOCAB_SIZE = len(word_to_index)
    INPUT_LENGTH = len(inputs[0])
    OUTPUT_LENGTH = len(targets[0])-1  # END_TOKEN 제외
    
    
    if (not (os.path.exists(hp.vocab_filename))):
        with open(hp.vocab_filename, 'wb') as f:
            pickle.dump({'word_to_index': word_to_index, 'index_to_word': index_to_word}, f)
    
    
    return np.array(inputs), np.array(targets), word_to_index, index_to_word,VOCAB_SIZE,INPUT_LENGTH,OUTPUT_LENGTH

def load_vocab(vocab_filename):
    if not os.path.isfile(vocab_filename):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), vocab_filename)
    with open(vocab_filename,'rb') as f:
        data=pickle.load(f,encoding='utf-8')
    
    return data['word_to_index'], data['index_to_word']
class DataFeeder():
    # DataFeeder using tf.data.Dataset
    def __init__(self,train_input,train_target, eval_input=None,eval_target=None,batch_size=128,buffer_size=50000,drop_remainder=False,num_epoch=3):
        self.train_input = train_input
        self.train_target = train_target
        self.eval_input = eval_input
        self.eval_target = eval_target
        
        self.buffer_size = buffer_size
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
    
    def mapping_fn(self,X, Y):
        inputs, labels = {'x': X}, Y
        return inputs, labels
    
    def train_input_fn(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.train_input, self.train_target))  # 여기의 argument가 mapping_fn의 argument가 된다.
        dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.batch(self.batch_size,drop_remainder=self.drop_remainder)
        dataset = dataset.repeat(count=self.num_epoch)
        dataset = dataset.map(self.mapping_fn)
        iterator = dataset.make_one_shot_iterator()
    
        return iterator.get_next()


    def eval_input_fn(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.eval_input, self.eval_target))
        dataset = dataset.map(self.mapping_fn)
        dataset = dataset.batch(self.batch_size,drop_remainder=self.drop_remainder)
        iterator = dataset.make_one_shot_iterator()
        
        return iterator.get_next()

