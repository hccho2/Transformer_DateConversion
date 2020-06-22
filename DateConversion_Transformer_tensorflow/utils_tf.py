
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.keras import preprocessing
import pickle
import matplotlib.pyplot as plt
from hparams import HParams
from sklearn.model_selection import train_test_split


GO_TOKEN = '<sos>'
END_TOKEN = '<eos>'
PAD = '<pad>'
UNK = '<unk>'

def load_data(hp,filename=None,train_flag=True):
    # 입력 파일에 data의 길이가 맞춰져 있다.
    
    if filename is None:
        filename = hp['data_filename']
    
    train_data = pd.read_csv(filename, header=None, delimiter='_' )
    if train_data.shape[1] == 1:
        inputs = train_data[0].tolist()
        inputs = [[c for c in x.lower().strip()] for x in inputs]
    else:
        inputs, targets = train_data[0].tolist(), train_data[1].tolist()
        inputs = [[c for c in x.lower().strip()] for x in inputs]
        targets = [[GO_TOKEN] + [c for c in x.lower().strip()] + [END_TOKEN] for x in targets]        

    
    if train_flag:
        
        tokenizer = Tokenizer(lower=True, oov_token=UNK)  # 소문자로 안 바뀜(tensorflow 1.10에서만) 다른 버전에서는 잘됨.   Filter 기능이 포함되어 있다.
        tokenizer.fit_on_texts(inputs+targets) # 전체 character 분석               
        
        inputs = tokenizer.texts_to_sequences(inputs)  # character를 숫자로 변환  ----> [[25, 13, 31, 26, 13, 24, 27, 13, 15, 6, 5, 10, 9, 6, 1, 4, 4, 17],[1, 2, 19, 14, 1, 19, 4, 2], ... ]
        targets = tokenizer.texts_to_sequences(targets)
        
        inputs = preprocessing.sequence.pad_sequences(inputs, maxlen=hp['MAX_LEN'], padding='post',truncating='post')
        
        word_to_index = tokenizer.word_index  # 0에는 아무것도 할당하지 않는다.
        word_to_index[PAD] = 0
        index_to_word = dict(map(reversed, word_to_index.items()))
        VOCAB_SIZE = len(word_to_index)
        INPUT_LENGTH = len(inputs[0])
        OUTPUT_LENGTH = len(targets[0])-1  # END_TOKEN 제외


        with open(hp['vocab_filename'], 'wb') as f:
            pickle.dump({'word_to_index': word_to_index, 'index_to_word': index_to_word, 'tokenizer': tokenizer}, f)

        
        return np.array(inputs,dtype=np.int32), np.array(targets,dtype=np.int32), word_to_index, index_to_word,VOCAB_SIZE,INPUT_LENGTH,OUTPUT_LENGTH

    else:
        with open(hp['vocab_filename'],'rb') as f:
            data_loaded = pickle.load(f)
            word_to_index = data_loaded['word_to_index']
            index_to_word = data_loaded['index_to_word']
            tokenizer = data_loaded['tokenizer']
            
        inputs = tokenizer.texts_to_sequences(inputs)
        inputs = preprocessing.sequence.pad_sequences(inputs, maxlen=hp['MAX_LEN'], padding='post',truncating='post')
        VOCAB_SIZE = len(word_to_index)
        INPUT_LENGTH = len(inputs[0])
        return np.array(inputs,dtype=np.int32), word_to_index, index_to_word, VOCAB_SIZE, INPUT_LENGTH


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        
        self.warmup_steps = warmup_steps
      
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class SequenceAccuracy(tf.keras.metrics.Metric):
    # sequence 전체가 일치할 때
    def __init__(self, name="sequence_accuracy", **kwargs):
        super(SequenceAccuracy, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="seq_accuracy", initializer="zeros")
        self.example_num = self.add_weight(name='example_num', initializer='zeros')

    def update_state(self, y_true, y_pred):  # tensorflow 2.0에서는 sample_weight가 있어야 한다. 2.2에서는 있으면 안된다.
        # y_pred: (N,T,D), y_true: (N,T)
        y_pred = tf.argmax(y_pred, axis=-1)
        values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
        values = tf.reduce_sum(tf.cast(tf.reduce_all(values,axis=-1),tf.float32))
        self.true_positives.assign_add(values)
        self.example_num.assign_add(tf.cast(tf.shape(y_true)[0],tf.float32))

    def result(self):
        return self.true_positives / self.example_num

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.0)
        self.example_num.assign(0.0)

def plot_attention_weights(index_to_word,attention, sentence, result, layer,id=0):
    fig = plt.figure(figsize=(16, 8))
    
    attention = attention[layer][id]  # batch중에서 id에 해당하는 것 (N,num_heads,decoder_length, encoder_length) ---> 추출
    sentence = sentence[id]
    result = result[id]


    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head+1)
        
        # plot the attention weights
        ax.matshow(attention[head], cmap='viridis')
        
        fontdict = {'fontsize': 10}
        
        ax.set_xticks(range(len(sentence)))
        ax.set_yticks(range(len(result)))
        
        ax.set_ylim(len(result)-1.5, -0.5)  # 결과의 마지막 <eos>가 보이지 않게 만든다.
            
        ax.set_xticklabels([index_to_word[i] for i in sentence], fontdict=fontdict, rotation=90)  # rotation=90 ---> 반시계방향 회전
        ax.set_yticklabels([index_to_word[i] for i in result], fontdict=fontdict)
        
        ax.grid(color='white', linestyle='-', linewidth=0.3, )
        
        ax.set_xlabel('Head {}'.format(head+1))
  
    plt.tight_layout()
    plt.show()


def load_data_test():
    
    hp = HParams(
        data_filename='date.txt',
        vocab_filename = 'vocab.pickle',
        model_save_dir = './saved_model',
        checkpoint_name = 'model_ckpt',
        MAX_LEN = 29,
        batch_size = 32,
        num_epoch = 10,
        
        
        num_layers = 6,
        hidden_dim = 32,
        dff = 128,
        num_heads = 8,
        drop_rate = 0.1,
        
        
    )
    
    
    train_flag = True
    
    if train_flag:
        inputs, targets, word_to_index, index_to_word, VOCAB_SIZE, INPUT_LENGTH, OUTPUT_LENGTH = load_data(hp,train_flag=train_flag)  # (50000, 29), (50000, 12)

        print(word_to_index)
        hp['vocab_size'] = VOCAB_SIZE   # set_hparam도 있음.  
    
        train_input, test_input, train_target, test_target = train_test_split(inputs, targets, test_size=0.1, random_state=13371447)
        print('train_input: {},test_input: {},train_target: {},test_target: {}'.format(train_input.shape,test_input.shape,train_target.shape,test_target.shape))
         
         
        dataset = tf.data.Dataset.from_tensor_slices((train_input, train_target))  # 여기의 argument가 mapping_fn의 argument가 된다.
        dataset = dataset.shuffle(buffer_size=hp['batch_size']*10)
        dataset = dataset.batch(hp['batch_size'],drop_remainder=False)
    
    
        for i,(x,y) in enumerate(dataset):
            if i% 300 == 0:
                print(x.shape,y.shape)
        print('='*10)
        for i,(x,y) in enumerate(dataset):
            if i% 1000 == 0:
                print(x.shape,y.shape)   
    
    else:
        inputs, word_to_index, index_to_word, VOCAB_SIZE, INPUT_LENGTH = load_data(hp,'test_data.txt',train_flag=train_flag)
    
        print(word_to_index)
        hp['vocab_size'] = VOCAB_SIZE   # set_hparam도 있음.  
    

if __name__ == '__main__':
    load_data_test()






