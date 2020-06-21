# coding: utf-8

'''
https://www.tensorflow.org/tutorials/text/transformer


'''

import tensorflow as tf
import numpy as np
import pandas as pd
from hparams import HParams   # pip install hparams
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.keras import preprocessing
import pickle,os, random,time,math
from sklearn.model_selection import train_test_split   # pip install -U scikit-learn          python -m pip install scipy --upgrade --force


GO_TOKEN = '<sos>'
END_TOKEN = '<eos>'
PAD = '<pad>'
UNK = '<unk>'

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

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


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


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
      
    pos_encoding = angle_rads[np.newaxis, ...]
      
    return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def create_padding_mask(seq):
    # input값인 seq에서 0인곳에는 1, 나머지는 0  ==> 나중에 여기에 -inf를 곱한다.
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len) [[0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],[0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], ... [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)     # =====> encoder_input에 대한 self attentin mask ---> (inp값이 0인곳은 1, 나머지는 0) -- 여기에 -inf를 나중에 곱하는 방식.
    
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)     # =====> decoder_input에 대한
    
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])  # tf.shape(tar)[1] = decoder_length
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
    return enc_padding_mask, combined_mask, dec_padding_mask



class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # x: (batch_size, input_seq_len, d_model), mask: (N,1,1,input_seq_len)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2



class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)


    def call(self, x, enc_output, training,look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        # look_ahead_mask: causal mask
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (v, k, q) ---> (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,self.d_model)  #(1,T,D)---> 실제 더할 때는 batch에 대한 broadcasting이 이루어진다.


        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len) --> (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
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

def train():

    ############ DATA
    inputs, targets, word_to_index, index_to_word, VOCAB_SIZE, INPUT_LENGTH, OUTPUT_LENGTH = load_data(hp)  # (50000, 29), (50000, 12)
    print(word_to_index)
    
    
    
    ########## Hyper parameters
    hp['vocab_size'] = VOCAB_SIZE   
    batch_size = hp['batch_size']
    encoder_length= INPUT_LENGTH  # = hp['MAX_LEN']
    decoder_length = OUTPUT_LENGTH
    num_layers = hp['num_layers']
    hidden_dim = hp['hidden_dim']
    dff = hp['dff']
    num_heads = hp['num_heads']
    num_epoch = hp['num_epoch']
    drop_rate = hp['drop_rate']

    model_save_dir = hp['model_save_dir']
    checkpoint_name = hp['checkpoint_name']



    
    train_input, test_input, train_target, test_target = train_test_split(inputs, targets, test_size=0.1, random_state=13371447)
    print('train_input: {},test_input: {},train_target: {},test_target: {}'.format(train_input.shape,test_input.shape,train_target.shape,test_target.shape))
    print('INPUT_LENGTH: {}, OUTPUT_LENGTH: {}'.format(INPUT_LENGTH,OUTPUT_LENGTH)) 
     
    train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_target))  # 여기의 argument가 mapping_fn의 argument가 된다.
    train_dataset = train_dataset.shuffle(buffer_size=20000)
    train_dataset = train_dataset.batch(batch_size,drop_remainder=False)



    ############## MODEL
    transformer = Transformer(num_layers=num_layers, d_model=hidden_dim, num_heads=num_heads, dff=dff,
                                     input_vocab_size=VOCAB_SIZE, target_vocab_size=VOCAB_SIZE,pe_input=encoder_length, pe_target=decoder_length,rate=drop_rate)

   
    
    learning_rate = CustomSchedule(hidden_dim)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')   

    
    ckpt = tf.train.Checkpoint(transformer=transformer,optimizer=optimizer)
    
    ckpt_manager = tf.train.CheckpointManager(ckpt, model_save_dir, checkpoint_name=checkpoint_name, max_to_keep=5)
    
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')

    start_epoch = 1
    s_time = time.time()
    step_count = 0    
    for epoch in range(start_epoch, num_epoch+1):
        for i, (encoder_inputs,targets) in enumerate(train_dataset):
            step_count += 1
            decoder_inputs = targets[:, :-1]
            decoder_outputs = targets[:, 1:]
    
    
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_inputs, decoder_inputs)
    
            with tf.GradientTape() as tape:
                predictions, _ = transformer(encoder_inputs, decoder_inputs, training=True, 
                                             enc_padding_mask=enc_padding_mask, look_ahead_mask = combined_mask, dec_padding_mask =dec_padding_mask)
                loss = loss_function(decoder_outputs, predictions)
    
            gradients = tape.gradient(loss, transformer.trainable_variables)    
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
            
            train_loss(loss)
            train_accuracy(decoder_outputs, predictions)

            if step_count % 100==0:
                print ('Epoch {:>3} step {:>5} Loss {:10.4f} Accuracy {:10.4f} Elapsed: {:>10}'.format(epoch, step_count, train_loss.result(), train_accuracy.result(), int(time.time()-s_time) ))
                
                

        if (epoch)%1==0:
            print('epoch: {:>5}'.format(epoch))
            # valid loss check
        
        
        if (epoch)%5==0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,ckpt_save_path))



def load_data_test():
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
    #load_data_test()
    train()




