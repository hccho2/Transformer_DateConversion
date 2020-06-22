# coding: utf-8

'''
keras Model, fit으로 train하기

custom metrics 정의하기
https://www.tensorflow.org/guide/keras/train_and_evaluate#custom_metrics

'''

import tensorflow as tf
import numpy as np
from hparams import HParams   # pip install hparams
from sklearn.model_selection import train_test_split   # pip install -U scikit-learn          python -m pip install scipy --upgrade --force

from transformer_tf import loss_function, Transformer,create_masks
from utils_tf import load_data,CustomSchedule,SequenceAccuracy


GO_TOKEN = '<sos>'
END_TOKEN = '<eos>'
PAD = '<pad>'
UNK = '<unk>'

hp = HParams(
    data_filename='date.txt',
    vocab_filename = 'vocab.pickle',
    model_save_dir = './saved_model_keras',
    MAX_LEN = 29,
    batch_size = 32,
    num_epoch = 20,
    
    # model parameters
    num_layers = 6,
    hidden_dim = 32,
    dff = 128,
    num_heads = 8,
    drop_rate = 0.1,
    
    
)


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

    
    train_input, test_input, train_target, test_target = train_test_split(inputs, targets, test_size=0.1, random_state=13371447)
    print('train_input: {},test_input: {},train_target: {},test_target: {}'.format(train_input.shape,test_input.shape,train_target.shape,test_target.shape))
    print('INPUT_LENGTH: {}, OUTPUT_LENGTH: {}'.format(INPUT_LENGTH,OUTPUT_LENGTH)) 
     
    train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_target))  # 여기의 argument가 mapping_fn의 argument가 된다.
    def map_fn(inp,tar):
        # dict 또는 tuple로 묶어서...
        #inputs = {'encoder_in': inp, 'decoder_in': tar[:,:-1]}
        #return inputs, tar[:,1:]
        return (inp,tar[:,:-1]), tar[:,1:]
    
    
    train_dataset = train_dataset.batch(batch_size,drop_remainder=False)
    train_dataset = train_dataset.map(map_fn)
    train_dataset = train_dataset.shuffle(buffer_size=20000).repeat()

    valid_dataset = tf.data.Dataset.from_tensor_slices((test_input, test_target))
    valid_dataset = valid_dataset.batch(len(test_input),drop_remainder=True)
    valid_dataset = valid_dataset.map(map_fn)
     
    ############## MODEL
    transformer = Transformer(num_layers=num_layers, d_model=hidden_dim, num_heads=num_heads, dff=dff,
                                     input_vocab_size=VOCAB_SIZE, target_vocab_size=VOCAB_SIZE,pe_input=encoder_length, pe_target=decoder_length,rate=drop_rate)
    

    learning_rate = CustomSchedule(hidden_dim)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    #train_loss = tf.keras.metrics.Mean(name='train_loss')  # keras에서 loss는 기본으로 metric에 포함되어 있다.
    char_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='char_accuracy')   
    seq_accuracy = SequenceAccuracy(name='seq_accuracy')   

    

    encoder_inputs = tf.keras.Input(shape=(INPUT_LENGTH,),dtype=tf.int32)
    decoder_inputs = tf.keras.Input(shape=(OUTPUT_LENGTH,),dtype=tf.int32)
    
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_inputs, decoder_inputs)
    
    predictions, _ = transformer(encoder_inputs, decoder_inputs, training=True, 
                                enc_padding_mask=enc_padding_mask, look_ahead_mask = combined_mask, dec_padding_mask =dec_padding_mask)


    model = tf.keras.Model([encoder_inputs,decoder_inputs],predictions)   # train_dataset의 return 형태와 같아야 한다.
    model.compile(optimizer = optimizer,loss=loss_function,metrics=[char_accuracy,seq_accuracy])

    
    print(model.summary())

    initial_epoch = 0
    latest = tf.train.latest_checkpoint(model_save_dir)  # 단순히 checkpoint파일을 읽어서...
    if latest:
        model.load_weights(latest)
        initial_epoch = int(latest.split('.')[2])
        
    file_path = model_save_dir + '/model_ckpt.{epoch:02d}.ckpt'   # 확장자에 따라서, checkpoint형식 또는 h5형식으로 저장.
    
    
    # priod=5(먹히지 안흔다. save_freq='epoch'으로 동작 1epoch마다 저장),                     save_freq=data갯수단위
    #save_freq = len(train_input)//batch_size * batch_size  ----> sample 갯수로 지정하면 된다고 하는데, 안됨. 무조건 1 epoch마다 저장하고 잇음.
    # tensorflow 2.2에서는 save_freq ---> step 단위
    
    # initial_epoch, epochs(initial_epoch에 추가하는 epoch수가 아님. 누적 epoch임)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(file_path,save_weights_only=True,verbose=1,save_freq=1406*5)  
    
    # training=True로 명시된 모델이므로, validation에서도 training=True가 적용된다.  keara구조에서는 control이 안된다.
    
    steps_per_epoch = len(train_input) // batch_size
    model.fit(train_dataset,initial_epoch=initial_epoch, epochs=num_epoch,verbose=1, steps_per_epoch = steps_per_epoch,
              validation_data=valid_dataset,validation_freq=1,callbacks = [checkpoint_callback])


def test():
    inputs, word_to_index, index_to_word, VOCAB_SIZE, INPUT_LENGTH = load_data(hp,'test_data.txt',train_flag=False)
    print(word_to_index)
    print('train_input: {}'.format(inputs.shape))
    print('INPUT_LENGTH: {}'.format(INPUT_LENGTH))

    
    raw_data = []
    pad_token = word_to_index[PAD]
    for i,d in enumerate(inputs):
        raw_data.append(''.join([index_to_word[x] for x in d if x != pad_token]))

    print(raw_data)


    test_dataset = tf.data.Dataset.from_tensor_slices(inputs)  # 여기의 argument가 mapping_fn의 argument가 된다.
    test_dataset = test_dataset.batch(len(inputs),drop_remainder=False)




    ########## Hyper parameters
    hp['vocab_size'] = VOCAB_SIZE   
    batch_size = hp['batch_size']
    encoder_length= INPUT_LENGTH  # = hp['MAX_LEN']
    decoder_length = 11
    num_layers = hp['num_layers']
    hidden_dim = hp['hidden_dim']
    dff = hp['dff']
    num_heads = hp['num_heads']
    drop_rate = hp['drop_rate']

    model_save_dir = hp['model_save_dir']



    transformer = Transformer(num_layers=num_layers, d_model=hidden_dim, num_heads=num_heads, dff=dff,
                                     input_vocab_size=VOCAB_SIZE, target_vocab_size=VOCAB_SIZE,pe_input=encoder_length, pe_target=decoder_length,rate=drop_rate)

    encoder_inputs = tf.keras.Input(shape=(INPUT_LENGTH,),dtype=tf.int32)
    decoder_inputs = tf.keras.Input(shape=(None,),dtype=tf.int32)
    
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_inputs, decoder_inputs)
    
    predictions, _ = transformer(encoder_inputs, decoder_inputs, training=False, 
                                enc_padding_mask=enc_padding_mask, look_ahead_mask = combined_mask, dec_padding_mask =dec_padding_mask)


    model = tf.keras.Model([encoder_inputs,decoder_inputs],predictions)
    print(model.summary())
    
    


    latest = tf.train.latest_checkpoint(model_save_dir)  # 단순히 checkpoint파일을 읽어서...
    if latest:
        model.load_weights(latest).expect_partial()
        print ('Latest checkpoint restored!!', latest)
    else:
        print('No Model Found!!')
        exit()


    result_all = []
    end_token = word_to_index[END_TOKEN]
    
    
    @tf.function(experimental_relax_shapes=True)
    def evaluate(encoder_inputs, decoder_inputs):
        return model.predict([encoder_inputs, decoder_inputs])
    
    
    for _, encoder_inputs in enumerate(test_dataset):
        batch_size = len(encoder_inputs)
        decoder_inputs_init = np.array([word_to_index[GO_TOKEN]]*batch_size).reshape(-1,1)
        decoder_inputs = tf.cast(decoder_inputs_init,tf.int32)
        for i in tf.range(decoder_length):
      
            #predictions = model.predict([encoder_inputs, decoder_inputs])   -----> 많이 느리다.
            predictions = model([encoder_inputs, decoder_inputs])
            
            predictions = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            
            decoder_inputs = tf.concat([decoder_inputs_init, predictions], axis=-1)

        for p in predictions:
            output = ''.join([index_to_word[x] for x in p.numpy() if x != end_token])
            result_all.append(output)
        
    for x,y in zip(raw_data,result_all):
        print('{:<30} ==> {}'.format(x,y))  
        
        
if __name__ == '__main__':
    train()

    #test()
