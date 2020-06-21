# coding: utf-8

'''
https://www.tensorflow.org/tutorials/text/transformer
https://www.tensorflow.org/guide/function?hl=ko  ===> train 속도 차이가 엄청나다.

'''

import tensorflow as tf
import numpy as np
from hparams import HParams   # pip install hparams

import time
from sklearn.model_selection import train_test_split   # pip install -U scikit-learn          python -m pip install scipy --upgrade --force

from tensorflow.python.ops.gen_batch_ops import batch

from transformer_tf import loss_function, Transformer,create_masks
from utils_tf import load_data,plot_attention_weights,CustomSchedule

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
    train_dataset = train_dataset.batch(batch_size,drop_remainder=True)
    
    valid_dataset = tf.data.Dataset.from_tensor_slices((test_input, test_target))
    valid_dataset = valid_dataset.batch(len(test_input),drop_remainder=True)


    ############## MODEL
    transformer = Transformer(num_layers=num_layers, d_model=hidden_dim, num_heads=num_heads, dff=dff,
                                     input_vocab_size=VOCAB_SIZE, target_vocab_size=VOCAB_SIZE,pe_input=encoder_length, pe_target=decoder_length,rate=drop_rate)   
    
    learning_rate = CustomSchedule(hidden_dim)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')   

    
    ckpt = tf.train.Checkpoint(transformer=transformer,optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, model_save_dir,checkpoint_name=checkpoint_name ,max_to_keep=5)
    
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)#.expect_partial()
        step_count = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        start_epoch = step_count // (len(train_input)//batch_size) + 1
        print ('Latest checkpoint restored!! {},  epochs = {}, step_count = {}'.format(ckpt_manager.latest_checkpoint, start_epoch-1, step_count))
        
    else:
        step_count = 0
        start_epoch = 1
    
    s_time = time.time()
    

    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    #train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int32), tf.TensorSpec(shape=(None, None), dtype=tf.int32),]
    
    # 구체적으로 명시하면 더 빨라진다. ----> 거의 2배
    train_step_signature = [tf.TensorSpec(shape=(batch_size, INPUT_LENGTH), dtype=tf.int32), tf.TensorSpec(shape=(batch_size, OUTPUT_LENGTH+1), dtype=tf.int32),]
    
    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
      
        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp, 
                                         True, 
                                         enc_padding_mask, 
                                         combined_mask, 
                                         dec_padding_mask)
            loss = loss_function(tar_real, predictions)
    
        gradients = tape.gradient(loss, transformer.trainable_variables)    
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        
        train_loss(loss)
        train_accuracy(tar_real, predictions)

    validation_step_signature = [tf.TensorSpec(shape=(None, INPUT_LENGTH), dtype=tf.int32), tf.TensorSpec(shape=(None, OUTPUT_LENGTH+1), dtype=tf.int32),]
    @tf.function(input_signature=validation_step_signature)
    def validation_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
      

        predictions, _ = transformer(inp, tar_inp, 
                                     False, 
                                     enc_padding_mask, 
                                     combined_mask, 
                                     dec_padding_mask)
        predictions = tf.argmax(predictions,axis=-1)
        compared = tf.equal(tf.cast(tar_real,tf.int32),tf.cast(predictions,tf.int32))   #비교하려면, tf.int32로 변환되어야 한다. tf.int64는 안된다.
        char_accuracy = tf.reduce_mean(tf.cast(compared,tf.float32))
        
        seq_accuracy =  tf.reduce_mean(tf.cast(tf.reduce_all(compared,axis=-1),tf.float32))
        
        return char_accuracy,seq_accuracy

    for epoch in range(start_epoch, start_epoch + num_epoch):
        for i, (encoder_inputs,targets) in enumerate(train_dataset):
            step_count += 1

            train_step(encoder_inputs, targets)

            if step_count % 100==0:
                print ('Epoch {:>3} step {:>5} Loss {:10.4f} Accuracy {:10.4f} Elapsed: {:>10}'.format(epoch, step_count, train_loss.result(), train_accuracy.result(), int(time.time()-s_time) ))
        
        # epoch마다 loss, accuracy reset.
        train_loss.reset_states()
        train_accuracy.reset_states()
        
        if epoch % 1== 0:
            for _, (encoder_inputs,targets) in enumerate(valid_dataset):
                char_accuracy,seq_accuracy = validation_step(encoder_inputs, targets)
                print('validation - char_accuracy: {:10.4f}, seq_accuracy: {:10.4f}'.format(char_accuracy,seq_accuracy))
        
        
        if (epoch)%5==0:
            ckpt_save_path = ckpt_manager.save(checkpoint_number = step_count)
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,ckpt_save_path))



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
    encoder_length= INPUT_LENGTH  # = hp['MAX_LEN']
    decoder_length = 11
    num_layers = hp['num_layers']
    hidden_dim = hp['hidden_dim']
    dff = hp['dff']
    num_heads = hp['num_heads']
    num_epoch = hp['num_epoch']
    drop_rate = hp['drop_rate']

    model_save_dir = hp['model_save_dir']
    checkpoint_name = hp['checkpoint_name']



    transformer = Transformer(num_layers=num_layers, d_model=hidden_dim, num_heads=num_heads, dff=dff,
                                     input_vocab_size=VOCAB_SIZE, target_vocab_size=VOCAB_SIZE,pe_input=encoder_length, pe_target=decoder_length,rate=drop_rate)



    ckpt = tf.train.Checkpoint(transformer=transformer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, model_save_dir,checkpoint_name=checkpoint_name ,max_to_keep=1)
    
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()   # expect_partial()이 없으면, warning이 많이 뜨다... optimizer관령 variables가 없다고...
        print ('Latest checkpoint restored!!', ckpt_manager.latest_checkpoint)

    else:
        print('No Model Found!!!')
        exit()


    def evaluate(encoder_inputs):
        # inp_sentences: batch_data.
        
        batch_size = len(encoder_inputs)
        decoder_inputs_init = np.array([word_to_index[GO_TOKEN]]*batch_size).reshape(-1,1)
        decoder_inputs = decoder_inputs_init
        for i in range(decoder_length):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_inputs, decoder_inputs)
      
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = transformer(encoder_inputs, 
                                                         decoder_inputs,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)
        
            
            predictions = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            
            decoder_inputs = tf.concat([decoder_inputs_init, predictions], axis=-1)
    
        return predictions, attention_weights

    result_tokens = []
    result_all = []
    end_token = word_to_index[END_TOKEN]
    for _, data in enumerate(test_dataset):
        predictions, attention_weights = evaluate(data)
        for p in predictions:
            result_tokens.append(p.numpy())
            output = ''.join([index_to_word[x] for x in p.numpy() if x != end_token])
            result_all.append(output)

        
        # id: batch data중에서 몇번째를 그릴 것인가?
        plot_attention_weights(index_to_word,attention_weights, data.numpy(), result_tokens, 'decoder_layer6_block2',id=2) # 1,2,..., num_layers

    for x,y in zip(raw_data,result_all):
        print('{:<30} ==> {}'.format(x,y))


def load_data_test():
    train_flag = False
    
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
    #test()




