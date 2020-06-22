# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

hparams = tf.contrib.training.HParams(

    log_dir = "hccho-ckpt" ,
    model_name = "DCT",   #DateConversion_Transformer
    ckpt_file_name_preface = 'model.ckpt',  # 이 이름을 바꾸면, get_most_recent_checkpoint도 바꿔야 한다.
    PARAMS_NAME = "params.json",
    
    data_filename = './data/date.txt',
    vocab_filename = './data/vocab.pickle',


    d_model = 64,   # hidden dimension of encoder/decoder
    d_ff = 128,      # hidden dimension of feedforward layer
    dropout_rate= 0.3,
    num_blocks = 6,  # number of encoder/decoder blocks
    num_heads = 8, # number of attention heads   d_model % num_heads = 0
    warmup_steps = 8000,

    BATCH_SIZE = 32,
    NUM_EPOCHS = 100,
    

    learning_rate = 0.001,
    label_smoothing = False,
 
)
