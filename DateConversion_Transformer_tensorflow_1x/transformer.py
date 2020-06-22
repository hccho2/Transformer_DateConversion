# coding: utf-8

import tensorflow as tf

from datafeeder import load_vocab
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme
from utils import convert_idx_to_token_tensor
from tqdm import tqdm
import logging

#logging.basicConfig(level=logging.INFO)

class Transformer:
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''
    def __init__(self, hp,train_mode=False):
        self.hp = hp
        self.train_mode=train_mode
        
        self.token2idx, self.idx2token = load_vocab(hp.vocab_filename)
        self.embeddings = get_token_embeddings(self.hp.VOCAB_SIZE, self.hp.d_model, zero_pad=True)

    def build_model(self, inputs,labels=None):
        batch_size = tf.shape(inputs)[0]
        # forward
        memory = self.encode(inputs,self.train_mode)
        
        
        if self.train_mode or labels is not None:
            decoder_inputs = labels[:,:-1]
            decoder_targets = labels[:,1:]
            logits, self.preds = self.decode(decoder_inputs, memory,self.train_mode)  # self.preds <--- argmax 취한 값
            
            
            # train scheme
            if self.hp.label_smoothing:
                y_ = label_smoothing(tf.one_hot(decoder_targets, depth=self.hp.VOCAB_SIZE))
                ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)  # lables(one-hot)
                nonpadding = tf.to_float(tf.not_equal(decoder_targets, self.token2idx["<pad>"]))  # 0: <pad>
                self.loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)
            else:
                weights = tf.ones(shape=[batch_size,self.hp.OUTPUT_LENGTH])
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=decoder_targets, weights=weights)   # targets(not one-hot)

        else:
    
            init_decoder_inputs = tf.ones((batch_size, 1), tf.int32) * self.token2idx["<sos>"]
            decoder_inputs = init_decoder_inputs
    
            for _ in range(self.hp.OUTPUT_LENGTH):
                logits, y_hat, = self.decode(decoder_inputs, memory, training=False)    
                decoder_inputs = tf.concat([init_decoder_inputs,y_hat],axis=-1)  # 생성된 것의 마지막만 살리는냐? 전체를 살리느냐?
            self.preds = y_hat

    def add_optimizer(self,global_step):
        # optimizer에 global_step을 넘겨줘야, global_step이 자동으로 증가된다.
        
        lr = noam_scheme(self.hp.learning_rate, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = optimizer.minimize(self.loss, global_step=global_step)
        
        
        return self.train_op


    def encode(self, x, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):

            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            enc *= self.hp.d_model**0.5 # scale

            enc += positional_encoding(enc, self.hp.INPUT_LENGTH)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        memory = enc
        return memory

    def decode(self, decoder_inputs, memory, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)

        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            # embedding
            dec = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)  # (N, T2, d_model)
            dec *= self.hp.d_model ** 0.5  # scale

            dec += positional_encoding(dec, self.hp.OUTPUT_LENGTH)
            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=True,
                                              scope="self_attention")

                    # Vanilla attention
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    ### Feed Forward
                    dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self.embeddings) # (d_model, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', dec, weights) # (N, T2, vocab_size)
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits, y_hat



