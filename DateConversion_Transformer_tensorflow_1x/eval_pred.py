# coding: utf-8

import tensorflow as tf
import numpy as np
from datafeeder import load_data,load_vocab
from datafeeder import DataFeeder
from hparams import hparams as hp
from sklearn.model_selection import train_test_split
from transformer import Transformer
import time
import pandas as pd
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from utils import seq_to_index,index_to_seq,prepare_dirs
import logging,os
import infolog
log = infolog.log


def model_fn(features, labels, mode):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT
    
    
    model = Transformer(hp,train_mode=TRAIN)                 
    model.build_model(features['x'], labels)       
    
    
    predictions = {'predition': model.preds}
    if PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
    
    
    loss = model.loss
    accuracy = tf.metrics.accuracy(labels[:,1:], model.preds)  # accuracy[0]는 매 mini batch마다 계산되지 않는다.
    
    # 마지막에 취하는 mean은 의미가 없다. 숫자 1개 이므로... tf.metrics op를 만들기 위해 형식상.
    seq_accuracy = tf.metrics.mean(tf.reduce_prod( tf.cast(tf.equal(predictions['predition'],labels[:,1:]),tf.float16) ,axis=-1))



    if EVAL:
        # eval_metric_ops는 마지막 한번만 출력
        eval_metric_ops = {'acc': accuracy, 'seq_accuracy': seq_accuracy}  # 여기는 tf.metrics로 만들어진 것만...  
        
        # evaluation_hooks는 지정한 주기에 따라 출력
        evaluation_hooks = tf.train.LoggingTensorHook({"acc" : accuracy[1], "seq_accuacy": seq_accuracy[1]}, every_n_iter=1)   # 각 iteration의 loss값
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops,predictions=predictions,evaluation_hooks=[evaluation_hooks])  # loss(iteration 평균), eval_metric_ops른 넣었기 때문에 2개가 return
    
    
    if TRAIN:
        start = time.time()  # 누적 경과 시간 출력
        global_step = tf.train.get_global_step()
        train_op = model.add_optimizer(global_step)
        logging_hook = tf.train.LoggingTensorHook({"loss----" : loss, "char accuracy" : accuracy[1], "seq_accuracy": seq_accuracy[1], "elapsed": tf.timestamp()-start }, every_n_iter=1000)
        return tf.estimator.EstimatorSpec(mode=mode,train_op=train_op,loss=loss,training_hooks = [logging_hook])


#load_path = None  # 새로운 training
load_path = 'hccho-ckpt\\DCT-2019-08-03_08-59-35'
#####


load_path,restore_path,checkpoint_path = prepare_dirs(hp,load_path)


#### log 파일 작성
log_path = os.path.join(load_path, 'train.log')
infolog.init(log_path, hp.model_name)

infolog.set_tf_log(load_path)  # Estimator --> log 저장
tf.logging.set_verbosity(tf.logging.INFO)   # 이게 있어야 train log가 출력된다.

# load data
inputs, targets, word_to_index, index_to_word, VOCAB_SIZE, INPUT_LENGTH, OUTPUT_LENGTH = load_data(hp)  # (50000, 29), (50000, 12)
hp.add_hparam('VOCAB_SIZE',VOCAB_SIZE)
hp.add_hparam('INPUT_LENGTH',INPUT_LENGTH)  # 29
hp.add_hparam('OUTPUT_LENGTH',OUTPUT_LENGTH) # 11


train_input, test_input, train_target, test_target = train_test_split(inputs, targets, test_size=0.1, random_state=13371447)



datfeeder = DataFeeder(train_input,train_target,test_input,test_target,batch_size=hp.BATCH_SIZE,num_epoch=hp.NUM_EPOCHS)
def seq_accuracy(labels,predictions):  #tf.estimator.EstimatorSpec에서 넘겨 받는 값 + features + labels  ---> 이런 값들을 argument로 받을 수 있다.
    return {'seq_accuracy': tf.metrics.mean(tf.reduce_prod( tf.cast(tf.equal(predictions['predition'],labels[:,1:]),tf.float16) ,axis=-1))}


my_config =tf.estimator.RunConfig(log_step_count_steps=1000,save_summary_steps=10000,save_checkpoints_steps=3000)   # INFO:tensorflow:global_step/sec: 317.864  <--- 출력회수 제어
est = tf.estimator.Estimator(model_fn=model_fn,model_dir=load_path,config = my_config) 
est = tf.contrib.estimator.add_metrics(est, seq_accuracy)
          
print("="*10, "Evaluation")
eval_result = est.evaluate(input_fn = datfeeder.eval_input_fn,steps=5)  # steps는 최대 실행 횟수이다. data가 다 소진되면 steps를 다 채우지 못할 수도 있다.
print('\nTest set loss: {loss:0.3f}, acc: {acc:0.3f}\n'.format(**eval_result))


# Test에서는 정답이 없기 때문에, 모델 내부에서는 accuracy 계산이 불가하다. 여기서는 label과 비교하여 seq_accuracy를 계산한다.
print("="*10, "Test")
predictions = est.predict(input_fn=datfeeder.eval_input_fn)   # user defined function이 아니면, lambda function으로 넘기면 안됨
       
predictions = [x['predition'] for x in predictions]

n_output= len(predictions)  # 짜투리 때문에 모든 다 변환되지 않았다.   ---> 이 문제는 해결되었음... 그래도 남겨두었다.
     
     
     
predictions = ["".join([index_to_word[x] for x in y if x != word_to_index['<eos>']]) for y in predictions]
test_input_ = ["".join([index_to_word[x] for x in y if x != word_to_index['<eos>']]) for y in test_input[:n_output]]
test_target_ = ["".join([index_to_word[x] for x in y if x not in [word_to_index['<sos>'],word_to_index['<eos>']]]) for y in test_target[:n_output]]
      
input_prediction = list(zip(test_input_,test_target_,predictions))
      
for i in range(10):
    print(input_prediction[i])
          
output = pd.DataFrame(data={'intput': test_input_,'label': test_target_,'prediction': predictions})
output.to_csv("result.csv",index=False,header=False, sep='|', quoting=3,escapechar='\\')




#############################################
print("="*20, "Test")
s = ['TUESDAY, SSEPTEBER 10, 1991 ','feb 30, 2012                 ','Monday, November 21, 1994    ','Monday, Noveber 21, 1994    ','Monday, 21 November , 1994    ']   # 길이 check 필요
test_input_data = np.array([seq_to_index(x,word_to_index) for x in s])
test_input_data = pad_sequences(test_input_data, maxlen=INPUT_LENGTH, padding='post')
test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":test_input_data},shuffle=False)

predictions = est.predict(input_fn=test_input_fn)
predictions = [x['predition'] for x in predictions]
outputs = [index_to_seq(x,index_to_word,word_to_index) for x in predictions]
for x in zip(s,outputs):
    print(x)














print('Done')