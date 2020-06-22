# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Utility functions
'''

import tensorflow as tf
# from tensorflow.python import pywrap_tensorflow
# import numpy as np
import json
import os, re
import logging

from glob import glob
from datetime import datetime



def get_most_recent_checkpoint(checkpoint_dir):
    checkpoint_paths = [path for path in glob("{}/*.ckpt-*.data-*".format(checkpoint_dir))]
    
    if checkpoint_paths == []: 
        return ''
    
    idxes = [int(os.path.basename(path).split('-')[1].split('.')[0]) for path in checkpoint_paths]

    max_idx = max(idxes)
    lastest_checkpoint = os.path.join(checkpoint_dir, "model.ckpt-{}".format(max_idx))

    #latest_checkpoint=checkpoint_paths[0]
    print(" [*] Found lastest checkpoint: {}".format(lastest_checkpoint))
    return lastest_checkpoint

def prepare_dirs(hp, load_path=None):
    # load_path = 'hccho-ckpt\\hccho-mm-2019-07-31_13-56-59'
    # checkpoint_path = 'hccho-ckpt\\hccho-mm-2019-08-02_10-21-12\\model.ckpt'
    # restore_path = 'hccho-ckpt\\hccho-mm-2019-08-02_09-56-45\\model.ckpt-120000'
    
    from shutil import copyfile as copy_file
    
    def get_time():
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    
    if load_path is None:
        load_path = os.path.join(hp.log_dir, "{}-{}".format(hp.model_name, get_time()))
        os.makedirs(load_path)
        
        save_hparams(load_path, hp)
        copy_file("hparams.py", os.path.join(load_path, "hparams.py"))
        
    else:
        load_hparams(hp, load_path)
        
    checkpoint_path = os.path.join(load_path, hp.ckpt_file_name_preface)
    restore_path = get_most_recent_checkpoint(load_path)


    
    return load_path,restore_path,checkpoint_path


def load_hparams(hparams, load_path, skip_list=[]):
    # log dir에 있는 hypermarameter 정보를 이용해서, hparams.py의 정보를 update한다.
    path = os.path.join(load_path, hparams.PARAMS_NAME)

    new_hparams = load_json(path)
    hparams_keys = vars(hparams).keys()

    for key, value in new_hparams.items():
        if key in skip_list or key not in hparams_keys:
            print("Skip {} because it not exists".format(key))  #json에 있지만, hparams에 없다는 의미
            continue

        if key not in ['xxxxx',]:  # update 하지 말아야 할 것을 지정할 수 있다.
            original_value = getattr(hparams, key)
            if original_value != value:
                print("UPDATE {}: {} -> {}".format(key, getattr(hparams, key), value))
                setattr(hparams, key, value)

def save_hparams(model_dir, hparams):
    
    param_path = os.path.join(model_dir, hparams.PARAMS_NAME)

    info = eval(hparams.to_json(),{'false': False, 'true': True, 'null': None})
    write_json(param_path, info)

    print(" [*] MODEL dir: {}".format(model_dir))
    print(" [*] PARAM path: {}".format(param_path))

def load_json(path, as_class=False, encoding='euc-kr'):
    import re
    with open(path,encoding=encoding) as f:
        content = f.read()
        content = re.sub(",\s*}", "}", content)
        content = re.sub(",\s*]", "]", content)

        if as_class:
            data = json.loads(content, object_hook=\
                    lambda data: namedtuple('Data', data.keys())(*data.values()))
        else:
            data = json.loads(content)

    return data   
def write_json(path, data):
    with open(path, 'w',encoding='utf-8') as f:
        json.dump(data, f, indent=4, sort_keys=True, ensure_ascii=False)





def calc_num_batches(total_num, batch_size):
    '''Calculates the number of batches.
    total_num: total sample number
    batch_size

    Returns
    number of batches, allowing for remainders.'''
    return total_num // batch_size + int(total_num % batch_size != 0)

def convert_idx_to_token_tensor(inputs, idx2token):
    '''Converts int32 tensor to string tensor.
    inputs: 1d int32 tensor. indices.
    idx2token: dictionary

    Returns
    1d string tensor.
    '''
    def my_func(inputs):
        return " ".join(idx2token[elem] for elem in inputs)

    return tf.py_func(my_func, [inputs], tf.string)

# # def pad(x, maxlen):
# #     '''Pads x, list of sequences, and make it as a numpy array.
# #     x: list of sequences. e.g., [[2, 3, 4], [5, 6, 7, 8, 9], ...]
# #     maxlen: scalar
# #
# #     Returns
# #     numpy int32 array of (len(x), maxlen)
# #     '''
# #     padded = []
# #     for seq in x:
# #         seq += [0] * (maxlen - len(seq))
# #         padded.append(seq)
# #
# #     arry = np.array(padded, np.int32)
# #     assert arry.shape == (len(x), maxlen), "Failed to make an array"
#
#     return arry

def postprocess(hypotheses, idx2token):
    '''Processes translation outputs.
    hypotheses: list of encoded predictions
    idx2token: dictionary

    Returns
    processed hypotheses
    '''
    _hypotheses = []
    for h in hypotheses:
        sent = "".join(idx2token[idx] for idx in h)
        sent = sent.split("</s>")[0].strip()
        sent = sent.replace("▁", " ") # remove bpe symbols
        _hypotheses.append(sent.strip())
    return _hypotheses


def save_variable_specs(fpath):
    '''Saves information about variables such as
    their name, shape, and total parameter number
    fpath: string. output file path

    Writes
    a text file named fpath.
    '''
    def _get_size(shp):
        '''Gets size of tensor shape
        shp: TensorShape

        Returns
        size
        '''
        size = 1
        for d in range(len(shp)):
            size *=shp[d]
        return size

    params, num_params = [], 0
    for v in tf.global_variables():
        params.append("{}==={}".format(v.name, v.shape))
        num_params += _get_size(v.shape)
    print("num_params: ", num_params)
    with open(fpath, 'w') as fout:
        fout.write("num_params: {}\n".format(num_params))
        fout.write("\n".join(params))
    logging.info("Variables info has been saved.")

def get_hypotheses(num_batches, num_samples, sess, tensor, dict):
    '''Gets hypotheses.
    num_batches: scalar.
    num_samples: scalar.
    sess: tensorflow sess object
    tensor: target tensor to fetch
    dict: idx2token dictionary

    Returns
    hypotheses: list of sents
    '''
    hypotheses = []
    for _ in range(num_batches):
        h = sess.run(tensor)
        hypotheses.extend(h.tolist())
    hypotheses = postprocess(hypotheses, dict)

    return hypotheses[:num_samples]

def calc_bleu(ref, translation):
    '''Calculates bleu score and appends the report to translation
    ref: reference file path
    translation: model output file path

    Returns
    translation that the bleu score is appended to'''
    get_bleu_score = "perl multi-bleu.perl {} < {} > {}".format(ref, translation, "temp")
    os.system(get_bleu_score)
    bleu_score_report = open("temp", "r").read()
    with open(translation, "a") as fout:
        fout.write("\n{}".format(bleu_score_report))
    try:
        score = re.findall("BLEU = ([^,]+)", bleu_score_report)[0]
        new_translation = translation + "B{}".format(score)
        os.system("mv {} {}".format(translation, new_translation))
        os.remove(translation)

    except: pass
    os.remove("temp")


# def get_inference_variables(ckpt, filter):
#     reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
#     var_to_shape_map = reader.get_variable_to_shape_map()
#     vars = [v for v in sorted(var_to_shape_map) if filter not in v]
#     return vars


def seq_to_index(s,word_to_index):
    s = s.lower()
    return [word_to_index[x] for x in s]

def index_to_seq(t,index_to_word,word_to_index):
    return "".join([index_to_word[x] for x in t if x != word_to_index['<eos>']])
