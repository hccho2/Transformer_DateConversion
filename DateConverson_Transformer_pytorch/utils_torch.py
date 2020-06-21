import torch
import torchtext
import pandas as pd
from natsort import natsorted
import os, pickle,random
from glob import glob

def get_latest_model(model_path):
    model_list = glob(os.path.join(model_path, '*.pth'))
    if len(model_list) <=0:
        print('No Model Found!!!')
        return None
    model_list = natsorted(model_list)

    return model_list[-1]

class DateDataset(torchtext.data.Dataset):
    def __init__(self, data_filename, fields, strip_flag=False, **kwargs):
        # fields: [('id', ID), ('text', TEXT), ('label', LABEL)]
        examples = []
        train_data = pd.read_csv(data_filename, header=None, delimiter='_' )
        if len(fields)==2:
            inputs, targets = train_data[0].tolist(), train_data[1].tolist()
    
            for line in zip(inputs,targets):
                if strip_flag: line = [l.strip() for l in line]
                examples.append(torchtext.data.Example.fromlist(line, fields))
        else:
            inputs = train_data[0].tolist()
            for line in inputs:
                if strip_flag: line = line.strip()
                examples.append(torchtext.data.Example.fromlist([line], fields))          
        
        super(DateDataset, self).__init__(examples, fields, **kwargs)




def load_data(data_filename,vocab_filename, fields,strip_flag, train_flag=False,batch_size=32,device=None):
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = DateDataset(data_filename, fields,strip_flag)

    
    TEXT = fields[0][1]
    
    # vocab --> pickle파일
    if (not (os.path.exists(vocab_filename))):
        TEXT.build_vocab(train_data, min_freq=1, max_size=100)   # build_vocab 단계를 거처야, 단어가 숫자로 mapping된다.
        # inference를 대비하여 vocab를 저장해 두어야 한다.
        with open(vocab_filename, 'wb') as f:
            pickle.dump(TEXT.vocab, f) 
    
    else:
        with open(vocab_filename,'rb') as f:
            TEXT.vocab = pickle.load(f)
    
    vocab_size = len(TEXT.vocab)
    print('단어 집합의 크기 : {}'.format(vocab_size))
    
    print(dict(TEXT.vocab.stoi))  # 단어 dict   ----> inference를 대비해서 저장해 두어야 함.
    
    
    if not train_flag:
        return train_data, TEXT
    
    # vocab 생성 후, 쪼개기...
    train_data, valid_data = train_data.split(split_ratio=0.9,random_state=random.seed(100))
    print('train size: ', len(train_data), 'valid size: ', len(valid_data))
    
    train_loader = torchtext.data.Iterator(dataset=train_data, batch_size = batch_size,shuffle=True,device=device)
    valid_loader = torchtext.data.Iterator(dataset=valid_data, batch_size = len(valid_data),shuffle=False,device=device)
    

    return train_loader, valid_loader, TEXT

def load_data_test():
    #TEXT = torchtext.data.Field(sequential=True, tokenize=list,batch_first=True,include_lengths=False,lower=True)
    TEXT = torchtext.data.Field(sequential=True, tokenize=list,batch_first=False,include_lengths=False,init_token='<sos>',eos_token='<eos>',lower=True)  # src, target 모두에 sos,eos가 붇는다.
    fields = [('src',TEXT),('target',TEXT)]
    
    train_loader, valid_loader, TEXT = load_data(data_filename='date.txt',vocab_filename = 'vocab.pickle', fields=fields,
                                                 strip_flag= True,train_flag=True,batch_size=4)

    
    vocab_size = len(TEXT.vocab)

    for i, d in enumerate(train_loader):
        print(d.src.shape, d.target.shape, d.src, d.target)
        if i>2: break

if __name__ == '__main__':
    load_data_test()
    

