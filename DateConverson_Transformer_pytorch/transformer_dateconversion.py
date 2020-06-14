# coding: utf-8

'''
https://discuss.pytorch.org/t/nn-transformer-explaination/53175


1. nn.Transformer는 positional embedding을 하지 않는다. positional embedding 처리한 것을 넘겨주어야 한다.
2. underfitting: hidden_dim=64, drop_rate=0.3 ---> train은 되는데, validation data에서 loss가 높고, accuracy가 5%미만
    --> drop_rate=0 --> 5epoch이하에서 train, valid 모두 우수
    --> hidden_dim = 8 ---> train안됨
    --> hidden_dim = 32, drop_rate=0   ---> 5 epoch train하면 됨
    --> hidden_dim = 32, drop_rate=0.1 ---> 15 epoch이면 train, valid 모두 100%

'''


import numpy as np
import torch
from torch import nn,optim
import torchtext
import pandas as pd
from glob import glob
from natsort import natsorted
import pickle,os, random,time,math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # --> device(type='cuda')
vocab_filename = 'vocab.pickle'
model_save_dir = './saved_model'
INPUT_LENGTH = 29
OUTPUT_LENGTH = 11

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

def get_latest_model(model_path):
    model_list = glob(os.path.join(model_path, '*.pth'))
    if len(model_list) <=0:
        print('No Model Found!!!')
        return None
    model_list = natsorted(model_list)

    return model_list[-1]


class DateDataset(torchtext.data.Dataset):
    def __init__(self, data_filename, fields, **kwargs):
        # fields: [('id', ID), ('text', TEXT), ('label', LABEL)]
        examples = []
        train_data = pd.read_csv(data_filename, header=None, delimiter='_' )
        if len(fields)==2:
            inputs, targets = train_data[0].tolist(), train_data[1].tolist()
    
            for line in zip(inputs,targets):
                examples.append(torchtext.data.Example.fromlist(line, fields))
        else:
            inputs = train_data[0].tolist()
            for line in inputs:
                examples.append(torchtext.data.Example.fromlist([line], fields))          
        
        super(DateDataset, self).__init__(examples, fields, **kwargs)

def load_data(data_filename,fields, train_flag=False):
    train_data = DateDataset(data_filename, fields)
    
#     for i, d in enumerate(train_data):
#         print(i, d.src)  # print(i, d.src,d.target)
#         if i>=2: break
    
    
    
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
    
    batch_size = 32
    train_loader = torchtext.data.Iterator(dataset=train_data, batch_size = batch_size,shuffle=True)
    valid_loader = torchtext.data.Iterator(dataset=valid_data, batch_size = len(valid_data),shuffle=False)
    
    
    # print('='*20, 'train_loader test')
    # for i, d in enumerate(train_loader):
    #     print(i,d.src.shape, d.target.shape, d.src, d.target)   # d.text[0], d.text[1] ----> Field에서 include_lengths=True로 설정.
    #     if i>=2: break
    #   
    # print('='*20, 'valid_loader test')
    # for i, d in enumerate(valid_loader):
    #     print(d.src.shape,d.target.shape,d.target)

    return train_loader, valid_loader, TEXT


#################
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self,vocab_size):
        super(Transformer,self).__init__()
        hidden_dim = 32
        drop_rate = 0.0
        decoder_length = 11
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_positional_encoding = PositionalEncoding(hidden_dim,drop_rate,INPUT_LENGTH)
        self.decoder_positional_encoding = PositionalEncoding(hidden_dim,drop_rate,OUTPUT_LENGTH)
        
        
        self.transformer_model = nn.Transformer(d_model=hidden_dim, nhead=8, num_encoder_layers=6,num_decoder_layers=6,dim_feedforward=128,dropout=drop_rate)
        self.out_linear = nn.Linear(hidden_dim, vocab_size)
        torch.nn.init.zeros_(self.out_linear.bias)
        
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.tgt_mask = self.transformer_model.generate_square_subsequent_mask(decoder_length).to(device)
        
    def forward(self,ecoder_inputs, decoder_inputs,input_length=None):
        x = self.embedding(ecoder_inputs)  # (29,N)  ---> (29,N,hidden_dim)
        x = self.encoder_positional_encoding(x)
        y = self.embedding(decoder_inputs) # (11,N)  ---> (11,N,hidden_dim)
        y = self.decoder_positional_encoding(y)
        if input_length:
            # inferece에서 mask를 적용하지 않으면, 결과가 안 좋다.
            tgt_mask = self.transformer_model.generate_square_subsequent_mask(input_length).to(device)
        else:
            tgt_mask = self.tgt_mask
        
        
        outputs = self.transformer_model(x,y,src_mask=None, tgt_mask=tgt_mask)  # tgt_mask필요, (T,N,D)
        outputs = self.out_linear(outputs) # (T,N,vocab_size)
        return outputs
    
    

def train():
    #TEXT = torchtext.data.Field(sequential=True, tokenize=list,batch_first=True,include_lengths=False,lower=True)
    TEXT = torchtext.data.Field(sequential=True, tokenize=list,batch_first=False,include_lengths=False,init_token='<sos>',eos_token='<eos>',lower=True)  # src, target 모두에 sos,eos가 붇는다.
    fields = [('src',TEXT),('target',TEXT)]
    
    train_loader, valid_loader, TEXT = load_data(data_filename='date.txt',fields=fields, train_flag=True)
    vocab_size = len(TEXT.vocab)
    

    model = Transformer(vocab_size=vocab_size).to(device)
#     for name, param in model.named_parameters():
#         print (name, param.shape)
    

    optimizer = optim.Adam(model.parameters(),lr=0.001)
    num_epoch = 200
    start_epoch=1
    saved_model = get_latest_model(model_save_dir)
    if saved_model:
        model.load_state_dict(torch.load(saved_model, map_location = device))
        start_epoch = int(os.path.basename(saved_model).split('-')[-1].split('.')[0])+1  # 특수한 상황....
        print('Model Found. start_epoch: {}'.format(start_epoch))
    model.train()
    
    s_time = time.time()
    step_count = 0
    for epoch in range(start_epoch, num_epoch+1):
        for i, d in enumerate(train_loader):
            step_count += 1
            optimizer.zero_grad()
            target = d.target.to(device)  #(T,N)
            encoder_inputs = d.src[1:-1,:].to(device)  #(INPUT_LENGTH,N)
            decoder_inputs = target[:-1,:]  # (OUTPUT_LENGTH,N)
            outputs = model(encoder_inputs,decoder_inputs)  # (T,N,D)
            
            loss = model.loss_fn(outputs.permute(1,2,0), target[1:,:].T)  # (T,N,D)  ---> CrossEntropyLoss에는 (N,D,T)를 넘겨야 한다. target에는 (N,T)
            loss.backward()
            optimizer.step()
            
            if step_count % 500 == 0:
                print('epoch: {:>3}, setp: {:>5}, loss: {:10.4f}, elapsed: {:>10}'.format(epoch, step_count, loss.item(), int(time.time()-s_time)), end='\t')
                predict = torch.argmax(outputs,-1).detach() # predict: (T,N), target: (T,N)
                print('seq_acc: {:10.4f}'.format((target[1:,:]==predict).prod(dim=0).float().mean().item()),end='\t')
                print(''.join([TEXT.vocab.itos[x] for x in encoder_inputs[:,0].to('cpu').numpy()]),'|', ''.join([TEXT.vocab.itos[x] for x in target[:,0].to('cpu').numpy()]),'-->',''.join([TEXT.vocab.itos[x] for x in predict[:,0].to('cpu').numpy()]))
                
        if (epoch)%1==0:
            model.eval()
            with torch.no_grad():
                for _, valid_batch in enumerate(valid_loader):  # len(valid_loader) = 1
                    target = valid_batch.target.to(device)  #(T,N)
                    encoder_inputs = valid_batch.src[1:-1,:].to(device)
                    decoder_inputs = target[:-1,:]
                    outputs = model(encoder_inputs,decoder_inputs)  # (T,N,D)                
                    predict = torch.argmax(outputs,-1).detach() # predict: (T,N), target: (T,N)
                    
                    valid_loss = model.loss_fn(outputs.permute(1,2,0), target[1:,:].T)
                    character_accuracy = (target[1:,:]==predict).float().mean().item()
                    sequence_accuracy = (target[1:,:]==predict).prod(dim=0).float().mean().item()
                    
                    print('valid_loss: {:10.4f},char_acc: {:10.4f}, seq_acc: {:10.4f} '.format(valid_loss,character_accuracy,sequence_accuracy))
            model.train()
            
        if (epoch)%5==0:         
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'epoch-{}.pth'.format(epoch)))  # weights만 저장
            print('Model Saved-epoch-{}.pth'.format(epoch))


def test():

    TEXT = torchtext.data.Field(sequential=True, tokenize=list,batch_first=False,include_lengths=False,lower=True,fix_length=INPUT_LENGTH,pad_token=' ')  # src, target 모두에 sos,eos가 붇는다.
    fields = [('src',TEXT)]    
    
    
    data, TEXT = load_data('test_data.txt', fields, train_flag=False)
    vocab_size = len(TEXT.vocab)
    data_loader = torchtext.data.Iterator(dataset=data, batch_size = len(data),shuffle=False)
    
    raw_data = []
    for i,d in enumerate(data):
        raw_data.append(''.join(d.src))
        #print(d.src)
    
    model = Transformer(vocab_size=vocab_size).to(device)
    model.eval()
    saved_model = get_latest_model(model_save_dir)
    if saved_model:
        model.load_state_dict(torch.load(saved_model, map_location = device))
        print('Model loaded!!!', saved_model)
    else:
        exit()
        
    batch_size = len(data)
    encoder_inputs = next(iter(data_loader))
    encoder_inputs = encoder_inputs.src.to(device)
    decoder_inputs_init = torch.tensor([TEXT.vocab.stoi['<sos>']] * batch_size).view(1,-1).to(device)
    decoder_inputs = decoder_inputs_init
    for i in range(OUTPUT_LENGTH):
        outputs = model(encoder_inputs, decoder_inputs,input_length=i+1)
        outputs = torch.argmax(outputs,-1)
        decoder_inputs = torch.cat([decoder_inputs_init,outputs],0)
    
    
    #print(decoder_inputs.T)
    results = []
    for i in range(batch_size):
        results.append( ''.join([TEXT.vocab.itos[x] for x in decoder_inputs.T[i] if x not in [2,3]]) )
    
    for x,y in zip(raw_data,results):
        print('{:<30} --> {:>15}'.format(x,y))

if __name__ == '__main__':
    #train()
    test()