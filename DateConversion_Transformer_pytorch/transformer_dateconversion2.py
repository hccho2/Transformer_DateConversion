# coding: utf-8

'''
1. 입력 data의 input이  가변 길이 방식:  '_'로 input/target을 분리한다.

september 27, 1994   _1994-09-27
August 19, 2003    _2003-08-19
2/10/93   _1993-02-10
10/31/90    _1990-10-31


2. transformer_dateconversion.py에서는 decoder의 causal padding만 적용.
    여기서는 encoder input을 가변 길이로 받아 padding하고, src_key_padding_mask를 적용한다.
    모델에 입력되는 mini batch의  encoder input은 mini batch마다 길이가 다르다. mini batch내의 가장 긴 놈을 기준으로 padding되어 있다.
    

    TEXT에 include_lengths를 해서 sequece mask를 만들 수 있다.
    https://discuss.pytorch.org/t/pytorch-equivalent-for-tf-sequence-mask/39036/2
    여기서는 include_lengths를 하지 않고, padding id와 비교하여 만들어 사용한다.

'''

import torch
from torch import nn,optim
import torchtext
from hparams import HParams   # pip install hparams
import os, time
from utils_torch import load_data, DateDataset,get_latest_model
from transformer_torch import Transformer

print('torch version: {}, cuda vertion: {}'.format(torch.__version__, torch.version.cuda))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # --> device(type='cuda')
print('device: ', device, 'available: ', torch.cuda.is_available())

hp = HParams(
    data_filename = 'date.txt',
    vocab_filename = 'vocab2.pickle',
    model_save_dir = './saved_model2',
    batch_size = 128,
    num_epoch = 10,
    
    INPUT_LENGTH = 29,
    OUTPUT_LENGTH = 11,
    
    lr = 0.001,
    drop_rate = 0.025,
    d_model = 32,
    nhead = 8,
    num_encoder_layers = 6,
    num_decoder_layers = 6,
    dim_feedforward =  128,
)

if not os.path.exists(hp['model_save_dir']):
    os.makedirs(hp['model_save_dir'])

   

def train():
    TEXT = torchtext.data.Field(sequential=True, tokenize=list,batch_first=False,include_lengths=False,init_token='<sos>',eos_token='<eos>',lower=True)  # src, target 모두에 sos,eos가 붇는다.
    fields = [('src',TEXT),('target',TEXT)]
    
    train_loader, valid_loader, TEXT = load_data(data_filename=hp['data_filename'],vocab_filename = hp['vocab_filename'] , fields=fields, strip_flag = True, train_flag=True)
    vocab_size = len(TEXT.vocab)
    
    
    padding_id = TEXT.vocab.stoi['<pad>']
    model = Transformer(hp,vocab_size=vocab_size,padding_id = padding_id).to(device)

    

    optimizer = optim.Adam(model.parameters(),lr=hp['lr'])
    num_epoch = hp['num_epoch']
    start_epoch=1
    saved_model = get_latest_model(hp['model_save_dir'])
    if saved_model:
        model.load_state_dict(torch.load(saved_model, map_location = device))
        start_epoch = int(os.path.basename(saved_model).split('-')[-1].split('.')[0])+1  # 특수한 상황....
        print('Model Found. start_epoch: {}'.format(start_epoch))
    print('batch_size: {}, num_epoch: {}'.format(hp['batch_size'], num_epoch))
    
    
    model.train()
    
    s_time = time.time()
    step_count = 0
    for epoch in range(start_epoch, num_epoch+1):
        for i, d in enumerate(train_loader):
            step_count += 1
            optimizer.zero_grad()
            target = d.target  #(T,N)
            encoder_inputs = d.src[1:-1,:]  #(INPUT_LENGTH,N)   ---> '<sos>', '<eos>'제거후 입력.
            decoder_inputs = target[:-1,:]  # (OUTPUT_LENGTH,N)
            outputs = model(encoder_inputs,decoder_inputs)  # (T,N,D)
            
            loss = model.loss_fn(outputs.permute(1,2,0), target[1:,:].T)  # (T,N,D)  ---> CrossEntropyLoss에는 (N,D,T)를 넘겨야 한다. target에는 (N,T)
            loss.backward()
            optimizer.step()
            
            if step_count % 100 == 0:
                print('epoch: {:>3}, setp: {:>5}, loss: {:10.4f}, elapsed: {:>10}'.format(epoch, step_count, loss.item(), int(time.time()-s_time)), end='\t')
                predict = torch.argmax(outputs,-1).detach() # predict: (T,N), target: (T,N)
                print('seq_acc: {:10.4f}'.format((target[1:,:]==predict).prod(dim=0).float().mean().item()),end='\t')
                print(''.join([TEXT.vocab.itos[x] for x in encoder_inputs[:,0].to('cpu').numpy() if x != padding_id]),'|', ''.join([TEXT.vocab.itos[x] for x in target[:,0].to('cpu').numpy()]),'-->',''.join([TEXT.vocab.itos[x] for x in predict[:,0].to('cpu').numpy()]))
                
        if (epoch)%1==0:
            model.eval()
            with torch.no_grad():
                for _, valid_batch in enumerate(valid_loader):  # len(valid_loader) = 1
                    target = valid_batch.target  #(T,N)
                    encoder_inputs = valid_batch.src[1:-1,:]
                    decoder_inputs = target[:-1,:]
                    outputs = model(encoder_inputs,decoder_inputs)  # (T,N,D)                
                    predict = torch.argmax(outputs,-1).detach() # predict: (T,N), target: (T,N)
                     
                    valid_loss = model.loss_fn(outputs.permute(1,2,0), target[1:,:].T)
                    character_accuracy = (target[1:,:]==predict).float().mean().item()
                    sequence_accuracy = (target[1:,:]==predict).prod(dim=0).float().mean().item()
                     
                    print('valid_loss: {:10.4f},char_acc: {:10.4f}, seq_acc: {:10.4f} '.format(valid_loss,character_accuracy,sequence_accuracy))
            model.train()
             
        if (epoch)%5==0:         
            torch.save(model.state_dict(), os.path.join(hp['model_save_dir'], 'epoch-{}.pth'.format(epoch)))  # weights만 저장
            print('Model Saved-epoch-{}.pth'.format(epoch))


def test():
    #TEXT = torchtext.data.Field(sequential=True, tokenize=list,batch_first=False,include_lengths=False,init_token='<sos>',eos_token='<eos>',lower=True)  # src, target 모두에 sos,eos가 붇는다.
    #<sos>를 붙혀서 넣어면, 정확도가 떨어진다.
    
    #아래 코드에서 첫 토큰을 제거하지 않고 넣기 때문에, init_token을 지정하지 않는 것이 맞다.
    TEXT = torchtext.data.Field(sequential=True, tokenize=list,batch_first=False,include_lengths=False,lower=True,pad_token='<pad>')  # src, target 모두에 sos,eos가 붇는다.
    
    fields = [('src',TEXT)]    
    
    
    data, TEXT = load_data(data_filename='test_data.txt',vocab_filename = hp['vocab_filename'] , fields=fields, strip_flag = True, train_flag=False)
    vocab_size = len(TEXT.vocab)
    data_loader = torchtext.data.Iterator(dataset=data, batch_size = len(data),shuffle=False)
    
    raw_data = []
    for i,d in enumerate(data):
        raw_data.append(''.join(d.src))
        #print(d.src)
    
    padding_id = TEXT.vocab.stoi['<pad>']
    model = Transformer(hp,vocab_size=vocab_size,padding_id = padding_id).to(device)
    model.eval()
    saved_model = get_latest_model(hp['model_save_dir'])
    if saved_model:
        model.load_state_dict(torch.load(saved_model, map_location = device))
        print('Model loaded!!!', saved_model)
    else:
        print('No Model Found!!!')
        exit()
        
    batch_size = len(data)
    encoder_inputs = next(iter(data_loader))
    encoder_inputs = encoder_inputs.src.to(device)
    decoder_inputs_init = torch.tensor([TEXT.vocab.stoi['<sos>']] * batch_size).view(1,-1).to(device)
    decoder_inputs = decoder_inputs_init
    for i in range(hp['OUTPUT_LENGTH']):
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