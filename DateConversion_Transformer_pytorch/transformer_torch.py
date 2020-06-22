
import torch
from torch import nn,optim
import math
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
    def __init__(self,hp,vocab_size,padding_id=None, device=None):
        super(Transformer,self).__init__()
        if device is None: 
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.padding_id = padding_id
        d_model = hp['d_model']
        nhead = hp['nhead']
        num_encoder_layers = hp['num_encoder_layers']
        num_decoder_layers = hp['num_decoder_layers']
        dim_feedforward = hp['dim_feedforward']
        drop_rate = hp['drop_rate']
        input_length = hp['INPUT_LENGTH']
        output_length = hp['OUTPUT_LENGTH']
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_positional_encoding = PositionalEncoding(d_model,drop_rate,input_length)
        self.decoder_positional_encoding = PositionalEncoding(d_model,drop_rate,output_length)
        
        
        self.transformer_model = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                                num_decoder_layers=num_decoder_layers,dim_feedforward=dim_feedforward,dropout=drop_rate)
        self.out_linear = nn.Linear(d_model, vocab_size)
        torch.nn.init.zeros_(self.out_linear.bias)
        
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.tgt_mask = self.transformer_model.generate_square_subsequent_mask(output_length).to(self.device)
        
    def forward(self,ecoder_inputs, decoder_inputs,input_length=None):
        x = self.embedding(ecoder_inputs)  # (29,N)  ---> (29,N,hidden_dim)
        x = self.encoder_positional_encoding(x)
        y = self.embedding(decoder_inputs) # (11,N)  ---> (11,N,hidden_dim)
        y = self.decoder_positional_encoding(y)
        if input_length:
            # inferece에서 mask를 적용하지 않으면, 결과가 안 좋다.
            tgt_mask = self.transformer_model.generate_square_subsequent_mask(input_length).to(self.device)
        else:
            tgt_mask = self.tgt_mask
        
        if self.padding_id is not None:
            src_key_padding_mask = torch.eq(ecoder_inputs,self.padding_id).T.to(self.device)   # (N,Te): src_lenght Te가 mini batch마다 변한다.
        else:
            src_key_padding_mask = None

        outputs = self.transformer_model(x,y,src_mask=None, tgt_mask=tgt_mask,src_key_padding_mask=src_key_padding_mask)  # tgt_mask필요, (T,N,D)
        outputs = self.out_linear(outputs) # (T,N,vocab_size)
        return outputs





