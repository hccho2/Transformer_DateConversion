# Date Conversion Transformer: Tensorflow 2.x vs Pytorch vs Keras vs Tensorflow 1.x

## Requirements
- torchtext
- hparams
- pandas
- pytorch
- tensorflow 2.2


## 구현 목록
	- Pytorch: torch.nn.Transformer API를 사용.
	- Tensorflow 2.x
	- Keras
	- Tensorflow 1.x

## Data 만들기

## Padding
- 2가지 방식의 padding을 살펴보자.
![padding](./padding.png)
- 가변 길이 방식: mini batch 내에서 가장 긴 sequence를 기준으로 padding이 된다.
- 고정 길이 방식: 고정된 길이가 될 수 있도록 padding을 한다.
- 2가지 방식의 padding이 가능하도록 tensorflow, torchtext에서 api를 제공하고 있다.

## Data Preprocessing: Tensorflow Tokenizer vs torchtext
# Tensorflow
```
# coding: utf-8

import tensorflow as tf
from konlpy.tag import Okt
okt = Okt()

from tensorflow.keras import preprocessing
samples = ['너 오늘 아주 이뻐 보인다', 
           '나는 오늘 기분이 더러워', 
           '끝내주는데, 좋은 일이 있나봐', 
           '나 좋은 일이 생겼어', 
           '아 오늘 진짜 너무 많이 정말로 짜증나', 
           '환상적인데, 정말 좋은거 같아']

label = [[1], [0], [1], [1], [0], [1]]


morph_sample = [okt.morphs(x) for x in samples]


tokenizer = preprocessing.text.Tokenizer(oov_token="<UKN>")   # oov: out of vocabulary
tokenizer.fit_on_texts(samples+['SOS','EOS']) 
print(tokenizer.word_index)  # 0에는 아무것도 할당되어 있지 않다.  --> pad를 할당하면 된다.


word_to_index = tokenizer.word_index
word_to_index['PAD'] = 0
index_to_word = dict(map(reversed, word_to_index.items()))

print('word_to_index(pad): ', word_to_index)
print('index_to_word', index_to_word)

sequences = tokenizer.texts_to_sequences(morph_sample)  # 역변환: tokenizer.sequences_to_texts(sequences)
print(sequences)
[[5, 2, 6, 7, 8],
[14, 1, 2, 1, 1, 11],
[12, 1, 3, 4, 1, 1],
[14, 3, 4, 15],
[16, 2, 17, 18, 19, 20, 21],
[1, 1, 1, 1, 23, 3, 1, 25]]

```
- 위에서 얻은 `sequences`에 padding을 붙혀보자.
```
padded_sequence = preprocessing.sequence.pad_sequences(sequences, maxlen=15, padding='post',truncating='post')

[[ 5  2  6  7  8  0  0  0  0  0  0  0  0  0  0]
 [14  1  2  1  1 11  0  0  0  0  0  0  0  0  0]
 [12  1  3  4  1  1  0  0  0  0  0  0  0  0  0]
 [14  3  4 15  0  0  0  0  0  0  0  0  0  0  0]
 [16  2 17 18 19 20 21  0  0  0  0  0  0  0  0]
 [ 1  1  1  1 23  3  1 25  0  0  0  0  0  0  0]]


```




# torchtext



## References
- <https://www.tensorflow.org/tutorials/text/transformer>
- <https://pytorch.org/tutorials/beginner/transformer_tutorial.html>