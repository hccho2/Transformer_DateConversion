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
- 고정 방식: 고정된 길이가 될 수 있도록 padding을 한다.

## Data Preprocessing: Tensorflow Tokenizer vs torchtext




## References
- <https://www.tensorflow.org/tutorials/text/transformer>
- <https://pytorch.org/tutorials/beginner/transformer_tutorial.html>