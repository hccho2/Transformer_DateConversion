Tensorflow High Level API인 `tf.estimator.Estimator`를 활용.
- `tf.estimator.Estimator`는 train 과정을 모니터링하기 편하고, train을 이어서 하는 것도 편리하다.
- `tf.estimator.Estimator`가 train 중에 log를 보여주지만, log 내용을 수정하려면 추가적인 hooking api를 사용해야 하는 번거로움이 있다. 
- `tf.train.LoggingTensorHook`, `tf.contrib.estimator.add_metrics`를 이용하면 된다.

- train
> python train.py

- test
> python eval_pred.py