# Time-Series-Forecasting-with-Transformers

We test the ability of the Transformer architecture for Time-Series forecasting. Transformers and the underlying self-attention mechansim has proven to be very successful in the domain of NLP using for example very deep Transformers in BERT.

Inspired by this [paper](https://arxiv.org/pdf/2001.08317.pdf), we try to adapt the Transformer to forecast bike share demand. We just use the encoding architecture of the Transformer and feed the encoding to a dense layer to do multistep forecasting, contrary to the paper where they also use the decoder in the manner of a seq2seq model. 

We see that this model actually does a very good job at least on this kind of data. It is very intersting to se how well this model can capture also long future horizons!

![a](https://github.com/chenkel-data/Time-Series-Forecasting-with-Transformers/blob/master/demand1.png)


The model seems to follow the patterns but only for times with large shares the model does not perform well. But those extreme events are handled differently.

![b](https://github.com/chenkel-data/Time-Series-Forecasting-with-Transformers/blob/master/demand2.png)


![c](https://github.com/chenkel-data/Time-Series-Forecasting-with-Transformers/blob/master/demand3.png)
