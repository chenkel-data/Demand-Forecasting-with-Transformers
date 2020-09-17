# Time-Series-Forecasting-with-Transformers

We test the ability to aply the Transformer architecture to Time-Series. Transformers and the underlying Self-attention mechansim has proven to be very successful in the domain of NLP using for example very deep Transformers in BERT.

Inspired by this paper, we try to adapt the Transformer to forecast bike share demand. We just use the encoding architecture of the Transformer and feed the encoding to a dense layer to do multistep forecasting, contrary to the paper where they use also the decoder in the manner for a seq2seq model. 

