Dataset imdb Reviews.
Task: sentiment classification.

I have taken two approaches. First was using GLOVE
words embeddings matrix as a pretrained embedding layer. I had to use tokenizer to transform raw dataset
and next recreate embedding matrix using tokenizer info and downloaded files. Everything had to be done in
memory efficient way, because I had only 16GB of RAM and model and dataset could easly exceed it. I have
set embedding layer to >not trainable< then I have used several architectures based on LSTM layers. It
occurs that this approach is inefficient - only about 70% accuraccy in peak.

So the next approach was to use subword embeddings. This time I used preprocesed dataset in form of
numerical tensors that keep information about subwords. Then I have added trainable embedding layer to the
model and had been experimenting with different LSTM layers, L2/L1regularization, dropout. I used only
ReLu as activation for hidden layers and sigmoid for output layer. I haven't use RNN due to extremally slow
learning rate (vanishing gradients, LSTMs solve this problem thanks to gates and memory cells).
It turns out that subword embedding, 4 bidirectional LSTM layers of 3x64+1x128 output units and two FC
layers (256) network was able to achieve highest accuraccy ~85%

LSTMs are generally slow, because there are a lot of reccurent layers so training took a lot of time even on
GPU. There was an overfit at 10 epoch, so there is no point in further training in this setup.
