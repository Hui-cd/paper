# Transformer Reading Note
## Model Architecture
<img src="image\transformer_model.png" width="400" height="300">

### Encoder Architecture
encoder is composed of a stack of N = 6 identical layers, each layer has 2 sub-layers which are multi-head self-attention and position-wise fully connected feed-forward network.

encoder由6层$layer$组成，其中每个$layer$由两个子层组成，其分别为 multi-head self-attention 和position-wise fully connected feed-forward network.

### Decoder Architecture
decoder is composed of a stack of N = 6 identical layers, each layer have 3 sub-layers, and the third layer is perform mulit-head attention layer over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. it also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.
#### Attention
