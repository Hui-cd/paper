# GPT1
#### 3.1 无监督的预训练
给定无监督词库 token U = {U1,...,Un} (均是token)，最大化：
L1(U) = $\sum$logP(Ui|Ui-k,...Ui-1; Θ)

其中k是size of context window ,P 是模型在参数Θ下的条件概率，这些参数使用随机梯度下降训练得到的

GPT 是用来多层Transformer的解码器

h0 = UWe + Wp

hl = transformer_block(hl−1)∀i ∈ [1, n]

P (u) = softmax(hn WeT )

ho是transformer解码器的输入，其中Wp是position， We 是embedding,U = (U-k,...,U-1) 是token的上下文向量，

#### 3.2 监督微调
P(y|x1,...,xm) = softmax(hml Wy).

L2(C) = $\sum$logP(y|x1,...,xm).

# GPT2
