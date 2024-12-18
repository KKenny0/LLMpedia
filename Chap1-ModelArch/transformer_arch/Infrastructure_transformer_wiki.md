> [Transformer (deep learning architecture)](https://arc.net/l/quote/ikjwlynv)


## Architecture
主要组件:
- Tokenizers:
- Embedding layer: convert tokens and positions of the tokens into vector representations
- Transformer layers: carry out repeated transformations on the vector representations, extracting more and more linguistic information.
  - Alternating attention and feedforward layers.
  - Two major types transformer layers: encoder layers and decoder layers
- Un-embedding layer: convert the final vector representations back to a probability distribution over the tokens


### Embedding
Each token is converted into an embedding vector via a lookup table.
Equivalently stated, it multiplies a `one-hot representation` of the token by an embedding matrix $$M$$.


### Un-embedding
Whereas an embedding layer converts a token into a vector, an un-embedding layer converts a vector into a probability distribution over tokens.

The un-embedding layer is a linear-softmax layer:
$$
UnEmbed(x) = softmax(xW+b)
$$


### Encoder-decoder (overview)
The original transformer model used an **encoder-decoder** architecture.
The encoder consists of encoding layers that process all the input tokens together one layer after another,
while the decoder consists of decoding layers that iteratively process the encoder's output and the decoder's output tokens so far.

Both the encoder and decoder layers have a **feed-forward neural network** for *additional processing* of their outputs and
contain **residual connections** and **layer normalization** steps.


### Feedforward network

The feedforward network (FFN) modules in a Transformer are **2-layered multilayer perceptrons**.
![2-layered multilayer perceptrons](https://wikimedia.org/api/rest_v1/media/math/render/svg/3018d1cafd676461ec6a1927aff651d73ce78377)


### Scaled dot-product attention

#### Attention head
The attention mechanism used in the Transformer architecture are **scaled dot-product attention units**.

For each vector $$x_{i, query}$$ in the query sequence, it is multiplied by a matrix $$W^Q$$ to produce a query vector
$$q_i = x_{i, query}W^Q$$. The matrix of all query vectors is the query matrix: $$Q = X_{query}W^Q$$.

Similarly, we construct the key matrix $$K = X_{key}W^K$$ and the value matrix $$V = X_{value}W^V$$.

It is usually the case that all $${\displaystyle W^{Q},W^{K},W^{V}}$$ are square matrices.

Attention weights are calculated using the query and key vectors：the attention weight from token $$i$$ to token $$j$$ 
is the dot product between $$q_i$$ and $$k_j$$.
The attention weights are divided by the square root of the dimension of the key vectors, $${\displaystyle {\sqrt {d_{k}}}}$$, 
which stabilizes gradients during training, and passed through a softmax which normalizes the weights

The matrices $${\displaystyle Q}$$, $${\displaystyle K}$$ and $${\displaystyle V}$$ are defined as the matrices 
where the {\displaystyle i}th rows are vectors $${\displaystyle q_{i}}$$, $${\displaystyle k_{i}}$$, and $${\displaystyle v_{i}}$$ respectively. 
Then we can represent the attention as:
![attention head](https://wikimedia.org/api/rest_v1/media/math/render/svg/0b2afc7240eb97375a384b1628c18438e3068e3f)

The attention mechanism requires the following three equalities to hold:
![attention mechanism equalities](https://wikimedia.org/api/rest_v1/media/math/render/svg/b25b25f22a8a09c26350d2f065628dd4b3911669)

![Scaled dot-product attention, block diagram.](https://upload.wikimedia.org/wikipedia/commons/1/1b/Transformer%2C_attention_block_diagram.png)

If the attention is used in a *self-attention* fashion, then $$X_{query} = X_{key} = X_{value}$$.
If the attention is used in a *cross-attention* fashion, then usually $$X_{query} \neq X_{key} = X_{val}$$.

#### Multihead attention

In a transformer model, an attention head consist of three matrices: \(W^Q\), \(W^K\), and \(W^V\),
where \(W^Q\) and \(W^K\) determine the relevance between tokens for attention scoring,
and \(W^V\) along with \(W^O\) influences how attended tokens affect subsequent layers and output logits.

Multiple attention heads in a layer allow the model to capture different definitions of "relevance".
As tokens progress through layers, the scope of attention can expand, enabling the model to grasp more complex and long-range dependencies. 

The outputs from all attention heads are concatenated and fed into the feed-forward neural network layers.

Concretely, let the multiple attention heads be indexed by $${\displaystyle i}$$, then we have:
![multihead attention formula](https://wikimedia.org/api/rest_v1/media/math/render/svg/2e699202635f8646a75cf1b0658697c6825e123a)
where the matrix $$X$$ is the concatenation of word embeddings, and the matrices $${\displaystyle W_{i}^{Q},W_{i}^{K},W_{i}^{V}}$$ 
are "projection matrices" owned by individual attention head $${\displaystyle i}$$, and $${\displaystyle W^{O}}$$ is a final projection matrix owned by the whole multi-headed attention head.

It is theoretically possible for each attention head to have a different head dimension $${\displaystyle d_{\text{head}}}$$, but that is rarely the case in practice.

As an example, in the smallest GPT-2 model, there are only self-attention mechanisms.
It has the following dimensions:
$$
d_{emb} = 768, n_{head}=12, d_{head}=64
$$
Since, $${\displaystyle 12\times 64=768}$$, its output projection matrix, 
$${\displaystyle W^{O}\in \mathbb {R} ^{(12\times 64)\times 768}}$$ is a square matrix.

#### Masked attention
It may be necessary to cut out attention links between some word-pairs.
For example, the decoder, when decoding for the token position $$t$$,
should not have access to the token at position $$t+1$$.
This may be accomplished before the softmax stage by adding a mask matrix $$M$$ 
that is $${\displaystyle -\infty }$$ at entries where the attention link must be cut, and $${\displaystyle 0}$$ at other places: 
![masked attention](https://wikimedia.org/api/rest_v1/media/math/render/svg/8d99a80dbf8da6e52c37ba3c9965387a19f82975)

For example, the following matrix is commonly used in decoder self-attention modules, called "causal masking":
![Causal masking](https://wikimedia.org/api/rest_v1/media/math/render/svg/981c71d86645b9f71d314dc671903905c0c30a9a)
In words, it means that each token can pay attention to itself, and every token before it, but not any after it. 


### Encoder
An encoder consists of an embedding layer, followed by multiple encoder layers.

Each encoder layer consists of **two major components**: a self-attention mechanism and a feed-forward layer.

The encoder layers are stacked.
The first encoder layer takes the sequence of input vectors from the embedding layer, producing a sequence of vectors. 
This sequence of vectors is processed by the second encoder, and so on. 
_The output from the final encoder layer is then used by the decoder._

As the encoder processes the entire input all at once, every token can attend to every other token (all-to-all attention), 
so there is no need for causal masking.

![One encoder layer](https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/Transformer%2C_one_encoder_block.png/440px-Transformer%2C_one_encoder_block.png)


### Decoder
Each decoder consists of **three major components**: a causally masked self-attention mechanism, a cross-attention mechanism, and a feed-forward neural network.

The decoder functions use an additional attention mechanism to draw relevant information from the encodings generated by the encoders.
This mechanism can also be called the _encoder-decoder attention_.

Like the first encoder, the first decoder takes positional information and embeddings of the output sequence as its input, rather than encodings.
The transformer must not use the current or future output to predict an output, so the output sequence must be partially masked to prevent this reverse information flow.
Thus, the self-attention module in the decoder is causally masked.

The cross-attention mechanism attends to the output vectors of the encoder, which is computed before the decoder starts decoding.
Schematically, we have:
![cross attention schema](https://wikimedia.org/api/rest_v1/media/math/render/svg/53517bba056de79c117a30490add4f73868af2b7)
where $${\displaystyle H^{E}}$$ is the matrix with rows being the output vectors from the encoder.

The last decoder is followed by a final un-embedding layer to produce the output probabilities over the vocabulary. 
Then, one of the tokens is sampled according to the probability, and the decoder can be run again to produce the next token, etc, autoregressively generating output text.


## Full transformer architecture

### Sublayers
Each encoder layer contains 2 sublayers: the self-attention and the feedforward network.
Each decoder layer contains 3 sublayers: the causally masked self-attention, the cross-attention, and the feedforward network.

The final points of detail are the *residual connections* and *layer normalization (LayerNorm, or LN)*,
which while conceptually unnecessary, are necessary for numerical stability and convergence.

The original 2017 Transformer used the post-LN convention.
It was difficult to train and required careful hyperparameter tuning and a "warm-up" in learning rate,
where it starts small and gradually increases.
