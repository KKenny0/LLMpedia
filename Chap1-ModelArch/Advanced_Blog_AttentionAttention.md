## 注意力机制

> [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/#multi-head-self-attention)
 
简而言之，深度学习中的注意力可以被广义地解释为重要性权重向量：为了预测或推断一个元素，例如图像中的一个像素或句子中的一个单词，我们使用注意力向量来估计它与其他元素的相关性（或 "关注"，您可能在许多论文中读到过），并将它们的值之和经注意力向量加权后作为目标的近似值。

## 为翻译而生

注意力机制的诞生是为了帮助神经机器翻译（NMT）记忆长句。
注意力发明的秘诀是在上下文向量和整个源输入之间创建捷径，而不是从编码器的最后一个隐藏状态中建立一个单一的上下文向量。
这些捷径连接的权重可针对每个输出元素进行定制。

虽然上下文向量可以访问整个输入序列，但我们不必担心遗忘问题。
源代码和目标代码之间的对齐是由上下文向量学习和控制的。从本质上讲，上下文向量需要消耗三项信息：
- 编码器隐藏状态
- 解码器隐藏状态
- 源和目标之间的对齐

![The encoder-decoder model with additive attention mechanism in Bahdanau et al., 2015.](https://lilianweng.github.io/posts/2018-06-24-attention/encoder-decoder-attention.png)


## 一系列注意力机制

### 概括

下面是几种流行的注意力机制和相应的对齐评分函数的汇总表：


| 名称                     |                      对齐评分函数                      |                          引用                           |
|:-----------------------|:------------------------------------------------:|:-----------------------------------------------------:|
| Content-base attention |      $$score(s_t, h_i) = cosine[s_t, h_i]$$      |     [Graves2014](https://arxiv.org/abs/1410.5401)     |
| Additive(*)            | $$score(s_t, h_i) = v^T_atanh(W_a[s_{t-1};h_i]$$ |  [Bahdanau2015](https://arxiv.org/pdf/1409.0473.pdf)  |
| Location-Base          |        $$\alpha_{t,i} = softmax(W_as_t)$$        |   [Luong2015](https://arxiv.org/pdf/1508.04025.pdf)   |
| General                |        $$score(s_t, h_i) = s^T_tW_ah_i$$         |   [Luong2015](https://arxiv.org/pdf/1508.04025.pdf)   |
| Dot-Product            |          $$score(s_t, h_i) = s^T_th_i$$          |   [Luong2015](https://arxiv.org/pdf/1508.04025.pdf)   |
| Scaled Dot-Product(^)  | $$score(s_t, h_i) = \frac{s^T_th_i}{\sqrt{n}}$$  | [Xu2015](http://proceedings.mlr.press/v37/xuc15.pdf)  |

以下是更广泛类别的注意力机制的摘要：

| 名称                |                          对齐评分函数                          |                          引用                          |
|:------------------|:--------------------------------------------------------:|:----------------------------------------------------:|
| Self-Attention(&) | 关联同一输入序列的不同位置。理论上，自注意力可以采用上述任何评分函数，但只需将目标序列替换为相同的输入序列即可。 |  [Cheng2016](https://arxiv.org/pdf/1601.06733.pdf)   |
| Global/Soft       |                       关注整个输入状态空间。                        | [Xu2015](http://proceedings.mlr.press/v37/xuc15.pdf) |
| Local/Hard        |                关注输入状态空间部分；即输入图像的一个 patch。                |  [Luong2015](https://arxiv.org/pdf/1508.04025.pdf)   |


### Self-Attention

自注意力，也称为内部注意力，是一种将单个序列的不同位置相关联的注意力机制，以便计算同一序列的表示。
它已被证明在机器阅读、抽象概括或图像描述生成中非常有用。

在下面的例子中，自注意力机制使我们能够学习当前单词和句子前一部分之间的相关性。
![The current word is in red and the size of the blue shade indicates the activation level.](https://lilianweng.github.io/posts/2018-06-24-attention/cheng2016-fig1.png)


### Soft vs Hard Attention
本文首先根据注意力是访问整个图像还是仅访问一个补丁，提出了“软”注意力和“硬”注意力之间的区别：
- **软**注意力：学习对齐权重，并将其 "柔和 "地置于源图像中的所有斑块上；这与 [Bahdanau et al., 2015](https://arxiv.org/abs/1409.0473) 中的关注类型基本相同。
  - 优点：模型平滑且可微分
  - 缺点：当源输入很大时，成本昂贵
- **硬**注意力：一次只选择要注意的图像的一个 patch
  - 优点：推理时的计算量较少
  - 缺点：该模型是不可微分的，需要更复杂的技术（例如方差减少或强化学习）来训练


### Global vs Local Attention

[Luong, et al., 2015](https://arxiv.org/pdf/1508.04025.pdf) 提出了“全局”和“局部”注意力。
全局注意力类似于软注意力，而局部注意力是硬注意力和软注意力的有趣混合，是对硬注意力的改进，使其可微分：模型首先预测当前目标词的单一对齐位置，然后使用以源位置为中心的窗口计算上下文向量。

![Global vs local attention ](https://lilianweng.github.io/posts/2018-06-24-attention/luong2015-fig2-3.png)


