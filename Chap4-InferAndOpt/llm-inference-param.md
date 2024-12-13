# LLM 推理常见参数

重点介绍 **top_k**、**top_p**、**temperature**、**repetition_penalty** 和 **greedy_search**。

## 背景介绍

常见的大型语言模型（LLM）通常只包含 Transformer 解码器（Decoder）。
在处理输入时，每个单词（Token）会先转换成一个嵌入向量（Token Embedding），然后这个向量被送入 Transformer 解码器。
解码器的最后一层输出的也是一个嵌入向量。
在预测下一个单词时，主要依赖于前一个单词的嵌入向量。

使用 Transformer 解码器进行文本生成的过程。简单来说，就是：
1. 输入句子"a robot must obey the orders given it"的词嵌入（embedding）；
2. 通过Transformer解码器生成每个词的新嵌入；
3. 使用最后一个词“it”的新嵌入生成新词“Okay”；
4. 将“Okay”的嵌入作为输入，生成下一个词“human”；
5. 重复步骤 3 和 4，继续生成新的词。

根据新生成的词嵌入（embedding）来生成下一个词（Token）的过程，具体来说是让**新生成的 embedding 与 token embedding 矩阵相乘**，
得到和词表中每个 Token 的相似性得分（`logits`），然后基于这个得分即可以选择生成新的 Token。


## Greedy Search

GreedySearch（贪心搜索）的思路就是**从相似性得分（logits）选择得分最高的 Token**（一般来说，都会将得分经过 softmax 层转换为概率分布，数值区间为`0~1`），直到结束。

在推理阶段，模型的权重都是确定的，并且也不会有 dropout 等其他随机性（忽略不可抗的硬件计算误差，如并行规约求和的累积误差等），
因此**如果是 greedy search，则对于同一输入，多次运行后模型的输出结果应该完全一致**。

**好处**：在**模型效果严格对齐时非常有必要**（比如将模型从 HF 模型转换为推理效率更高的 Faster Transformer 模型，并且使用 Faster Transformer 进行推理部署）。

**坏处**：**模型效果可能不是最优的，也会缺乏一定的多样性**，比如用同样的问题问 ChatGPT，其答案并不会每次都一样。


## Beam Search

Beam Search **不是每次都取得分最大的 Token，而是始终保留 `beam_size` 个得分最大的序列**。

共有 `“a”,“given”,“human”,“it”,“must”,“obey”,“Okay”,“orders”,“robot”,“the”,“.”,“EOS”` 这些 Tokens。
以输入序列 `“a”, “robot”, “must”, “obey”, “the”, “orders”, “given”, “it”` 为例:
1. **Step1**: 使用 Token “it”对应新生成的嵌入来计算 logits，最终`“Okay”`对应的得分 0.91 和`“.”`对应的得分 0.84 最高，所以选择`“Okay”`和`“.”`作为下一个 Token。
2. **Step2**: 分别使用 “Okay” 和 “.” 来计算 logits:
   - 对于“Okay”，最终“human”对应的得分0.89和“the”对应的得分0.72最高，对应候选序列：
     - “okay”+“human” = 0.91 * 0.89 = 0.8099
     - “okay”+“the” = 0.91 * 0.72 = 0.6552
   - 对于“.”，最终“the”对应的得分0.92和“EOS”对应的得分0.70最高，对应候选序列:
     - “.”+“the” = 0.84 * 0.92 = 0.7728
     - “.”+“EOS” = 0.84 * 0.70 = 0.5880
3. **以此类推**: 直到候选最大遇到了 `EOS`。

由于 beam search 会同时保留多个序列，因此**就更容易得到得分更高的序列，并且`beam_size`越大，获得更高得分的概率越高**。

由于每个 step 都需要进行 `beam_size` 次前向计算，计算量扩大 `beam_size` 倍。
另一方面，LLM 推理中一般都会使用 KV Cache，这也就会进一步增大 KV Cache 的内存占用，同时增加了 KV Cache 管理的复杂度。
这也就是在 LLM 推理中为什么比较少使用beam search。


## Top k

**不管是 greedy search，还是 beam search，对于固定输入，模型的输出是固定不变的**。
为了增加模型输出的多样性，人们提出了 `top-k 采样策略`，其**先选出分数最高的 k 个，然后将其分数作为权重进行随机采样，得到下一个 Token**，这也就引入了随机性。

还是以上面的例子来介绍，假设`k=3`：
- **Step1**：使用 Token `“it”` 对应的新 embedding 来计算 logits，选出得分最高的 3 个 Token：`“Okay”, “.”, “EOS”`，对应的权重为 `[0.91,0.84,0.72]`，使用该权重进行随机采样，获得新 Token“Okay”——可根据概率前缀和和。
- **以此类推**

如果**top_k==1，则对应 greedy search**。

在`top_k`中，每次都是从 k 个 Token 中采样，但是难免会出现一些特殊的 case，比如某一个 Token 的分数非常高，其他分数都很低，此时仍旧会有一定的概率采样到那些分数非常低的 Token，导致生成输出质量变差。


## Top p

**在每个 step 中，都对输出分数进行排序，然后将分数从大到小累加，直到累加分数大于设置的 p 为止，
然后和 `top_k` 类似， 将每个选择出来的 token 分数作为权重进行随机采样**。
这样，每次候选的 Token 个数都会因为 Token 分数的分布不同而不一样。


虽然从理论上讲，**`top_p` 似乎比 `top_k` 更优雅，但这两种方法在实践中都很好用。
`top_p` 也可以与 `top_k` 结合使用，这可以避免分数非常低的 Token，同时提供一些动态选择的空间。


## Temperature

事实上，**在 top_k 和 top_p 的采样中并不是完全按照分数权重来采样的**，
一般采样前我们会将候选Token的得分向量经过softmax（公式如下图）转换为概率，然后按照概率分布采样。

很多时候我们想要控制采样的随机性，可以使用带有温度系数 T 的 softmax 实现，
温度系数 T 为大于 0 的任意值（Huggingface中限制0.0<T<100.0）。
当**T=1**时，输出分布将与标准 softmax 输出相同。
T 的值越大，输出分布就越平滑，T 的值越小，输出分布越陡峭。

$$
\operatorname{softmax}\left(y_{i}\right)=\frac{e^{\frac{y_{i}}{T}}}{\sum_{j=1}^{n} e^{\frac{y_{j}}{T}}}
$$


## Repetition Penalty

思想比较简单，**记录之前已经生成过的 Token，当预测下一个 Token 时，人为降低已经生成过的 Token 的分数，使其被采样到的概率降低**。

直接基于上述带温度系数T的softmax进行实现，其中的`g`表示已经生成过的Token列表，如果某个Token已经在生成过的Token列表`g`中，则对其对应的温度系数T乘上一个系数`θ`，`θ`为大于0的任意值。

-   [ ] `θ=1`，表示不进行任何惩罚
-   [ ] `θ>1`，相当于尽量避免重复
-   [ ] `θ<1`，相当于希望出现重复

$$
p_{i}=\frac{\exp \left(x_{i} /(T \cdot I(i \in g))\right.}{\sum_{j} \exp \left(x_{j} /(T \cdot I(j \in g))\right.} \quad I(c)=\theta ~if~ c ~is ~True ~else ~1
$$


## 总结

- `GreedySearch`是最简单、最直接的方式，其可以保证稳定的输出，相应的，`BeamSearch`可以进一步提升生成效果，但是代价更高，也是可以保证稳定的输出。
- `top_p`和`top_k`都可以用于增加模型**生成结果的多样性**，输出结果往往会变。
- `temperature` 用于控制随机性，`temperature` 越大，随机性越强，`temperature` 越小，随机性越弱。
- 重复惩罚`repetition_penalty`用于避免模型一直输出重复的结果，`repetition_penalty`越大，出现重复性可能越小，repetition_penalty越小，出现重复性可能越大。

---

Reference:
- [LLM 推理常见参数](https://github.com/wdndev/llm_interview_note/blob/main/06.%E6%8E%A8%E7%90%86/LLM%E6%8E%A8%E7%90%86%E5%B8%B8%E8%A7%81%E5%8F%82%E6%95%B0/LLM%E6%8E%A8%E7%90%86%E5%B8%B8%E8%A7%81%E5%8F%82%E6%95%B0.md)
