# Adapter based PEFT

## Adapter Tuning

### 背景

作为 Prompt 和 Prefix 微调技术的替代方案。
作者建议使用适配器模块（Adapter module）进行迁移学习。
Adapter 是在预训练网络层之间添加的新模块，在训练过程中仅训练新参数，原始 LLM 被冻结，因此只学习原始 LLM 的一小部分参数。
这意味着模型对之前的任务几乎具有完美的记忆，并使用少量的新参数来学习新任务。

### 技术原理
针对每个 transformer sub-layers (Attention 和 Feed Forward Layers)，在 sub-layers 之后插入两个串行的适配器模块。
适配器始终直接应用于子层的输出，分别是多头注意力的投影之后和第二个 feed-forward 层之后。
在训练时，固定住原来预训练模型的参数不变，只对新增的 Adapter 结构和 Layer Norm 层进行微调，从而保证了训练的高效性。

### 具体细节
每个 Adapter 模块主要由**两个前馈（Feedforward）子层组成**，第一个前馈子层（Feedforward down-project）将 Transformer 块的输出作为输入，
将原始输入维度 `d`（高维特征）投影到 `m`（低维特征），通过控制 `m` 的大小来限制 Adapter 模块的参数量，通常情况下，`m<<d`。

然后，中间通过一个非线性层。在输出阶段，通过第二个前馈子层（Feedforward up-project）还原输入维度，将 `m`（低维特征）重新映射回 `d`（原来的高维特征），作为 Adapter 模块的输出。
同时，通过一个skip connection来将 Adapter 的输入重新加到最终的输出中去，这样可以保证，即便 Adapter 一开始的参数初始化接近0，
Adapter 也由于skip connection的设置而接近于一个恒等映射，从而确保训练的有效性。
$$
h \leftarrow h+f\left(h W_{\text {down }}\right) W_{u p}
$$

### How to decide the value of m

- Adapter 模块中 `m` 的大小决定了待优化的参数量，因此需要对参数和性能进行权衡。
- 原论文的实验研究发现，不同的 Adapter 大小 `m` 的性能十分稳定，因此对于给定的模型，任何下游任务都可以使用固定的大小 `m`。= 

---

Refs:
- [3.adapter-tuning.md](https://github.com/wdndev/llm_interview_note/blob/main/05.%E6%9C%89%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83/3.adapter-tuning/3.adapter-tuning.md)
- [Summary Of Adapter Based Performance Efficient Fine Tuning (PEFT) Techniques For Large Language Models](https://siddharth-1729-65206.medium.com/summary-of-adapter-based-performance-efficient-fine-tuning-peft-techniques-for-large-language-fa65d0c2d55f)
- [Parameter Efficient Fine Tuning](https://medium.com/aimonks/parameter-efficient-fine-tuning-075954d1db51)
- [大模型参数高效微调技术原理综述（四）-Adapter Tuning及其变体](https://zhuanlan.zhihu.com/p/636038478)