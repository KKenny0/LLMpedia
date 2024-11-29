## Adapter based PEFT

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
每个 Adapter 模块主要


---

Refs:
- [3.adapter-tuning.md](https://github.com/wdndev/llm_interview_note/blob/main/05.%E6%9C%89%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83/3.adapter-tuning/3.adapter-tuning.md)
- [Summary Of Adapter Based Performance Efficient Fine Tuning (PEFT) Techniques For Large Language Models](https://siddharth-1729-65206.medium.com/summary-of-adapter-based-performance-efficient-fine-tuning-peft-techniques-for-large-language-fa65d0c2d55f)
- [Parameter Efficient Fine Tuning](https://medium.com/aimonks/parameter-efficient-fine-tuning-075954d1db51)
- [大模型参数高效微调技术原理综述（四）-Adapter Tuning及其变体](https://zhuanlan.zhihu.com/p/636038478)