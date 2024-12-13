# 总结

## BitFit

对微调机制的一种积极探索，也很简单，通过仅调整bias效果就能有不错的效果，但没有具体阐述原理，就是通过猜测加实验得到的结果。
同时，作者提出一个观点：微调的过程不是让模型适应另外的数据分布，而是让模型更好的应用出本身的表征能力。

特点：
- 训练参数量极小（约0.1%）
- 在大部分任务上效果会差于LoRA、Adapter等方法


## Prefix Tuning

在每一个 Transformer 层都带上一些virtual token作为前缀，以适应不同的任务。

特点：
- 前缀 Token 会占用序列长度，有一定的额外计算开销。
- Prefix Tuning 的线性插值比较复杂。


## Prompt Tuning

该方法可以看作是 Prefix Tuning 的简化版本，针对不同任务，仅在输入层引入 virtual tokens 形式的软提示（soft prompt）。
特点：
- 相对于 prefix tuning，参与训练的参数量和改变的参数量更小，更节省显存。
- 对一些简单的 NLU 任务还不错，但对硬序列标记任务（即序列标注）表现欠佳。


## P-Tuning

将 prompt 转换为可以学习的 Embedding 层，并用 MLP+LSTM 的方式来对 Prompt Embedding进行一层处理。
相比Prefix Tuning，仅在输入层加入的可微的virtual token；另外，virtual token的位置也不一定是前缀，插入的位置是可选的。

特点：
- 引入一个prompt encoder（由一个双向的 LSTM +两层 MLP 组成）来建模virtual token的相互依赖会收敛更快，效果更好。


## P-Tuning v2

该方法在每一个 Transformer 层都加入了prompt token作为输入，引入多任务学习，针对不同任务采用不同的提示长度。
并且回归传统的分类标签范式，而不是映射器。

特点：
- 解决了Prompt Tuning无法在小模型上有效提升的问题
- 移除了对模型效果改进较小的重参数化的编码器（如：Prefix Tuning中的 MLP、P-Tuning 中的 LSTM）
- 对于一些复杂的硬序列标记任务（即序列标注）取得了不错的效果。


## Adapter Tuning

该方法设计了 Adapter 结构，并将其嵌入 Transformer 的结构里面，针对每一个 Transformer 层，增加了两个 Adapter 结构，在训练时，固定住原来预训练模型的参数不变，只对新增的 Adapter 结构和 Layer Norm 层进行微调。
特点：
- 通过在 Transformer 层中嵌入 Adapter 结构，在推理时会额外增加推理时长


## LoRA

该方法通过低秩分解来模拟参数的改变量，从而以极小的参数量来实现大模型的间接训练。

特点：
- 将 BA 加到 W 上可以消除推理延迟。
- 可以通过可插拔的形式切换到不同的任务。
- 设计的比较好，简单且效果好。


## QLoRA

使用一种新颖的高精度技术将预训练模型量化为 4 bit，然后添加一小组可学习的低秩适配器权重，这些权重通过量化权重的反向传播梯度进行微调。

特点：
- 使用 QLoRA 微调模型，可以显著降低对于显存的要求。同时，模型训练的速度会慢于 LoRA。


## 多种不同的高效微调方法对比

总的来说，像P-Tuning v2、LoRA等都是综合评估很不错的高效微调技术。
如果显存资源有限可以考虑 QLoRA；如果只是解决一些简单任务场景，可以考虑P-Tuning、Prompt Tuning也行。


## 总结

主要有如下几类参数高效微调方法：
- 增加额外参数，如：Prefix Tuning、Prompt Tuning、Adapter Tuning及其变体
- 选取一部分参数更新，如：BitFit
- 引入重参数化，如：LoRA、QLoRA
- 混合高效微调，如：MAM Adapter、UniPELT
