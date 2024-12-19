# LoRA

## 背景

神经网络中的全连接层借助于矩阵乘法得以实现，然而，很多全连接层的权重矩阵都是满秩的。
当针对特定任务进行微调后，**模型中权重矩阵其实具有很低的本征秩**（intrinsic rank），
因此，论文的作者认为**权重更新的那部分参数矩阵尽管随机投影到较小的子空间，仍然可以有效的学习，可以理解为针对特定的下游任务这些权重矩阵就不要求满秩**。

LoRA 通过冻结原始模型权重并对一组单独的权重应用更新来修改微调过程，然后将其添加到原始参数中。

优点：
- 训练和任务适应的效率
  - LoRA 引入了低秩矩阵，仅修改原始模型权重的子集，与全套参数相比，这些矩阵很小，可以实现更高效的更新。
  - LoRA 的重点是针对最有影响力的参数来改变模型 Transformer 层中的权重矩阵。这种选择性更新简化了模型适应新的任务或数据集的过程。
- 减少计算资源需求
- 保留预训练模型权重
  - 传统微调中，模型所有权重都会发生变化，这可能导致模型最初拥有的一般知识丢失。LoRA 选择性更新权重的方法确保了预训练模型中嵌入的核心结构与知识在很大程度上得以维持。
  - 这种保留对于维持模型的广泛理解和功能至关重要，同时仍允许其适应特定任务或数据集。它确保微调后的模型保留原始模型的优势，例如对语言和上下文的理解，同时在目标领域获得新的功能或改进的性能。

## 技术原理

LoRA 的核心思想就是**通过低秩分解来模拟参数的改变量，从而以极小的参数量来实现大模型的间接训练。**
在涉及矩阵相乘的模块，在原始的 PLM 旁边增加一个新的通路，通过前后两个矩阵 `A, B` 相乘，第一个矩阵 `A` 负责降维，第二个矩阵 `B` 负责升维，
中间层维度为 `r`，从而来模拟所谓的本征秩。

以一个简化的示例说明，我们称矩阵 `WO (d x d)` 为 LLM 的参数集合，并将 `∆W (measure by d x d)` 作为将在微调中加入的权重调整的矩阵。
在 LoRA 方法中，完成训练后，对于大小为 `1 x d` 的新输入 `x`, 模型将 `x` 乘以 `W` 和 `∆W`，从而产生两个 `d` 大小的输出向量。
然后将这些向量按元素相加在一起以产生最终结果，表示为 `h`。

参数 `r` 对于确定 `A` 和 `B` 的大小至关重要。
较小的 `r` 值意味着更少的参数和更快的训练时间，尽管如果 `r` 设置得太低，可能会导致模型性能下降。

Transformer 的权重矩阵包括 Attention 模块里用于计算 `query`, `key`, `value` 的 `Wq`, `Wk`, `Wv` 以及多头 attention 的 `Wo`, 以及 MLP 层的权重矩阵。
原论文中 LoRA 只应用于 Attention 模块中的 4 种权重矩阵，而且通过消融实验发现同时调整 `Wq` 和 `Wv` 会产生最佳结果。


# QLoRA

## 背景

大部分的量化方法虽然可以减少 LLM 的内存占用，但此类技术仅适用于推理场景。
作者提出了 QLoRA，首次证明了**可以在不降低任何性能的情况下微调量化为 4bit 模型**。

## 技术原理

QLoRA（Quantized Low-Rank Adaptation）是 LoRA 的扩展，使该方法更加高效。
QLoRA 使用**一种新颖的高精度技术将预训练模型量化为 4bit，然后添加一小组可学习的低秩适配器权重，这些权重通过量化权重的反向传播梯度进行微调**。
QLoRA 引入的一些改进功能包括：
- **4-bit NormalFloat (NF4)**: 可以将其视为一种紧凑、优化的格式，用于记录模型的数据。它实现了一种平衡，非常适合遵循正态分布的权重，通过将数据精度缩小到4位来减少内存使用。
- **双量化 (Double quantization)**: 对第一次量化后的那些常量再进行一次量化，减少存储空间。
- **分页优化器 (Paged optimizers)**: 这些优化器有效地处理突发的内存需求，确保训练过程保持平稳和高效。使用 NVIDIA 统一内存特性，该特性可以在在 GPU 偶尔 OOM 的情况下，进行 CPU 和 GPU 之间自动分页到分页的传输，以实现无错误的 GPU 处理。该功能的工作方式类似于 CPU 内存和磁盘之间的常规内存分页。使用此功能为优化器状态（Optimizer）分配分页内存，然后在 GPU 内存不足时将其自动卸载到 CPU 内存，并在优化器更新步骤需要时将其加载回 GPU 内存。

作者在原论文中也指出了一些有趣的点，比如：指令调优虽然效果比较好，但只适用于指令相关的任务，在聊天机器人上效果并不佳，而聊天机器人更适合用Open Assistant数据集去进行微调。
通过指令类数据集的调优更像是提升大模型的推理能力，并不是为聊天而生的。

# 使用 LoRA 和 QLoRA 进行微调实用技巧

## Enable LoRA for Multiple Layers

将 LoRA 的应用扩展到模型的更多层可以进一步增强其适应性和性能。传统上，只有特定层可能会进行微调以降低复杂性，但跨多个层应用 LoRA 可以实现更细微的调整。

该方法可以通过在整个模型架构中进行细粒度调整，从而在专门任务中带来更显着的改进。然而，平衡调整的层数与可用计算资源至关重要，因为每个附加层都会增加所需的内存和处理能力。

## Implement Numerous Training Epochs

训练 epoch 的数量对完善模型的性能至关重要，尤其是在专业领域。
通过对数据集进行多次迭代，模型可以更有效地从数据中学习，从而提高其泛化能力。

在使用较小的数据集或对高度特定的任务进行微调时，这一点尤为重要，因为它允许模型逐步完善其理解能力。
应监控微调以避免过拟合，即模型在训练数据上表现良好，但在未见数据上表现不佳。


# LoRA 和 Adapter-tuning 的区别

- 插入位置。LoRA 是以残差连接的形式“并联”在 Transformer 的 `Q, K, V, O` 矩阵上，而 Adapter 是插入在 Transformer Block 的 Feed-forward layer 后；
- 推理延迟。LoRA 在训练完后其参数可以与原有预训练模型直接合并，变回单分支结构，不会引入额外延迟；而 Adapter 由于引入了额外的串联网络层，会带来额外的延迟；
- 参数存储。LoRA 微调，训练完毕后只需保存 LoRA 本身的参数；Adapter 需要保存整个原有模型的参数。

---

References:
- [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685)
- [LoRA-Interview_Note-CN](https://github.com/wdndev/llm_interview_note/blob/main/05.%E6%9C%89%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83/4.lora/4.lora.md)
- [Understanding LLM Fine Tuning with LoRA-run.ai](https://www.run.ai/guides/generative-ai/lora-fine-tuning)