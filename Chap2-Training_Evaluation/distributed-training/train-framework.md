[# DeepSpeed 和 Megatron 的区别和联系

## 总结

|          | DeepSpeed | DeepSpeed 代表性功能        | Megatron | Megatron 代表性功能     | 备注                                              |
|----------|-----------|------------------------|----------|--------------------|-------------------------------------------------|
| GPU 底层优化 | 有         | 开创性的全栈 GPU 内核设计 FP6 量化 | 更牛逼      | Fused CUDA Kernels | Megatron 是 NVIDIA 亲儿子，底层优化信手拈来                  |
| 数据并行     | 更牛逼       | Zero 系列的分布式数据并行方案      | 有        | 优化器分片              | Megatron 做了类似 ZERO-1 的优化器分片，但数据并行没有 DeepSpeed 强 |
| 模型并行     | 有         |                        | 更牛逼      |                    | Megatron 的张量并行很牛                                |


## Megatron-LM

1. Megatron-LM-1: 利用了张量并行和数据并行。
2. Megatron-LM-2: 新增了 pipeline 并行——`virtual pipeline: 1F1B-interleaving`，成为和 DeepSpeed 类似的 3D 并行的训练框架。
3. Megatron-LM-3: 增加了`Sequence Parallelism`、`Selective Activation Recomputation` 和 `Checkpointing Skipping`。
   - Sequence Parallelism: 在 Tensor Parallelism 的基础上，将 Transformer 核的 LayerNorm 以及 Dropout 层的输入按 Sequence Length 维度进行了切分，使得各个设备上面只需要做一部分的Dropout和LayerNorm。

## DeepSpeed

DeepSpeed 实现了三种并行方法（数据并行训练、模型并行训练和流水线并行训练）的灵活组合：零冗余优化起（Zero Redundancy Optimizer, 缩写为 ZeRO）是一种用于大规模分布式深度学习的新型内存优化技术。

ZeRO 作为 DeepSpeed 的一部分，用于提高显存效率和计算效率。
ZeRO 支持的数据并行、流水线并行和张量切片模型并行。
ZeRO 可以克服数据并行和模型并行的局限性，同事实现两者的优点。
通过在数据并行进程之间划分模型状态参数、梯度和优化器状态来消除数据并行进程中的内存冗余，而不是复制它们。

ZeRO 有三个主要的优化阶段，它们对应优化器状态、梯度和参数的划分：
1. Pos：减少 4 倍内存，通信量与数据并行性相同
2. Pos+g：减少 8 倍内存，通信量与数据并行性相同
3. Pos+g+p：内存减少与数据并行度 Nd 呈线性关系。

ZeRO-3 offload 是 ZeRO Stage 3 和 ZeRO offload 相结合的一种高效且易于使用的实施方式。
主要好处是：
1. 极高的内存效率，可以在有限的GPU资源上运行非常大的模型；
2. 极易使用:扩展到超过一万亿个参数，而不需要以复杂的方式组合多种并行技术；
3. 每个GPU的高性能吞吐量和跨GPU的超线性可扩展性，用于分布式训练。


## Megatron-DeepSpeed

176B BLOOM 使用 [Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed)进行训练，结合了两种技术：
- [DeepSpeed](https://github.com/microsoft/DeepSpeed): 深度学习优化库，让分布式训练变得简单、高效且有效;
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM): NVIDIA 的应用深度学习研究团队开发的大型、强大的 transformer 模型框架。

DeepSpeed 团队将 DeepSpeed 库中的 ZeRO 分片和流水线并行与 Megatron-LM 中的张量并行相结合，开发了一种基于 3D 并行的方案。

训练 BLOOM 时采用了两个框架的以下组件：

| 组件         | DeepSpeed | Megatron-LM |
|------------|-----------|-------------|
| ZeRO 数据并行  | 是         |             |
| 张量并行       |           | 是           |
| 流水线并行      | 是         |             |
| BF16 优化器   | 是         |             |
| CUDA 融合核函数 |           | 是           |
| DataLoader |           | 是           |

Megatron-LM 和 DeepSpeed 都有流水线并行和 BF16 优化器实现，但我们使用 DeepSpeed 的实现，因为它们集成了 ZeRO。

Megatron-DeepSpeed 实现了 3D 并行以允许大模型以非常有效的方式进行训练。
1. 数据并行(Data Parallelism, DP): 相同的设置和模型被复制多份，每份每次都被馈送不同的一份数据。处理是并行完成的，所有份在每个训练步结束时同步。
2. 张量并行(Tensor Parallelism, TP): 每个张量被分成多个块，因此张量的每个分片都位于其指定的 GPU 上，而不是让整个张量驻留在单个 GPU 上。在处理过程中，每个分片在不同的 GPU 上分别并行处理，结果在步骤结束时同步。这就是所谓的水平并行，因为是做的水平拆分。
3. 流水线并行(Pipeline Parallelism, PP): 模型在多个 GPU 上垂直（即按层）拆分，因此只有一个或多个模型层放置在单个 GPU 上。每个 GPU 并行处理流水线的不同阶段，并处理 batch 的一部分数据。
4. 零冗余优化器(Zero Redundancy Optimizer, ZeRO)：执行与 TP 类似的张量分片，但整个张量会及时重建以进行前向或反向计算，因此不需要修改模型。还支持各种卸载技术以补偿有限的 GPU 内存。

### 数据并行

在该方法中，模型被完全复制到每个 GPU，然后在每次迭代后所有模型相互同步各自的状态。
这种方法可以通过投入更多 GPU 资源的方式加快训练速度，解决问题。
但它有个限制，即只有当模型能够放进单个 GPU 时才有效。


### ZeRO 数据并行

ZeRO 数据并行就是通常的 DDP，只是没有每个 GPU 都复制完整的模型参数、梯度和优化器状态，而是每个 GPU 只存储其中的一部分。
在随后的运行过程中，当需要给定层的完整层参数时，所有 GPU 同步以相互提供它们缺失的部分。

该组件由 DeepSpeed 实现。


### 张量并行

在张量并行中，每个 GPU 仅处理张量的一部分，并且仅当某些算子需要完整的张量时才触发聚合操作。

Transformer 类模型的主要模块为：一个全连接层`nn.Linear`，后面跟一个非线性激活层`GeLU`。

基于权重矩阵按列拆分，随后的 GEMM 按行拆分方案，可以更新任意深度的 MLP，只需在每个`拆列-拆行`序列之后同步 GPU。

需要特别考虑的是: 由于前向和后向传播中每层都有两个 all reduce，因此 TP 需要设备间有非常快速的互联。
因此，除非你有一个非常快的网络，否则不建议跨多个节点进行 TP。

该组件由 Megatron-LM 实现。
Megatron-LM 最近扩展了张量并行能力，新增了序列并行的能力，用于难以使用前述切分算法的算子，如 LayerNorm。


### 流水线并行

朴素流水线并行是将模型各层分组分布在多个 GPU 上，并简单地将数据从 GPU 移动到 GPU。
该机制相对简单-将所需层用`.to()`方法绑到相应设备，现在只要数据进出这些层，这些层就会将数据切换到与该层相同的设备，其余部分保持不变。

这其实就是垂直模型并行。
例如，如果下图显示一个 8 层模型:
```text
===================  ===================
|  0 | 1 | 2 | 3  |  |  4 | 5 | 6 | 7  |
===================  ===================
        GPU0                 GPU1
```

当数据需要从第 3 层传到第 4 层时，它需要从 GPU0 传输到 GPU1，这会引入通信开销。
如果参与的 GPU 位于同一计算节点 (例如同一台物理机器) 上，则传输非常快，但如果 GPU 位于不同的计算节点 (例如多台机器) 上，通信开销可能会大得多。


**问题**
- 该方法为什么被称为 朴素 流水线并行呢，它又有什么缺陷呢？主要是因为**该方案在任意给定时刻除了一个 GPU 之外的其他所有 GPU 都是空闲的**。
- 共享嵌入可能需要在 GPU 之间来回复制。作者使用的流水线并行（PP）与上述朴素 PP 几乎相同，但它解决了 GPU 闲置问题，方法是将传入的 batch 分块为 micros batch 并人工创建流水线，从而允许不同的 GPU 同时参与计算过程。


### DP + PP

DeepSpeed 流水线 并行教程 中有一张图演示了如何将 DP 与 PP 结合起来，如下所示:
![DP+PP](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero-dp-pp.png)

DP rank 0 是看不见 GPU2 的， DP rank 1 是看不到 GPU3 的。对于 DP 而言，只有 GPU 0 和 1，并向它们馈送数据。GPU0 使用 PP “秘密地” 将它的一些负载卸载到 GPU2。同样地， GPU1 也会得到 GPU3 的帮助。

由于每个维度至少需要 2 个 GPU，因此这儿至少需要 4 个 GPU。


### DP+PP+TP

将 PP、TP 和 DP 相结合，称为 3D 并行，如下图所示：
![3D Parallelism](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-deepspeed-3d.png)

由于每个维度至少需要 2 个 GPU，因此在这里你至少需要 8 个 GPU 才能实现完整的 3D 并行。


### ZeRO DP+PP+TP

DeepSpeed 的主要功能之一是 ZeRO，它是 DP 的超级可伸缩增强版。
通常它是一个独立的功能，不需要 PP 或 TP。但它也可以与 PP、TP 结合使用。

- ZeRO-DP 与 PP（以及 TP）结合时，通常只启用 ZeRO 阶段 1，只对优化器状态进行分片。
- 由于 PP，层数已经比正常情况下少，因此不会节省很多内存。PP 已经将梯度大小减少了 `1/pp`，因此在此基础上的梯度分片和纯 DP 相比节省不了多少内存。


### BF16 Optimizer

用 FP16 训练大型 LLM 模型是一个禁忌。
BF16 格式的关键是它的指数位数与 FP32 相同，因此不会溢出，但 FP16 经常溢出！
FP16 的最大数值范围为 64k，您只能进行较小数的乘法。
例如你可以做 250*250=62500，但如果你尝试 255*255=65025，你就会溢出，这是导致训练出现问题的主要原因。
这意味着你的权重必须保持很小。
一种称为损失缩放 (loss scaling) 的技术有助于缓解这个问题，但是当模型变得非常大时，FP16 较小的数值范围仍然是一个问题。

无论使用 BF16 还是 FP16，都有一个权重副本始终在 FP32 中——这是由优化器更新的内容。
因此 16 位格式仅用于计算，优化器以全精度更新 FP32 权重，然后将它们转换为 16 位格式以用于下一次迭代。


---

References:
- [DeepSpeed与Megatron的区别和联系](https://blog.csdn.net/lianghuaju/article/details/138897906)
- [大模型-LLM分布式训练框架总结](http://www.uml.org.cn/ai/202311304.asp)
- [千亿参数开源大模型 BLOOM 背后的技术](https://huggingface.co/blog/zh/bloom-megatron-deepspeed)
