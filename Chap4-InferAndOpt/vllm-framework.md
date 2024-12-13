# vLLM

## Overview

vLLM 声称的特性：
- 优秀的 serving 吞吐量
- **PagedAttention** 对 KV Cache 的有效管理
- 传入请求的**continuous batching**, 而不是 static batching
- 高性能 CUDA kernel
- 流行的 HF 模型集成
- 各种 decoder 算法的高吞吐量服务，包括 parallel sampling 和 beam search 等
- tensor parallel
- 兼容 OpenAI 的 API 服务器

两个主要特性：Continuous batching 和 PagedAttention。


## Continuous Batching

### LLM Decoder 推理基础

LLM 推理分为两步：Prompt + Token Generation

LLM（大型语言模型）的推理速度主要受显存容量限制，而不是GPU的计算能力。
显存大小决定了能处理的批次大小（batch size）和句子长度（sequence length）。
例如，一个13亿参数的模型在 A100-40G GPU上，模型参数占了 26 GB显存，剩下 14 GB可以存储大约 14000 个 token 的状态（每个 token 的 state 花 1M 左右空间）。
如果句子长度设为512，最大批次大小为28；如果句子长度为2048，最大批次大小为7。这还没有考虑到中间计算过程中的内存占用。

量化技术（quantization）可以提高显存利用率，增加单卡上的批次大小和句子长度，但需要修改模型权重。
有些方法如 FlashAttention 和连续批处理（continuous batching）不需要修改权重，也能提高内存I/O效率。

### LLM batching

LLM（大型语言模型）在处理批量请求时存在一些挑战，主要是因为模型的推理过程是逐步进行的。
在批量处理中，有的请求可能很快就完成了，但是要将这些请求释放并加入新的请求到还在处理的 batch 中，这个过程比较复杂。
这导致GPU的利用率不高，因为不同的请求生成的序列长度不同，有的短有的长。
例如，如果一个请求生成了2个token，而另一个生成了5个，那么在短请求结束后，GPU就会空闲，直到长请求完成。
这种空闲时间就是GPU未被充分利用的表现，传统的静态批量处理方法无法有效利用这些空闲时间。

简而言之，LLM在批量处理时，由于请求处理时间不一，导致GPU利用率低，难以有效利用空闲时间。

### Continuous batching

简单来说，一旦一个batch中的某个seq完成生成，发射了一个end-of-seq token，就可以在其位置插入新的seq继续生成token，从而达到比static batching更高的GPU利用率。


## PagedAttention

PagedAttention 是对 KV Cache 所占空间的分页管理，是一个典型的**以内存空间换计算开销**的手段，
vLLM 和 TensorRT-LLM 都应用了这个手段来节约 KV Cache 占用的内存。

### KV Cache

LLM 的核心是自回归 Transformer 模型。
该模型可基于输入（prompt）和其之前输出的 token 序列生成词（token），一次生成一个。
对于每次请求，这个成本高昂的过程都会重复，直到模型输出终止 token。
这种按序列的生成过程会让工作负载受到内存限制，从而无法充分利用 GPU 的计算能力，并会限制服务的吞吐量。

通过批量方式同时处理多个请求可以提高吞吐量。
但是，要在单一批次中处理许多请求，就需要高效地管理每个请求所占用的内存空间。

有学者观察到当前的 LLM 服务系统都没有高效地馆里 KV 缓存。
主要原因是它们会将请求的 KV 缓存保存在邻接的内存空间中。

但是，不同于传统深度学习工作负载中的张量，KV 缓存有其自己的独特性质：
它会在模型生成新 token 的过程中随时间动态地增长和缩小，而且它的持续时间和长度是无法事先知晓的。

### vLLM 架构

vLLM 采用一种集中式调度器（scheduler）来协调分布式 GPU 工作器（worker）的执行。
**KV 缓存管理器由 PagedAttention 驱动，能以分页方式有效管理 KV 缓存。**
具体来说，KV 缓存管理器通过集中式调度器发送的指令来管理 GPU 工作器上的物理 KV 缓存内存。

### PagedAttention：解决内存瓶颈

在自回归解码过程中，所有输入到 LLM 的 token 会产生注意力键和值的张量，这些张量**保存**在 GPU 内存中以生成下一个 token。
这些缓存键和值的张量通常被称为 **KV 缓存**，其具有：
- **内存占用大**：在 LLaMA-13B 中，缓存单个序列最多需要 1.7 GB 内存；
- **动态且不可预测**：KV 缓存的大小取决于序列长度，这是高度可变和不可预测的。因此，这对有效地管理 KV 缓存挑战较大。该研究发现，由于碎片化和过度保留，现有系统浪费了 60% - 80% 的内存。

与传统的注意力算法不同，**PagedAttention 允许在非连续的内存空间中存储连续的键和值。**
具体来说，**PagedAttention 将每个序列的 KV 缓存划分为块，每个块包含固定数量 token 的键和值。**
在注意力计算期间，PagedAttention 内核可以有效地识别和获取这些块。

### KV 缓存管理器

对 KV 缓存的请求会被表示成一系列逻辑 KV 块，在生成新 token 和它们的 KV 缓存时从左向右填充。
最后一个 KV 块中未填充的位置留给未来填充。

在 PagedAttention 中，内存浪费只会发生在序列的最后一个块中。

PagedAttention 还有另一个关键优势 —— **高效的内存共享**。
例如在并行采样中，多个输出序列是由同一个提示（prompt）生成的。
在这种情况下，提示的计算和内存可以在输出序列中共享。
