## 编码器模型

> [Encoder models](https://huggingface.co/learn/nlp-course/en/chapter1/5)
 

编码器模型仅使用 Transformer 模型的编码器。
在每个阶段，**注意力层都可以访问初始句子中的所有单词**。
这些模型通常被描述为具有“双向”注意力，并且通常被称为**自编码模型**。

这些模型的预训练通常围绕着*以某种方式破坏给定的句子*（例如，通过屏蔽其中的随机单词）并要求模型查找或重建初始句子。

编码器模型**最适合需要理解完整句子的任务**，例如句子分类、命名实体识别（以及更一般的单词分类）和提取式问答。

这个模型家族的代表包括:
- BERT
- ALBERT
- DistillBERT
- ELECTRA
- RoBERTa

## BERT

> [BERT](https://huggingface.co/docs/transformers/model_doc/bert)

BERT 模型在 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) 中提出。
它是一个双向 Transformer，使用掩码语言建模（masked language modeling）目标和下一句预测（next sentence prediction）相结合的方式对大型语料库进行预训练。

**使用技巧**

- BERT 是一种具有绝对位置嵌入的模型，因此通常建议将输入填充到右侧而不是左侧。
- BERT 使用掩码语言模型 (MLM) 和下一句预测 (NSP) 目标进行训练。一般来说，它在预测屏蔽标记和 NLU 方面非常有效，但对于文本生成来说并不是最佳选择。
- 通过使用随机屏蔽（random masking）来破坏输入，更准确地说，在预训练期间，给定百分比的标记（通常为 15%）被以下方式屏蔽：
  - 标记有 0.8 的概率打上特殊掩码 token
  - 标记有 0.1 的概率打上随机 token（与原始标记不同）
  - 标记有 0.1 的概率使用原始 token
- 该模型必须预测原始句子，但还有第二个目标：输入是两个句子 A 和 B（中间有一个分隔标记）。句子在语料库中连续的概率为 50%，其余 50% 的句子不相关。该模型必须预测句子是否连续。

## ALBERT

> [ALBERT](https://huggingface.co/docs/transformers/model_doc/albert)

ALBERT 模型在 [《ALBERT：A Lite BERT for Self-supervised Learning of Language Representations》](https://arxiv.org/abs/1909.11942) 中提出。
它提出了两种参数减少技术来降低内存消耗并提高 BERT 的训练速度：
- 将嵌入矩阵拆分为两个较小的矩阵
- 使用重复层（repeating layers）在组之间进行分割

ALBERT 还使用了一种自我监督损失，重点是对句子间的连贯性进行建模，实验表明这始终有助于具有多句子输入的下游任务。

**使用技巧**

- ALBERT 是一个具有绝对位置嵌入的模型，因此通常建议将输入填充在右侧而不是左侧。
- ALBERT 使用重复层，这会导致内存占用较小，但计算成本仍然类似于具有相同数量隐藏层的类 BERT 架构，因为它必须迭代相同数量的（重复）层。
- 层被划分为共享参数的组(以节省内存)。下一个句子预测被一个句子顺序预测所取代: 在输入中，我们有两个句子 A 和 B (是连续的) ，我们要么输入 A 后面跟着 B，要么输入 B 后面跟着 A。模型必须预测它们是否被交换了。


## DistilBERT

> [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)

DistilBERT 在 [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) 中提出。
DistilBERT 是一个通过蒸馏 BERT 基础训练而成的小型、快速、廉价且轻量的 Transformer 模型。

在这项工作中，作者提出了一种预训练较小的通用语言表示模型的方法，称为 DistilBERT，然后可以对其进行微调，使其在广泛的任务（如其较大的对应任务）上具有良好的性能。
虽然大多数先前的工作研究了如何使用蒸馏来构建特定于任务的模型，但作者在预训练阶段利用了**知识蒸馏**，并表明可以将 BERT 模型的大小减少 40%，同时保留 97% 的语言理解能力，速度提高 60%。
为了利用预训练期间较大模型学到的归纳偏差，作者引入了结合语言建模、蒸馏和余弦距离损失的三重损失。

**使用技巧 (HF)**

- DistilBERT 没有 `token_type_ids`，不需要指示哪个 token 属于哪个段。只需使用分隔标记 `tokenizer.sep_token` （或 `[SEP]`）分隔片段。
- DistilBERT 没有选择输入位置（`position_ids` 输入）的选项。
- 与 BERT 相同但更小。通过预训练 BERT 模型的蒸馏进行训练，这意味着它经过训练可以预测与较大模型相同的概率。实际目标是以下各项的组合：
  - 找到与教师模型（Teacher Model）相同的概率
  - 正确预测屏蔽标记（但没有下一句目标）
  - 学生模型和教师模型的隐藏状态之间的余弦相似度


## ELECTRA

> [ELECTRA](https://huggingface.co/docs/transformers/model_doc/electra)

ELECTRA 模型是在 [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/pdf?id=r1xMH1BtvB) 中提出的。 
ELECTRA 是一种新的预训练方法，它训练两个 Transformer 模型：生成器（generator）和判别器（discriminator）。
生成器的作用是替换序列中的标记，因此被训练为掩码语言模型。
判别器是我们感兴趣的模型，它尝试识别序列中哪些标记被生成器替换。

论文摘要：
> 掩码语言建模 (MLM) 预训练方法（例如 BERT）通过用 `[MASK]` 替换一些标记来破坏输入，然后训练模型来重建原始标记。
> 虽然它们在转移到下游 NLP 任务时会产生良好的结果，但它们通常需要大量计算才能有效。
> 作为替代方案，我们提出了一种样本效率更高的预训练任务，称为替换令牌检测。
> 我们的方法不是屏蔽输入，而是通过用从小型生成器网络采样的合理替代方案替换一些令牌来破坏输入。
> 然后，我们不是训练一个预测损坏令牌原始身份的模型，而是训练一个判别模型来预测损坏输入中的每个令牌是否被生成器样本替换。
> 彻底的实验证明这个新的预训练任务比 MLM 更有效，因为该任务是在所有输入标记上定义的，而不仅仅是被屏蔽的小子集。

**使用技巧**

- ELECTRA 是预训练方法，因此对底层模型 BERT 几乎没有任何改变。唯一的变化是嵌入大小和隐藏大小的分离：嵌入大小通常较小，而隐藏大小较大。附加的投影层（线性）用于将嵌入从嵌入大小投影到隐藏大小。在嵌入大小与隐藏大小相同的情况下，不使用投影层。
- ELECTRA 是一个 Transformer 模型，使用另一个（小型）掩码语言模型进行预训练。输入被该语言模型破坏，它接受随机屏蔽的输入文本，并输出一个文本，ELECTRA 必须预测其中哪个标记是原始标记，哪个标记被替换。与 GAN 训练一样，先对小语言模型进行几步训练（但目标是原始文本，而不是像传统的 GAN 设置那样愚弄 ELECTRA 模型），然后再对 ELECTRA 模型进行几步训练。


## RoBERTa

> [RoBERTa(https://huggingface.co/docs/transformers/model_doc/roberta)

RoBERTa 模型在 [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) 中提出。
它建立在 BERT 的基础上，修改了关键的超参数，删除了下一句话预训练目标，并使用更大的小批量和学习率进行训练。

论文摘要：
> 我们提出了 BERT 预训练的复制研究（Devlin 等人，2019），该研究仔细测量了许多关键超参数和训练数据大小的影响。
> 我们发现 BERT 的训练明显不足，但可以匹配或超过其之后发布的每个模型的性能。
> 实验结果凸显了以前被忽视的设计选择的重要性，

**使用技巧**

- 此实现与 BertModel 相同，只是对嵌入进行了细微调整，以及 RoBERTa 预训练模型的设置。
- RoBERTa 具有与 BERT 相同的架构，但使用字节级 BPE （byte-level BPE）作为 tokenizer（与 GPT-2 相同）并使用不同的预训练方案。
- RoBERTa 与 BERT 类似，但具有更好的预训练技术：
  - 动态屏蔽：令牌在每个时期的屏蔽方式不同，而 BERT 则一劳永逸。
  - 句子打包（Sentence packing）：句子打包在一起达到 512 个标记（因此句子的顺序可能跨越多个文档）。
  - 较大批次：训练使用较大批次。
  - 字节级 BPE 词汇：使用 BPE，将字节作为子单元，而不是字符，以适应 Unicode 字符。
