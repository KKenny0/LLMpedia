## Encoder-Decoder 模型
编码器-解码器模型（也称为序列到序列模型）使用 Transformer 架构的两个部分。
在每个阶段，编码器的注意力层可以访问初始句子中的所有单词，而解码器的注意力层只能访问位于输入中给定单词之前的单词。

这些模型的预训练可以使用编码器或解码器模型的目标来完成，但通常会涉及一些更复杂的内容。
例如，T5 的预训练方法是用一个掩码特殊字符替换随机跨度的文本（可能包含多个单词），然后目标是预测这个掩码单词所替换的文本。

序列到序列模型最适合根据给定输入生成新句子的任务，例如摘要、翻译或生成式问答。
这个模型家族的代表包括:
- BART
- T5


## BART

> [BART](https://huggingface.co/docs/transformers/model_doc/bart)

Bart 模型是在 [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) 中提出。
论文摘要：
- Bart 使用标准的 seq2seq/机器翻译架构，带有双向编码器（如 BERT）和从左到右的解码器（如 GPT）。
- 预训练任务涉及随机打乱原始句子的顺序和新颖的填充方案，其中文本跨度被单个掩码标记替换。
- BART 在针对文本生成进行微调时特别有效，而且也适用于理解任务。

**使用技巧**

- BART 是一种具有绝对位置嵌入的模型，因此通常建议将输入填充到右侧而不是左侧。
- 具有编码器和解码器的序列到序列模型。编码器接收的是已损坏版本的 tokens，解码器接收的是原始 tokens（但有一个掩码来隐藏未来的词块，就像普通的变换 transformer 解码器一样）。在编码器的预训练任务中应用了以下变换组合：
  - Token 随机遮掩：与BERT一样，采用随机的 token 并将其替换为 `[MASK]`。
  - Token 随机删除：输入中的随机tokens被删除。与token masking不同的是，模型必须决定哪些位置是缺失的输入。
  - 文本填充：使用单个掩码标记遮掩 k 个范围的标记（0 范围对应 `[MASK]` token 的插入）。
  - 排列句子：以句号(full stop)为单位将文档划分为句子，这些句子进行随机排列。
  - 旋转文档：均匀随机选择一个 token，将文档旋转使得其以该 token 开始。这将训练模型识别文档开头的能力。


## T5

> [T5](https://huggingface.co/docs/transformers/model_doc/t5)

T5 模型在 [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf) 中提出。

论文摘要：
> 在本文中，我们通过引入一个统一的框架来探索 NLP 迁移学习技术的前景，该框架将每个语言问题转换为文本到文本的格式。
> 我们的系统研究比较了数十种语言理解任务的预训练目标、架构、未标记数据集、迁移方法和其他因素。
> 通过将我们探索中的见解与规模和新的“巨大的干净爬行语料库”相结合，我们在涵盖摘要、问答、文本分类等的许多基准上取得了最先进的结果。

**使用技巧**

- T5是一个编码器-解码器模型，预先对无监督和监督任务的多任务混合进行训练，并将每个任务转换为文本到文本的格式。T5通过为每个任务相应的输入添加不同的前缀，可以很好地处理各种开箱即用的任务，例如，对于翻译: *将英语翻译成德语: ...*，对于摘要: *总结: ...*。
- 预训练包括监督训练和自监督训练。
- 自监督培训练使用损坏的 tokens，通过随机删除 15% 的 tokens 并用单个哨兵令牌（sentinel tokens）替换它们(如果几个连续的令牌被标记为删除，则整个组被替换为单个哨兵令牌)。编码器的输入是被破坏的句子，解码器的输入是原始句子，目标是由哨兵标记分隔的丢弃标记。
  - Original text: Thank you ~~for inviting~~ me to your party ~~last~~ week.
  - Inputs: Thank you <X> me to your party <Y> week.
  - Targets: <X> for inviting <Y> last <Z>
- T5 使用相对标量嵌入。编码器输入填充可以在左侧和右侧完成。

