## 解码器模型

解码器模型仅使用 Transformer 模型的解码器。
在每个阶段，对于给定的单词，注意力层只能访问句子中位于该单词之前的单词。
这些模型通常称为自回归模型。

解码器模型的预训练通常围绕预测句子中的下一个单词进行。
这些模型最适合涉及文本生成的任务。
这个模型家族的代表包括:
- CTRL
- GPT
- GPT-2

## CTRL

CTRL 模型在 [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858) 中提出。
它是一个因果（单向）转换器，使用语言模型在大约 140 GB 文本数据的非常大的语料库上进行预训练，第一个标记保留为控制代码（例如链接、书籍、维基百科等）。

论文摘要：
> 这是一个条件 Transformer 模型，经过训练以控制样式、内容和特定任务行为的控制代码为条件。
> 控制代码源自与原始文本自然共存的结构，保留了无监督学习的优势，同时提供对文本生成的更明确的控制。
> 这些代码还允许 CTRL 预测训练数据的哪些部分最有可能给出序列。
> 这提供了一种通过基于模型的源归因来分析大量数据的潜在方法。

**使用技巧**

- CTRL 利用控制代码来生成文本：它需要从某些单词、句子或链接开始生成，以生成连贯的文本。
- CTRL 是一个具有绝对位置嵌入的模型，因此通常建议将输入填充到右侧而不是左侧。
- CTRL 经过因果语言建模 (CLM) 目标的训练，因此在预测序列中的下一个标记方面非常强大。利用此功能，CTRL 可以生成语法连贯的文本。


## OpenAI GPT

OpenAI GPT 模型在 [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) 中提出。
它是一个因果（单向）转换器，使用具有长范围依赖性的大型语料库（多伦多图书语料库）上的语言建模进行预训练。

论文摘要：
> 实验证明，通过在各种未标记文本的语料库上对语言模型进行生成式预训练，然后对每个特定任务进行区分性微调，可以实现这些任务的巨大收益。
> 与以前的方法相比，我们在微调过程中利用任务感知输入转换来实现有效的传输，同时需要对模型架构进行最小的更改。

**使用技巧**

- GPT 是一种具有绝对位置嵌入的模型，因此通常建议将输入填充到右侧而不是左侧。
- GPT 使用因果语言建模 (CLM) 目标进行训练，因此在预测序列中的下一个标记方面非常强大。利用此功能，GPT 可以生成语法连贯的文本。

OpenAI GPT-2 模型在 [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 中提出。
它是一个因果（单向）Transformer，使用语言建模在约 40 GB 文本数据的非常大的语料库上进行预训练。

论文摘要：
> GPT-2 是一个基于 Transformer 的大型语言模型，拥有 15 亿个参数，在包含 800 万个网页的数据集上进行训练。
> GPT-2 的训练目标很简单：根据某个文本中所有先前的单词来预测下一个单词。
> 数据集的多样性导致这个简单的目标包含跨不同领域的许多任务的自然发生的演示。
> GPT-2 是 GPT 的直接扩展，参数增加了 10 倍以上，训练数据量增加了 10 倍以上。
