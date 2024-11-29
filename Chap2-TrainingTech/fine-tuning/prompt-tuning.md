## Prompting

### Prefix Tuning

#### 背景
Prefix tuning 使预训练语言模型适应特定任务，而不修改原始模型的权重。
Prefix tuning 涉及将一系列连续的特定于任务的向量（称为前缀）添加到 LM 的输入中。 
Transformer 可以处理这些前缀向量，就好像它们是“虚拟令牌(virtual tokens)”序列一样。
与提示不同，前缀向量并不对应于真实的标记，而是在训练过程中学习的。

#### 技术细节

**软提示创建**
- 在 prefix tuning 中，我们为模型中的每个 transformer block 创建一个称为“软提示”的**张量**
- 这个软提示是一组可学习的参数，特定于我们想要调整模型的任务

**软提示处理**
- 在使用软提示之前，它会经过一组全连接层（**防止直接更新 prefix 的参数导致训练不稳定和性能下降**）
- 这些层将软提示转换为合适的表示形式，可以与 transformer block 的主输入相结合

**输入修改**
- 将转换后的软提示与 transformer block 的主输入连接起来
- 这种串联沿着序列长度维度发生，这意味着软提示作为附加标记添加到输入序列的开头

**Transformer Block 处理**
- 修改后的输入（现在包括软提示）通过标准 transformer block 操作
- 这些操作包括自注意力、层归一化、前馈神经网络层以及残差连接

**训练**
- 在训练期间，仅更新软提示，而预训练模型的权重保持冻结
- 该模型通过根据特定任务的训练数据调整软提示来学习适应特定任务
- 通过保持原始模型的权重不变，前缀微调可以实现高效的适应，而无需对整个模型进行微调

#### 具体例子

**定义前缀**
- 前缀是添加到输入序列之前的连续向量序列
- 前缀的长度是一个超参数，您可以根据目标个性的复杂性和可用的计算资源来选择。常见前缀长度范围为 10 到 50 个 token
- 前缀被初始化为大小为 $$(prefix_length, embed_length)$$ 的可训练矩阵 $$P$$
- 前缀矩阵 𝑃 的每一行对应一个前缀标记，该行中的值表示该标记的嵌入
- 前缀矩阵 𝑃 是随机初始化的或使用与目标个性相关的真实单词的激活来初始化。使用相关单词进行初始化可以为前缀提供良好的起点，并有可能加快训练过程中的收敛速度。

**修改模型架构**
- Transformer 层中的**自注意力机制**允许前缀标记被关注并影响输入标记的表示，从而有效地引导模型的行为
- 重要的是，在训练期间，仅更新前缀矩阵 𝑃，而预训练模型的参数（即 Transformer 层中的权重矩阵）保持冻结。这确保了前缀适应目标个性，同时保留预训练模型捕获的一般语言理解。

#### 优点
- 它允许预先训练的模型有效地适应新任务，而无需修改原始模型的权重
- 与微调整个模型相比，它需要更少的可训练参数，从而提高计算效率
- 它可以应用于任何基于 Transformer 的预训练模型，而不需要特定于任务的架构。

#### 实用优势
- 前缀微调允许对任务进行独立训练，从而实现可扩展的个性化，而不会造成数据交叉污染
- 每个用户的数据可以隔离，并且可以为每个用户训练个性化的前缀，保证隐私性和模块化。任务的独立性还可以实现跨用户的高效批处理，以及创建在同一任务上训练的多个前缀的集合


### Prompt Tuning
Prompt Tuning **通过反向传播更新参数来学习 prompts，而不是人工设计 prompts；同时冻结模型原始权重，只训练 prompts 参数**，训练完以后，用同一个模型可以做多任务推理。

Prompt Tuning 可以看作是 Prefix Tuning 的简化版本，它给**每个任务定义了自己的 prompt，然后拼接到数据上作为输入，但只在输入层加入 prompt tokens**，并且不需要加入 MLP 进行调整来解决难训练的问题。


### P-Tuning
P-Tuning 没有使用固定的离散提示，而是引入了可学习的连续提示嵌入。
这些嵌入就像在训练过程中学习的一组“虚拟”单词。

#### 工作原理
P-Tuning 将 **Prompt 转换为可学习的 Embedding 层，并用 MLP+LSTM 的方式来对 Prompt Embedding 进行一层处理**。

相比Prefix Tuning，P-Tuning 加入的可微的virtual token，**仅限于输入层，没有在每一层都加；另外，virtual token的位置也不一定是前缀，插入的位置是可选的**。
这里的出发点实际是把传统人工设计模板中的真实 token 替换成可微的virtual token。

经过预训练的LM的词嵌入已经变得高度离散，如果随机初始化virtual token，容易优化到局部最优值，而这些virtual token理论是应该有相关关联的。
因此，作者通过实验发现**用一个prompt encoder来编码会收敛更快，效果更好**。
即用一个 LSTM+MLP 去编码这些virtual token以后，再输入到模型。

---

Refs:
- [2.prompting.md](https://github.com/wdndev/llm_interview_note/blob/main/05.%E6%9C%89%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83/2.prompting/2.prompting.md)
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://training.continuumlabs.ai/training/the-fine-tuning-process/parameter-efficient-fine-tuning/prefix-tuning-optimizing-continuous-prompts-for-generation)

