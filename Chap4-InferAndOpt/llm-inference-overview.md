## 推理

LLM 推理有两个阶段：

- prefill: 输入 prompt 处理的阶段，会生成 cache
- decode/generation: 后续新生成 token 的阶段，会利用 prefill 的 cache 以及当前阶段本身产生的 cache

![Prefill and Decode flow](https://lh7-qw.googleusercontent.com/docsz/AD_4nXdQMgudBg_76tgcH5aNZVw_Mpx8gkSlUVcLvY642nKJQe_u-_huKTx7Fu2mJ-oa67CDjyOLjekLdRhpiWn4dB1Vit4_I0wruarwAfQnA5l1MBGzHJ1zXFDmlDATP8P__R6mZdXzJizlrMzvfGierM-bzBiR?key=YpIla8VBZeGwFtgrJhOaEQ)

**Prefill**

- Prefill 阶段会并行处理输入的所有 token，这里处理方式使得即使在较小的 bathc size 下也能打满 GPU 的利用率；
- 由于在 prefill 阶段需要处理长输入，所以这个阶段的计算开销很大，显卡利用率很容易打满；
- 增加 batch size 时，prefill 阶段每个 token 的处理开销几乎保持不变，这意味着 prefill 的效率在小 batch size 时就很高，说明开销是固定的。

**Decode 阶段**

- Decode 阶段是自回归的，每次只生成一个 token，因此这一阶段的 GPU 利用率较低；
- IO 密集型：Decode 过程中需要频繁地读取 KV Cache，导致 IO 开销较大。即使输入的长度始终为 1，反复的 KV Cache 访问也使得这一阶段成为 IO 密集型；
- 扩大 batch size 可以显著降低 decode 阶段的开销，因为更大的 batch size 能更有效地分摊固定的 IO 读写成本，不过开再大也不能完全打满GPU，毕竟KV Cache的读写开销过大，导致decode阶段难以成为计算密集型。


## 推理的相关概念

### 为什么大模型推理时显存涨得那么多

显存涨得很多且一直占着显存不释放的原因：
1. **模型参数占用显存**：
2. **输入数据占用显存**：
3. **中间计算结果占用显存**：
4. **内存管理策略**：某些框架在推理时采用了一种**延迟释放显存**的策略，即不会立即释放，而是保留一段时间以备后续使用。这种策略可以减少显存的分配和释放频率，提高推理效率。

### 大模型生成时的参数怎么设置

1. **Temperature**: 用于调整随机从生成模型中抽样的程度。使得相同的提示可能会产生不同的输出。温度为 0 将始终产生相同的输出，该参数设置越高随机性越大。
2. **Beam search**: 作为在给定可能选项的情况下**选择最佳输出的最终决策步骤**。Beam Search 宽度一个参数，用于确定算法在搜索的每个步骤中应该考虑的候选数量。
3. **Top p**: 动态设置 tokens 候选列表的大小。将可能性之和不超过特定值的 top tokens 列入候选名单。目的是限制可能被采样的低概率 token 的长度。
4. **Top k**: 允许其他高分 tokens 有机会被选择。这种采样引入的随机性有助于在很多情况下生成的质量。top k = 3 意味着选择前三个 token。

若 **Top k** 和 **Top p** 都启用，则 Top p 在 Top k 之后起作用。

### 有哪些省内存的大语言模型训练/微调/推理方法
一些常见的方法：
1. **参数共享**：通过共享模型中的参数，可以减少内存占用。例如，可以在不同的位置共享相同的嵌入层或注意力机制。
2. **梯度累积**：在训练过程中，将多个小批次的梯度累积起来，然后进行一次参数更新。这样可以减少每个小批次的内存需求，特别适用于GPU内存较小的情况。
3. **梯度裁剪**：通过限制梯度的大小，可以避免梯度爆炸的问题，从而减少内存使用。
4. **分布式训练**：将训练过程分布到多台机器或多个设备上，可以减少单个设备的内存占用。分布式训练还可以加速训练过程。
5. **量化**：将模型参数从高精度表示（如 FP32）转换为低精度表示（如 INT8 或 FP16），可以减少内存占用。量化方法可以通过减少参数位数或使用整数表示来实现。
6. **剪枝**：
7. **蒸馏**：
8. **分块处理**：

### 如何让大模型输出合规化
可以采取以下方法:
1. **数据清理和预处理**：
2. **引入合规性约束**：
3. **限制模型访问权限**：
4. **解释模型决策过程**：
5. **审查和验证模型**：
6. **监控和更新模型**：
7. **合规培训和教育**：

### 应用模式变更
1. **任务定制化**：通过对模型进行微调或迁移学习，使其适应特定的应用场景。例如，将大语言模型用于自动文本摘要、机器翻译、对话系统等任务。
2. **个性化交互**：通过对用户输入进行理解和生成相应的回复，实现更自然、智能的对话体验。这可以应用于智能助手、在线客服、社交媒体等场景。
3. **内容生成与创作**：将其应用于内容生成和创作领域。例如，自动生成新闻报道、创意文案、诗歌等内容，提供创作灵感和辅助创作过程。
4. **情感分析与情绪识别**：通过大语言模型对文本进行情感分析和情绪识别，帮助企业或个人了解用户的情感需求和反馈，以改善产品、服务和用户体验。
5. **知识图谱构建**：利用大语言模型的文本理解能力，将其应用于知识图谱的构建和更新。通过对海量文本进行分析和提取，生成结构化的知识表示，为知识图谱的建设提供支持。
6. **法律和合规应用**：大语言模型可以用于法律和合规领域，例如自动生成法律文件、合同条款、隐私政策等内容，辅助法律专业人士的工作。
7. **教育和培训应用**：将大语言模型应用于教育和培训领域，例如智能辅导系统、在线学习平台等，为学生提供个性化的学习辅助和教学资源。

---

References:

- [LLM Inference at scale with TGI](https://huggingface.co/blog/martinigoyanes/llm-inference-at-scale-with-tgi#:~:text=The%20process%20of%20LLM%20inference,role%20in%20the%20overall%20process.)
- [一起理解下LLM的推理流程](https://mp.weixin.qq.com/s/gCQGnmZIokPFT8xvna17lQ)
- [Understanding the LLM Inference Workload](https://www.youtube.com/watch?v=z2M8gKGYws4&list=PL_lsbAsL_o2B_znuvm-pDtV_cRhpqZb8l&index=23)
- [推理](https://github.com/wdndev/llm_interview_note/blob/main/06.%E6%8E%A8%E7%90%86/1.%E6%8E%A8%E7%90%86/1.%E6%8E%A8%E7%90%86.md)