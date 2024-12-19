# 评测

## 大模型怎么评测

**自动评测**和**人工评测**，这两种方法在评测语言模型和机器翻译等任务时起着重要作用。
- 自动评测：基于计算机算法和自动生成的指标，能够快速且高效地评测模型的性能。
- 人工评测：人类专家的主观判断和质量评测，能够提供更深入、细致的分析和意见。


## 大模型的 honest 原则是如何实现的？模型如何判断回答的知识是训练过的已知的知识？怎么训练这种能力？

大模型需要遵循的**helpful，honest，harmless**的原则。

可以有意构造如下的训练样本，以提升模型遵守 honest 原则的能力：微调时构造知识问答类训练集，给出不知道的不回答，加强 honest 原则；
阅读理解题，读过的要回答，没读过的不回答，不要胡说八道。


## 如何衡量大模型水平？

在评测 LLMs 性能时，选择合适的和领域对于展示大型语言模型的表现、优势和劣势至关重要。
为了更清晰地展示 LLMs 的能力水平，文章将现有的任务划分为以下 7 个不同的类别：
1. 自然语言处理：包括自然语言理解、推理、自然语言生成和多语言任务
2. 鲁棒性、伦理、偏见和真实性
3. 医学应用：包括医学问答、医学考试、医学教育和医学助手
4. 社会科学：
5. 自然科学与工程：包括数学、通用科学和工程
6. 代理应用：将 LLMs 作为代理使用
7. 其他应用


## 大模型评估方法有哪些？

1. **直接评估指标**：准确率和 F1 得分这类传统指标。通常情况下，**这种方法涉及从模型中获取单一的输出，并将其与参考值进行比较，可以通过约束条件或提取所需信息的方式来实现评估。**
2. **间接或分解的启发式方法**：这类方法中，**利用较小的模型来评估主模型生成的答案**，这些较小的模型可以是微调过的模型或原始的分解模型。
3. **基于模型的评估**：这种方法中，**模型本身提供最终的评估分数或评估结果**。这种方法也引入了额外的可变因素。即使模型可以获取到ground truth信息，评估指标本身也可能在评分过程中产生随机因素或不确定因素。


## 大模型评估工具有哪些

- **ChatbotArena**: 借鉴游戏排位赛机制，让人类对模型两两评价
- **SuperCLUE**: 中文通用大模型综合性评测基准，尝试全自动测评大模型
- **C-Eval**: 采用 1.4 万道涵盖 52 个学科的选择题，评估模型中文能力
- **FlagEval**：采用“能力—任务—指标”三维评测框架


---

References:
- [1.评测](https://github.com/wdndev/llm_interview_note/blob/main/09.%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0/1.%E8%AF%84%E6%B5%8B/1.%E8%AF%84%E6%B5%8B.md)