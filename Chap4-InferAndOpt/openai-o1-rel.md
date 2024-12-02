# OpenAI o1 相关进展

## OpenAI o1 Learning to Reason with LLMs

**How reasoning works**

The o1 models introduce **reasoning tokens**. 
The models use these reasoning tokens to "think", breaking down their understanding of the prompt and considering multiple approaches to generating a response.
After generating reasoning tokens, the model produces an answer as visible completion tokens, and discard the reasoning tokens from its context.

> Reasoning tokens are not visible via the API, they still occupy space in the model's context window. 

- [Learning to reason with llms](https://openai.com/index/learning-to-reason-with-llms/)
- [Evaluation of OpenAI o1: Opportunities and Challenges of AGI](https://arxiv.org/abs/2409.18486)


## Skywork-o1

昆仑万维发布了一系列融合了类似 o1 的慢思考和推理能力的模型：
- **Skywork o1 Open-Llama-3.1-8B**: 通过“o1-style”数据显著增强模型，以提高推理技能
- **Skywork o1 Open-PRM-Qwen-2.5-1.5B**: 旨在通过增量过程奖励来增强推理能力，非常适合小规模解决复杂问题
- **Skywork o1 Open-PRM-Qwen-2.5-7B**: 扩展 1.5B 模型的功能，以处理更苛刻的推理任务。

### 方法

三阶段训练计划：
- **反思推理训练**：利用专有的多智能体系统为长期思考任务生成高质量、多样化的数据，然后进行持续的预训练和监督微调。
- **推理能力的强化学习**：引入 Skywork o1 过程奖励模型（PRM），专为增强逐步推理而定制。我们的实验证实，Skywork-PRM 结合专有的推理强化算法，有效捕获了中间推理步骤对最终结果的影响。
- **推理规划**：天工自助研发的 $$Q^{*}** 在线推理算法，结合模型思维，寻找最优推理路径。

- [Skywork/Skywork-o1-Open-Llama-3.1-8B](https://huggingface.co/Skywork/Skywork-o1-Open-Llama-3.1-8B)


## LLaVa-CoT: Let Vision Language Models Reason Step-by-Step

**LLaVA-CoT模型的介绍**
LLaVA-CoT 是一种新型的视觉语言模型，旨在通过自主多阶段推理来提高对复杂问题的解决能力。
与依赖链式思考提示的模型不同，LLaVA-CoT 能够独立地进行多阶段的推理过程，这使得它在处理需要深入推理的视觉问答任务时表现出色。

**视觉语言模型（VLMs）在系统和结构化推理方面的挑战**
尽管大型语言模型在推理能力上取得了显著进步，但现有的视觉语言模型在执行系统和结构化推理时常常遇到困难。
这些模型往往缺乏一个清晰的推理过程，导致在逻辑推理任务中效果不佳。
LLaVA-CoT 的设计正是为了解决这一问题，通过引入结构化的推理步骤来提高模型的性能。

**LLaVA-CoT模型的设计和结构**
LLaVA-CoT 模型将答案生成过程拆解为四个阶段：摘要、视觉解释、逻辑推理和结论生成。
每个阶段都有其独特的目的，并使用专门的标签来标记，以便模型能够清晰地识别自己所处的推理阶段，并理解每个阶段的主要任务。
- **Summary Stage**: 初始阶段，模型提供了对该问题的高度概括性解释，概述了它打算解决的问题的主要方面。
- **Caption Stage**: 如果有图像，模型会提供与问题相关的视觉元素的简明概述，帮助理解多模态输入。
- **Reasoning Stage**: 在初步总结的基础上，模型进行有条理的逻辑推理，得出初步答案。
- **Conclusion Stage**: 在最后阶段，模型会根据前面的推理综合出一个答案。 在这里，结论阶段的输出是提供给用户的直接答复，而前面三个阶段则是内部 "隐藏阶段"，代表了 LLaVA-CoT 的推理过程。该阶段的输出会根据用户的要求进行调整：例如，如果用户要求简短的回答，结论就会简明扼要；如果需要详细的解释，结论就会提供详尽全面的回答。

每个阶段都由模型自行决定启动，无需外部提示工程框架或额外提示。
具体来说，作者为模型提供了四对特殊标记：<SUMMARY></SUMMARY>、<CAPTION></CAPTION>、<REASONING></REASONING> 和 <CONCLUSION></CONCLUSION>。
这些标记分别对应于总结回答方法、描述相关图像内容、进行推理和准备最终答案

**LLaVA-CoT-100k数据集的构建**
为了训练 LLaVA-CoT 模型，研究者们构建了一个名为 LLaVA-CoT-100k 的数据集，该数据集整合了多个视觉问答数据源，并提供了结构化推理注释。
这为模型的训练提供了丰富的、带有详细推理过程的样本。

- [LLaVA-CoT: Let Vision Language Models Reason Step-by-Step](https://arxiv.org/pdf/2411.10440)


## Marco-o1

**Marco-o1模型的介绍**
Marco-o1 是由阿里巴巴国际数字商务团队开发的一个大型推理模型，旨在解决开放式问题，这些问题通常缺乏明确的标准和难以量化的奖励。
该模型不仅关注于数学、物理和编程等有标准答案的学科，还特别强调开放式解决方案的重要性。
Marco-o1 模型通过结合 Chain-of-Thought（CoT）微调、Monte Carlo Tree Search（MCTS）和创新的推理策略，优化复杂现实世界问题的解决能力。

**Marco-o1模型的目标和挑战**
Marco-o1 模型的目标是探索大型推理模型的技术路线图，并着重于开放式问题和多语言应用。
尽管当前模型主要展现出类似于 o1 的推理特征，但其性能尚未达到完全实现的"o1"模型水平。
这项研究工作是一个持续优化和改进的过程，团队致力于不断提升模型的性能。

**Marco-o1模型的技术特点**
Marco-o1 模型的技术特点包括采用 CoT 微调和 MCTS，以及推理行动策略。
CoT 微调通过结合开源 CoT 数据集和合成数据，提升模型处理复杂任务的能力。
MCTS 允许模型探索多个推理路径，并使用 softmax 应用的对数概率来指导模型找到最优解。
此外，推理行动策略涉及在步骤和迷你步骤中变化行动的粒度，以优化搜索效率和准确性。

**Marco-o1模型的实验设置和主要结果**
实验基于 Qwen2-7B-Instruct 模型，通过使用训练数据进行监督式微调来创建 Marco-o1-CoT。
此外，还在 MCTS 框架内使用 Marco-o1-CoT，通过不同的行动粒度（步骤和迷你步骤）进行区分。
在测试中，所有模型都使用 CoT 提示以确保推理过程的一致性，并在 MGSM 数据集的英文和中文子集上进行测试。

**Marco-o1模型的结论和未来工作**
Marco-o1 模型通过整合 CoT 微调、MCTS 和新颖的推理行动策略，增强了模型的推理能力。
MCTS 的整合允许模型扩展解决方案空间，并通过不同粒度的行动（步骤和迷你步骤）进行实验，显示出更细的搜索分辨率在提高准确性方面的潜力。
未来，团队计划通过结果奖励建模（ORM）和过程奖励建模（PRM）来优化 MCTS 的奖励信号，并探索强化学习技术来微调 Marco-o1 的决策过程，以进一步提升其解决复杂现实世界任务的能力。

- [Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions](https://arxiv.org/abs/2411.14405)


## QwQ

- [qwq-32b-preview](https://qwenlm.github.io/blog/qwq-32b-preview/)



## DeepSeek-R1-Lite-Preview

- [deepseek-r1-lite-preview](https://api-docs.deepseek.com/news/news1120)

---

- [OpenAI o1 的后续工作](https://mp.weixin.qq.com/s/fVWe8wlwhAN6Cw-2p2FH6w)
- [OpenAI o1 的后续工作](https://mp.weixin.qq.com/s/jP00sy_wicIJ1wPSafskIA)
