# Summary

* [前言](README.md)

* [模型结构](Chap1-ModelArch/README.md)
  * Transformer 架构与模型
    * [Transformer 结构-Wiki](Chap1-ModelArch/transformer_arch/Infrastructure_transformer_wiki.md)  
    * [Transformer 可视化解释](Chap1-ModelArch/transformer_arch/Infrastructure_transformer_transformer-explainer.md)
    * [土猛的员外-Transformer 架构的整体指南](Chap1-ModelArch/transformer_arch/Infrastructure_luxiangdong_Transformer-OverallArch.md)
    * [Encoder 模型](Chap1-ModelArch/transformer_arch/Infrastructure_HF_Encoder-models.md)
    * [Decoder 模型](Chap1-ModelArch/transformer_arch/Infrastructure_HF_Decoder-models.md)
    * [Encoder-Decoder 模型](Chap1-ModelArch/transformer_arch/Infrastructure_HF_Encoder-Decoder-models.md)
  * 注意力机制
    * [Lilian-Attention?Attention!](Chap1-ModelArch/attention/Advanced_Blog_AttentionAttention.md)
    * [缓存优化与效果-KV](Chap1-ModelArch/attention/mha-related.md)

* [训练与评估](Chap2-Training_Evaluation/README.md)
  * [分布式训练](Chap2-Training_Evaluation/distributed-training/overview.md)
    * [分布式训练框架](Chap2-Training_Evaluation/distributed-training/train-framework.md) 
    * [数据并行](Chap2-Training_Evaluation/distributed-training/data-parallel.md)
    * [流水线并行](Chap2-Training_Evaluation/distributed-training/pipeline-parallel.md)
    * [张量并行](Chap2-Training_Evaluation/distributed-training/pipeline-parallel.md)
    * [MoE 并行](Chap2-Training_Evaluation/distributed-training/moe-parallel.md)
  * [微调](Chap2-Training_Evaluation/fine-tuning/overview.md)
    * [LLM 训练-ybq](Chap2-Training_Evaluation/fine-tuning/llm-training-ybq.md)  
    * [关于微调的讨论](Chap2-Training_Evaluation/fine-tuning/finetuning-discussion.md)
    * [预训练](Chap2-Training_Evaluation/fine-tuning/pretrain.md)
    * [Prompt-Tuning](Chap2-Training_Evaluation/fine-tuning/prompt-tuning.md)
    * [Adapter-Tuning](Chap2-Training_Evaluation/fine-tuning/adapter-tuning.md)
    * [LoRA](Chap2-Training_Evaluation/fine-tuning/lora.md)
    * [总结](Chap2-Training_Evaluation/fine-tuning/summary.md)
  * [大模型评估](Chap2-Training_Evaluation/evaluation/README.md)
    * [评测](Chap2-Training_Evaluation/evaluation/evaluating.md)
    * [幻觉](Chap2-Training_Evaluation/evaluation/hallucination.md)
  * 蒸馏和压缩
    * 知识蒸馏
    * 剪枝和量化

* [Prompt 工程](Chap3-PromptEngr/README.md)
  * Prompt 技术
    * [宝玉老师-如何写好提示词？](Chap3-PromptEngr/prompt-tech_baoyu_how-to-write-good-prompt.md)
  * 应用场景
    * [OpenAI-生成提示词的提示词](Chap3-PromptEngr/prompt-app_openai-prompt-generation.md)
    * [GPT 翻译 Prompt](Chap3-PromptEngr/prompt-app_baoyu_translator-gpt-prompt.md)

* [推理与优化](Chap4-InferAndOpt/README.md)
  * [LLM 推理过程](Chap4-InferAndOpt/llm-inference-overview.md)
  * [LLM 推理框架](Chap4-InferAndOpt/infer-framework.md)
  * [vLLM 介绍](Chap4-InferAndOpt/vllm-framework.md)
  * [LLM 推理参数](Chap4-InferAndOpt/llm-inference-param.md)
  * [OpenAI O1 相关进展](Chap4-InferAndOpt/openai-o1-rel.md)
  * 推理加速
  * 多模态处理
    * 图像-文本模型
    * 跨模态注意力机制
  * 内容与计算优化

* [应用方向](Chap5-App/README.md)
  * 文本生成与摘要
    * 自然语言生成
    * 文本摘要
  * 问答与对话技术
    * Chatbot 技术
    * 问答系统与检索增强生成
      * [大模型主流应用RAG的介绍——从架构到技术细节](Chap5-App/rag_intro_luxiangdong.md)
      * [Top 20+ RAG Interview Questions](Chap5-App/rag_interview-ques_analytics.md)
  * 代码生成与分析
    * 编程助手
    * 自动代码补全


