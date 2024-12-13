# 推理框架特点

**关键点**

1. **vLLM**: 适用于大批量 Prompt 输入，并对推理速度要求高的场景；
2. **Text generation inference**: 依赖 HuggingFace 模型，并且不需要为核心模型增加多个 adapter 的场景；
3. **CTranslate2**: 可在 CPU 上进行推理；
4. **OpenLLM**: 为核心模型添加 adapter 并使用 HuggingFace Agents，尤其是不完全依赖 PyTorch；
5. **Ray Serve**: 稳定的 Pipeline 和灵活的部署，它最适合更成熟的项目；
6. **MLC LLM**: 可在客户端（边缘计算）（例如，在 Android 或 iPhone 平台上）本地部署 LLM；
7. **DeepSpeed-MII**: 使用 DeepSpeed 库来部署 LLM。


## vLLM

1. 功能
    - **Continuous batching**: 有 iteration-level 的调度机制，每次迭代 batch 大小都有所变化，因此 vLLM 在大量查询下仍可以很好的 g 工作；
    - **PagedAttention**: 受操作系统中虚拟内存和分页的经典思想启发的注意力算法，这就是模型加速的秘诀。
2. 优点
    - **文本生成的速度**: vLLM 的推理速度是最快的；
    - **高吞吐量服务**: 支持各种解码算法，比如 parallel sampling, beam search 等；
    - **与 OpenAI API 兼容**: 如果使用 OpenAI API, 只需要替换端点的 URL 即可。
3. 缺点
    - **添加自定义模型**: 如果模型没有使用与 vLLM 中现有模型类似的架构，添加过程会变得更加复杂。例如，增加 Falcon 的支持；
    - **缺乏对适配器（LoRA、QLoRA等）的支持**: 没有单独使用模型和适配器权重的选项，限制了有效利用此类模型的灵活性。
    - **缺乏权重量化**: 有时，LLM可能不需要使用GPU内存，这对于减少GPU内存消耗至关重要。


## Text generation inference

Text generation inference是用于文本生成推断的Rust、Python和 gRPC 服务器，在 HuggingFace 中已有的 LLM 推理 API 使用。

1. 功能
   - **内置服务评估**：监控服务器负载并深入了解其性能；
   - **使用 flash attention (v2) 和 Paged attention 优化 transformer 推理代码**: 并非所有模型都内置了对这些优化的支持，该技术可以对未使用该技术的模型可以进行优化。
2. 优点
   - **支持 HuggingFace 模型**: 轻松运行自己的模型或使用任何 HuggingFace 模型中心；
   - **对模型推理的控制**: 该框架提供了一系列管理模型推理的选项，包括精度调整、量化、张量并行性、重复惩罚等。
3. 缺点
   - **缺乏对适配器的支持**: 没有官方支持或文档 显示说明对适配器的支持，尽管可以使用适配器部署 LLM；
   - **从源代码（Rust+CUDA内核）编译**: 对于不熟悉 Rust 的人，将客户化代码纳入库中变得很有挑战性；
   - **文档不完整**: 所有信息都可以在项目的自述文件中找到。尽管它涵盖了基础知识，但必须在问题或源代码中搜索更多细节。

---

Reference:
- [llm推理框架简单总结](https://github.com/wdndev/llm_interview_note/blob/main/06.%E6%8E%A8%E7%90%86/0.llm%E6%8E%A8%E7%90%86%E6%A1%86%E6%9E%B6%E7%AE%80%E5%8D%95%E6%80%BB%E7%BB%93/0.llm%E6%8E%A8%E7%90%86%E6%A1%86%E6%9E%B6%E7%AE%80%E5%8D%95%E6%80%BB%E7%BB%93.md)