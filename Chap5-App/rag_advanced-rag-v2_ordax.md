# RAG 技术超全全景图：从基础到高级实践 

> [RAG技术超全全景图: 从基础到高级实践](https://x.com/shao__meng/status/1859389153441612149)
> [全景图](https://miro.com/app/board/uXjVNimscLw=/)


## RAG 基础

### 为什么要使用 RAG
LLM 生成的答案可能不准确：
- LLMs 存在幻觉问题
- 相关信息可能超过的 LLMs 的训练语料范围
- LLM 无法访问最新信息

### 基础 RAG 的工作流程
文档->向量存储->检索->响应。

### RAG 的主要使用场景
- **高级问答系统**：RAG 检索和生成准确回复，增强信息可及性。
- **内容创作和总结**：RAG 擅长基于特定的主题或 Prompts 生成一致的上下文。它从数据源中提取相关信息用以生成总结。
- **对话代理和 Chatbots**：RAG 模型增强了对话代理，使他们能够从外部数据源中获取上下文相关的信息。
- **信息抽取**：RAG 提升了搜索结果的相关性和准确度。此外，它将检索式方法与生成能力结合在一起。
- **教育工具和资源**：嵌入在教育工具中的 RAG 模型通过个性化体验彻底改变了学习方式。他们检索并生成量身定制的解释。
- **法条搜索和分析**：检索相关法条信息并辅助专业人士起草文档，分析案件和提出论点。
- **内容推荐系统**：基于检索信息进行个性化推荐生成，以提升用户体验

### 分块策略
- Fixed-size Chunking
- Recursive Chunking
  - Bigger Chunk
    - Smaller Chunk
- Document Based Chunking
  - by sentence
  - by Markdown syntax
- Semantic Chunking

### 高级检索技术
- 混合融合（Hybrid Fusion）: 词汇和语义搜索融合在一起，目的是使二者在相关性建模方面相辅相成。
  - Vector Search
  - Keyword Search
- 查询重写与融合: 由于原始查询可能不是最优的，我们先提示 LLM 对查询进行改写，然后进行检索-增强阅读。
  - Query 1
  - Query 2
  - Query 3
- 嵌入表格: 我们的目标是获得 query-table 对的相关性排名，这样排名靠前的表应该与查询更相关。
  - table 1
  - text 1
  - table 2
  - text 2
- 自动合并检索器: 文件被分割成不同层次的块，然后最小的块被发送到索引。这样做的目的是搜索更细粒度的信息，然后扩展上下文窗口，再将所述上下文输入 LLM 进行推理。
  - chunk 1
  - chunk 2
    - parent chunk 1
  - chunk 3
    - chunk 3
- 句子窗口检索器: 文档中的每个句子都是单独嵌入的，这使得查询与上下文之间的距离搜索非常准确。
  - sentence 1
    - chunk 1
  - sentence 2
    - chunk 2
  - sentence 3
    - chunk 3
- 节点引用检索器: 执行检索时，检索的是引用而不是原始文本。您可以让多个引用指向同一个节点。
  - Small Chunk 1
    - Chunk 1
  - Small Chunk 2
    - Chunk 2
  - Small Chunk 3
    - Chunk 3
- 多文档 Agents: 每个文档 agent 有两个工具——一个向量存储索引和一个摘要索引，并根据路由查询决定使用哪个工具。
  - Doc Agent 1
    - Vector Search or Summarization
  - Doc Agent 2
    - Vector Search or Summarization
- 元数据增强检索: 使用元数据进行检索，可通过过滤（如关键词）提高搜索效率。
  - metadata + Chunk 1
  - metadata + Chunk 2
  - metadata + Chunk 3

### 高级生成技术
- 信息压缩: 信息压缩减少了噪音，并帮助缓解上下文窗口限制
  - Retrieved Doc
    - Compressed Doc
- 生成微调: 微调，以帮助确保检索到的文档与 LLM 保持一致(e.g. self-rag)
- 结果重排序: 结果重排序缓解 LLMs 中的 lost-in-the-middle 现象
- 适配器方法: 这种方法附加外部适配器以使相关文档与 LLM 保持一致 (e.g., PRCA, RECOMP, PKG)

### 评估框架
- 关键能力评估
  - 噪声鲁棒性: 处理噪声或检索文档中所包含的不相关信息的能力
  - 负样本拒绝: 当缺乏足够知识（内部&外部）时拒绝回答的能力
  - 信息融合: 整合多个来源的信息来回答更复杂问题的能力
  - 反事实鲁棒性: 识别和处理检索文档中已知错误信息的能力
- 质量评分
  - 上下文相关性: 检索到的上下文需要与回答用户的问题相关
  - 回答相关性: 生成的回复需要直接解决用户的提问
  - 忠诚度: 生成的回复必须忠于检索到的上下文
- 评估框架
  - RGB
  - RECALL
  - RAGAS
  - ARES
  - TruLens
