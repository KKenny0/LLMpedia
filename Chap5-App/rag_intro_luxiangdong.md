> 主要内容：
> - 大模型在实际应用中存在的问题
> - RAG 架构解析
> - RAG 技术架构的细节展示

## LLM 的问题

- **幻觉问题**：
- **新鲜度问题**：
- **数据安全**：


## RAG 架构解析
RAG 总概地可以理解为“**索引、检索和生成**”。
RAG 的主要组成，依次是：数据提取——向量化——创建索引——检索——自动排序——LLM 归纳生成。

## RAG 技术细节概览

- 数据索引
  - 数据清洗
    - Loader
    - 数据处理
    - 元数据
  - Chunking
    - 固定大小
    - 基于意图
      - 句分割
        - Naive
        - NLTK
        - spaCy
      - 递归分割
      - 特殊分割
    - 策略因素
      - 索引类型（文本类型和长度）
      - 模型类型（tokens 限制等）
      - 查询长度和复杂度
      - 应用类型（检索、问答、摘要...）
  - Embedding
    - BGE
    - m3e
    - ...
- 检索
  - 检索优化
    - 元数据过滤：比如有时间的元数据检索，就可以过滤掉非 5 月份的向量
    - 图数据库
    - 检索路由
      - 相似度
      - 关键词
      - SQL
    - 重排序：相似度和相关性不一致
    - 查询轮换
      - 修改 prompt，重新提交
      - HyDE：生成相似 prompt（更标准）
      - 子查询：多种分布式查询方式，比如：树、向量、顺序查询等
- 生成
  - 问答过程中，内部运转的驱动器
  - ReAct
  - Prompt 优化器
    - 行业知识
    - 模板
  - 框架
    - LangChain
    - LlamaIndex
    - 自研

### 数据索引

**数据清洗**
- 数据提取：包括数据 Loader，提取 PDF、WORD、MARKDOWN 等；
- 数据处理：包括数据格式处理，不可识别内容的剔除，格式化等；
- 元数据提取：提取文件名、时间、章节 title、图片 alt 等信息。

**分块**
- 固定大小的分块：取决于 embedding 模型的情况。但是会损失很多语义，这样对检索是非常不友好的，解决方法是增加 overlap，一头一尾去保存相邻的 chunk 头尾的 tokens 内容；
- 基于意图的分块方式：
  - 句分割：通过句号和换行来切分。NLTK 和 spaCy 也提供了基于意图的分割模型；
  - 递归分割：通过分治的方法，用递归切分到最小单元到一种方式；
  - 特殊分割：
- 影响分块策略的因素：
  - 取决于类型类型，包括文本类型和长度，文章和微博推文的分块方式就会不同；
  - 取决于模型类型，使用的 LLM、Embedding 不同，限制长度不一样，考虑的分块尺寸也会受影响；
  - 取决于问答的文本长度和复杂度：
  - 应用类型：RAG 的应用是检索、问答和摘要等，对分块策略都要不同的影响。

### 检索环节
- **元数据过滤**：比如，用户问“帮我整理一下 XX 部门今年 5 月份的所有合同中，包含 XX 设备采购的合同有哪些？”如果有元数据，就可以去搜索“**XX 部门+2023年 5 月**”的相关数据。


---

References:
- [大模型主流应用RAG的介绍——从架构到技术细节](https://luxiangdong.com/2023/09/25/ragone/)