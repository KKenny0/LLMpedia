基于 TRT-LLM 的 LLM 推理流程。

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

---

References:

- [LLM Inference at scale with TGI](https://huggingface.co/blog/martinigoyanes/llm-inference-at-scale-with-tgi#:~:text=The%20process%20of%20LLM%20inference,role%20in%20the%20overall%20process.)
- [一起理解下LLM的推理流程](https://mp.weixin.qq.com/s/gCQGnmZIokPFT8xvna17lQ)
- [Understanding the LLM Inference Workload](https://www.youtube.com/watch?v=z2M8gKGYws4&list=PL_lsbAsL_o2B_znuvm-pDtV_cRhpqZb8l&index=23)
