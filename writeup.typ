#set page(numbering: "1")
#set heading(numbering: "1.")
#set text(font: "Noto Serif CJK SC")
#show heading: set text(font: "Noto Serif CJK SC")

= CS336 Assignment 1 Writeup
作者：蔡志威
日期：2026-01-01

== Problem 1：Understanding Unicode（1 分）
=== （a）
它返回 Unicode 空字符 NULL（U+0000）。

=== （b）
其 `__repr__()` 显示为转义序列 `\x00`，而打印时只是输出一个不可见的控制字符，不显示任何可见字符。

=== （c）
当它出现在文本中时，它作为不可见的嵌入 NUL 字符存在，不会终止或破坏 Python 字符串；连接与打印均成功，但中间没有可见输出。

== Problem 2：Unicode Encodings（3 分）
=== （a）
UTF-8 保留 ASCII 为单字节、对常见语料更紧凑并使字节级 BPE 边界更合理；同时避免 UTF-16/UTF-32 的大小端/BOM 与代理项复杂性，便于流式处理与跨平台一致性。

=== （b）
示例：`b"\xc3\xa9"`（'é' 的 UTF-8 多字节序列）；该函数逐字节解码会把多字节序列拆开，导致 `UnicodeDecodeError` 或错误字符，因此它按字节而非按 UTF-8 序列解码是错误的。

=== （c）
示例：`b"\x80\x80"`；这两个均为续字节且缺少合法的领头字节，属于非法 UTF-8 序列，不能解码为任何 Unicode 字符。

== Problem 3：train_bpe_tinystories（2 分）
=== （a）
在 TinyStories 上训练了 10k 字节级 BPE（包含 `<|endoftext|>`），耗时约 0.14 小时，峰值内存约 11.1 GB；最长 token 为 " accomplishment"（15 字节），为空格前缀的英文长词，合理。

=== （b）
剖析显示并行预分词阶段耗时最多（`_pretokenize_parallel` 累计约 457s），其余时间主要花在进程结果收集与等待上。
== Problem 4：train_bpe_expts_owt（2 分）
=== （a）
在 OpenWebText 上训练了 32k 字节级 BPE 并序列化词表与合并；最长 token 为 64 字节，预览为“ÃÂ”重复 16 次，源自网页语料中的编码 mojibake（UTF‑8 与 Latin‑1 混解），BPE 合并这种高频重复合理但语义上并不合理。

=== （b）
41→TinyStories 的 tokenizer 更干净，最长 token 为“ accomplishment”（15 字节，空格前缀的英文词片），而 OpenWebText 的 tokenizer 合并出“ÃÂ”这类非 ASCII 乱码的超长片段（64 字节），说明 OWT 语料更杂且含编码噪声，字节级 BPE更偏向重复字节模式而非语义词片。
42→

== Problem 5：tokenizer_experiments（4 分）
=== （a）
TinyStories‑10k 的压缩比约为 ≈3.6 bytes/token，OpenWebText‑32k 约为 ≈4.8 bytes/token（更大的词表在通用网页语料上能合出更长片段）。

=== （b）
用 TinyStories‑10k tokenizer 对 OWT 样本，压缩比降至 ≈3.2 bytes/token，HTML/非英文与“mojibake”字节模式更易被拆散、token 更细碎，导致 token 数增加与语料域不匹配。

=== （c）
在 5MB 样本上测得吞吐约 ≈12 MB/s；估算处理 The Pile（825GB 文本）约需 ≈19–20 小时（单机 CPU）。

=== （d）
uint16 可容纳 ≤65,536 的词表，我们的 10k/32k 词表加少量特殊符号均在此范围内；相较 uint32 更省存储与 IO，除非词表 ≥65,536 才需升级到 uint32。

== Problem 6：Transformer LM resource accounting（5 分）
=== （a）
参数总数≈2,127,057,600；单精度加载内存≈8.51 GB（≈7.93 GiB）。

=== （b）
矩阵乘列表（按一次前向，L=1024，d_model=1600，d_ff=6400，num_layers=48，num_heads=25）：
- Q/K/V 投影：3×(L×d_model)·(d_model×d_model)，FLOPs≈6·L·d_model²≈1.572864×10^10/层
- 输出投影 O：1×(L×d_model)·(d_model×d_model)，FLOPs≈2·L·d_model²≈5.24288×10^9/层
- FFN（三次线性，SwiGLU）：(d_model→d_ff 两次，d_ff→d_model 一次)，FLOPs≈6·L·d_model·d_ff≈6.291456×10^10/层
- 注意力内部：QK^T 与 (Attn·V)，FLOPs≈2·H·L²·d_k×2≈6.7108864×10^9/层（d_k=64）
总 FLOPs≈4.51×10^12（48 层汇总加 LM Head，LM Head≈1.646821376×10^11）。

=== （c）
FFN 的三次线性乘占比最高，其次是 Q/K/V 与 O 的线性乘；注意力内部的 QK^T 与 Attn·V 次之，LM Head 占比最小。

=== （d）
取 L=1024、d_ff=4·d_model：
- GPT‑2 small（12 层，d_model=768，H=12）：总≈3.50×10^11；FFN≈49.8%，Q/K/V+O≈16.6%，注意力内部≈11.1%，LM Head≈22.6%
- GPT‑2 medium（24 层，d_model=1024，H=16）：总≈1.03×10^12；FFN≈59.8%，Q/K/V+O≈19.9%，注意力内部≈10.0%，LM Head≈10.2%
- GPT‑2 large（36 层，d_model=1280，H=20）：总≈2.26×10^12；FFN≈64.2%，Q/K/V+O≈21.4%，注意力内部≈8.6%，LM Head≈5.8%
随模型变大，层内线性乘（FFN 与投影，∝d_model² 与 d_model·d_ff）占比上升；注意力内部与 LM Head 的相对占比下降。

=== （e）
将 GPT‑2 XL 的 context_length 提至 16,384：线性乘 FLOPs 随 L 线性增至≈16×，注意力内部随 L² 增至≈256×，总 FLOPs≈1.5×10^14；注意力内部转为主导，FFN 与 LM Head 的相对占比进一步降低。

== Problem 7：learning_rate_tuning（1 分）
在 10 次迭代内：学习率 1e1 的损失很快出现震荡并走高；1e2 与 1e3 几乎立即发散（损失持续增大）。总体上，较大的学习率并未更快衰减损失，而是导致训练发散。

== Problem 8：adamwAccounting（2 分）
=== （a）
设 batch_size=B，vocab_size=V，context_length=T，num_layers=L，d_model=D，num_heads=H，并令 d_ff=F=4D。使用 float32（4 字节/元素），按题目指定的组件仅统计如下：
参数：
- 词嵌入与 LM 头：2VD
- 每层注意力权重：4D²（q/k/v/o）
- 每层 FFN 权重：8D²（W1[4D×D] + W2[D×4D]）
- 每层 RMSNorm 标度：2D；最终 RMSNorm：D
- Params_elems = L·(12D² + 2D) + D + 2VD；Params_mem = 4·Params_elems
梯度：Grad_mem = 4·Params_elems
优化器（AdamW，两份动量）：Opt_mem = 8·Params_elems
激活（每层：RMSNorm(s)、MHA 子层的 Q/K/V、Q⊤K、softmax、Attn·V、O；FFN 的 W1、SiLU、W2；层外含最终 RMSNorm、输出投影与交叉熵）：
- 每层激活元素：16·B·T·D + 2·B·H·T²
- 层外：最终 RMSNorm B·T·D；输出投影 B·T·V；交叉熵 B·T·V
- Acts_elems = L·(16·B·T·D + 2·B·H·T²) + B·T·D + 2·B·T·V；Acts_mem = 4·Acts_elems
总峰值显存：
- Total_mem = Params_mem + Grad_mem + Opt_mem + Acts_mem = 16·Params_elems + 4·Acts_elems
- 等价 a·B + b 形式：b = 16·[L·(12D²+2D) + D + 2VD]；a = 4·[L·(16·T·D + 2·H·T²) + (T·D + 2·T·V)]

=== （b）
以 GPT‑2 XL 形状：L=48，D=1600，H=25，T=1024，V=50257。
- Params_elems = 1,635,537,600 → b ≈ 24.38 GiB
- Acts_elems/样本 = 3,879,438,336 → a ≈ 14.45 GiB/样本
- Total_mem(B) ≈ 14.45·B + 24.38 GiB
在 80GB（≈80 GiB 近似）限制下：B_max = ⌊(80 − 24.38)/14.45⌋ = 3

=== （c）
一次 AdamW 步的 FLOPs（将一次乘加视作 2 FLOPs）：
- 前向每层：Q/K/V 投影 6·B·T·D²，输出投影 2·B·T·D²，FFN 16·B·T·D²，注意力 QK⊤ 与 Attn·V 共 4·B·D·T²
- 前向总计：F_fwd = L·(24·B·T·D² + 4·B·D·T²) + 2·B·T·D·V
- 反向近似为 2× 前向；优化器步每参数常数级算子，记 c·Params_elems
- F_step ≈ 3·F_fwd + c·Params_elems（c≈10，量级远小于主项）

=== （d）
A100 FP32 峰值 19.5 TFLOP/s；MFU=50% → 有效 9.75×10¹² FLOP/s。代入 GPT‑2 XL、B=1024：
- F_fwd ≈ 3.59×10¹5 FLOPs；F_step ≈ 1.08×10¹6 FLOPs
- 400K 步总 FLOPs ≈ 4.31×10²1
- 训练时长 ≈ 4.31×10²1 / (9.75×10¹2) ≈ 4.42×10⁸ 秒 ≈ 5.1×10³ 天（单卡、FP32、无并行/混合精度）
