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
