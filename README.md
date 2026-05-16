# LlamaIndex + Ollama 本地 RAG 示例

这个项目不使用 OpenAI API，全部通过 Ollama 在本机运行。

## 运行步骤

```bash
cd "/Users/mimi/Documents/New project 3/llamaindex_ollama_rag"
source .venv/bin/activate

ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text

pip install -r requirements.txt
python ingest.py
python ask.py "这个项目为什么不需要 OpenAI 额度？"
```

把你自己的 `.txt`、`.md` 文件放进 `data/` 后，重新运行：

```bash
python ingest.py
```

然后再提问：

```bash
python ask.py "根据资料总结重点"
```

## PDF 支持

这个项目支持读取 `data/` 目录里的 PDF 文件。PDF 会通过 PyMuPDF 提取文字，并按页加入本地索引。

添加 PDF 后，必须重新运行：

```bash
python ingest.py
```

然后再提问：

```bash
python ask.py "根据 PDF 内容回答问题"
```

如果 PDF 是扫描版图片，里面没有可复制文本，当前版本无法直接读取，需要后续增加 OCR。
