from pathlib import Path
import sys

from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


BASE_DIR = Path(__file__).parent
STORAGE_DIR = BASE_DIR / "storage"
SOURCE_PREVIEW_CHARS = 300
SIMILARITY_TOP_K = 3


def configure_llama_index() -> None:
    load_dotenv()
    Settings.llm = Ollama(model="qwen2.5:1.5b", request_timeout=300.0)
    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://127.0.0.1:11434",
    )


def main() -> None:
    configure_llama_index()

    if not (STORAGE_DIR / "docstore.json").exists():
        raise RuntimeError("Index not found. Run `python ingest.py` first.")

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = "这个项目使用什么本地模型？"

    storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    index = load_index_from_storage(storage_context)
    qa_prompt = PromptTemplate(
        "你是一个严谨的中文知识库问答助手。"
        "请只根据下面提供的资料回答问题。"
        "如果资料里没有答案，请明确说“资料中没有找到相关信息”。"
        "除非用户明确要求其他语言，否则始终使用中文回答。\n\n"
        "资料：\n{context_str}\n\n"
        "问题：{query_str}\n\n"
        "中文回答："
    )

    query_engine = index.as_query_engine(
        similarity_top_k=SIMILARITY_TOP_K,
        text_qa_template=qa_prompt,
    )

    response = query_engine.query(question)

    print("\n回答：")
    print(response)

    print("\n参考资料：")
    for i, source_node in enumerate(response.source_nodes, start=1):
        metadata = source_node.node.metadata
        file_name = metadata.get("file_name", "unknown")
        file_path = metadata.get("file_path", "")
        page_label = metadata.get("page_label")
        score = source_node.score
        text = source_node.node.get_content()
        preview = text[:SOURCE_PREVIEW_CHARS].replace("\n", " ")

        print(f"\n{i}. {file_name}")
        if page_label:
            print(f"   页码：{page_label}")
        if file_path:
            print(f"   路径：{file_path}")
        if score is not None:
            print(f"   相似度：{score:.4f}")
        print(f"   片段：{preview}")



if __name__ == "__main__":
    main()
