from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding


BASE_DIR = Path(__file__).parent
STORAGE_DIR = BASE_DIR / "storage"
PREVIEW_CHARS = 180


def configure_llama_index() -> None:
    load_dotenv()
    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://127.0.0.1:11434",
    )


def main() -> None:
    configure_llama_index()

    if not (STORAGE_DIR / "docstore.json").exists():
        raise RuntimeError("Index not found. Run `python ingest.py` first.")

    storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    index = load_index_from_storage(storage_context)
    docstore = index.storage_context.docstore
    nodes = list(docstore.docs.values())

    print(f"Total chunks: {len(nodes)}")

    for i, node in enumerate(nodes, start=1):
        metadata = node.metadata
        file_name = metadata.get("file_name", "unknown")
        page_label = metadata.get("page_label")
        text = node.get_content().replace("\n", " ")
        preview = text[:PREVIEW_CHARS]

        suffix = f", page {page_label}" if page_label else ""
        print(f"\n{i}. {file_name}{suffix}")
        print(f"   chars: {len(text)}")
        print(f"   preview: {preview}")


if __name__ == "__main__":
    main()
