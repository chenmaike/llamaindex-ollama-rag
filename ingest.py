from pathlib import Path

import fitz
from dotenv import load_dotenv
from llama_index.core import Document, Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
CHUNK_SIZE = 320
CHUNK_OVERLAP = 60


def configure_llama_index() -> None:
    load_dotenv()
    Settings.llm = Ollama(model="qwen2.5:1.5b", request_timeout=300.0)
    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://127.0.0.1:11434",
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def load_pdf_documents() -> list[Document]:
    documents: list[Document] = []
    empty_pages: list[tuple[Path, int]] = []

    for pdf_path in sorted(DATA_DIR.rglob("*.pdf")):
        pdf = fitz.open(pdf_path)
        for page_index, page in enumerate(pdf, start=1):
            text = page.get_text().strip()
            if not text:
                empty_pages.append((pdf_path, page_index))
                continue

            documents.append(
                Document(
                    text=text,
                    metadata={
                        "file_name": pdf_path.name,
                        "file_path": str(pdf_path),
                        "file_type": "pdf",
                        "page_label": page_index,
                    },
                )
            )

    for pdf_path, page_index in empty_pages:
        print(
            "Warning: "
            f"{pdf_path.name} 第 {page_index} 页没有提取到文字，"
            "可能是扫描页或图片页，当前版本不会把这一页加入索引。"
        )

    return documents


def load_text_documents() -> list[Document]:
    return SimpleDirectoryReader(
        str(DATA_DIR),
        exclude=["*.pdf"],
    ).load_data()


def main() -> None:
    configure_llama_index()

    documents = load_text_documents() + load_pdf_documents()
    if not documents:
        raise RuntimeError(f"No documents found in {DATA_DIR}")

    index = VectorStoreIndex.from_documents(documents)

    index.storage_context.persist(persist_dir=str(STORAGE_DIR))

    print(f"Indexed {len(documents)} document(s).")
    print(f"Chunk size: {CHUNK_SIZE}, chunk overlap: {CHUNK_OVERLAP}")
    print(f"Persisted index to {STORAGE_DIR}")


if __name__ == "__main__":
    main()
