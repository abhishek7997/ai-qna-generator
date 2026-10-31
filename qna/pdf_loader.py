from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdfs(data_dir: Path) -> list[list[Document]]:
    """Return one list of page Documents per PDF in data_dir."""
    return [
        [
            Document(
                page_content=page.extract_text() or "",
                metadata={"source": str(file_path), "page": i},
            )
            for i, page in enumerate(PdfReader(file_path).pages)
        ]
        for file_path in sorted(data_dir.glob("*.pdf"))
    ]
