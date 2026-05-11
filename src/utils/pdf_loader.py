from pypdf import PdfReader
import re
from src.core.config import settings

class PDFLoader:
    @staticmethod
    def extract_text(pdf_path: str) -> str:
        """Extracts and cleans text from a PDF file."""
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + " "
        
        # Basic cleaning: remove extra whitespace and newlines
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        return full_text

    @staticmethod
    def get_chunks(text: str) -> list[str]:
        """Split text into overlapping chunks based on config."""
        words = text.split()
        chunks = []
        step = settings.CHUNK_SIZE - settings.CHUNK_OVERLAP
        
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + settings.CHUNK_SIZE])
            chunks.append(chunk)
            if i + settings.CHUNK_SIZE >= len(words):
                break
        return chunks