import os
from typing import List, Dict, Tuple
import pypdf

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


class PDFParserError(Exception):
    pass


class PDFParserService:
    @staticmethod
    def validate_pdf(file_path: str) -> None:
        """Validate that the file exists, is non-empty, <= 10MB, and has PDF signature."""
        if not os.path.exists(file_path):
            raise PDFParserError("Uploaded file not found on server.")

        size = os.path.getsize(file_path)
        if size == 0:
            raise PDFParserError("The uploaded PDF file is empty.")

        if size > MAX_FILE_SIZE:
            raise PDFParserError(f"File size exceeds 10 MB limit ({size / (1024*1024):.2f} MB).")

        # Check PDF magic bytes (%PDF)
        with open(file_path, "rb") as f:
            header = f.read(5)
            if not header.startswith(b"%PDF"):
                raise PDFParserError("The uploaded file does not appear to be a valid PDF document.")

    @staticmethod
    def parse_pdf(file_path: str) -> Tuple[List[Dict[str, any]], int]:
        """
        Extract text from each page of the PDF.
        Returns:
            Tuple of (pages_data, total_pages)
            pages_data: [{"page": 1, "text": "...", "char_count": 120}, ...]
        """
        PDFParserService.validate_pdf(file_path)

        pages_data = []
        try:
            reader = pypdf.PdfReader(file_path)
            total_pages = len(reader.pages)

            if total_pages == 0:
                raise PDFParserError("The PDF contains no pages.")

            extracted_any_text = False
            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""

                clean_text = text.strip()
                if clean_text:
                    extracted_any_text = True

                pages_data.append({
                    "page": page_num,
                    "text": clean_text,
                    "char_count": len(clean_text)
                })

            if not extracted_any_text:
                raise PDFParserError(
                    "No extractable text was found in the PDF. It might be a scanned image-only PDF."
                )

            return pages_data, total_pages

        except PDFParserError:
            raise
        except Exception as e:
            raise PDFParserError(f"Failed to parse PDF document: {str(e)}")
