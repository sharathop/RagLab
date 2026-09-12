import re
from typing import List, Dict, Any
from app.models.session import ChunkMetadata


class ChunkerService:
    @staticmethod
    def chunk_pages(
        pages_data: List[Dict[str, Any]],
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ) -> List[ChunkMetadata]:
        """
        Split page texts into overlapping chunks with precise metadata.
        Uses sliding window with sentence/word boundary snapping.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size - 1.")

        chunks: List[ChunkMetadata] = []
        chunk_counter = 0

        for page_item in pages_data:
            page_num = page_item["page"]
            text = page_item["text"]

            if not text:
                continue

            # Normalize whitespace
            normalized_text = re.sub(r'\s+', ' ', text).strip()
            if not normalized_text:
                continue

            step = chunk_size - chunk_overlap
            start_idx = 0
            text_len = len(normalized_text)

            while start_idx < text_len:
                end_idx = min(start_idx + chunk_size, text_len)
                
                # If not at the end, try to snap to nearest punctuation or space
                if end_idx < text_len:
                    # Look backwards for sentence end
                    sentence_boundary = -1
                    for punct in [". ", "? ", "! ", "\n"]:
                        boundary = normalized_text.rfind(punct, start_idx + step // 2, end_idx)
                        if boundary > sentence_boundary:
                            sentence_boundary = boundary + len(punct)

                    if sentence_boundary > start_idx:
                        end_idx = sentence_boundary
                    else:
                        # Otherwise snap to last space
                        space_idx = normalized_text.rfind(" ", start_idx + step // 2, end_idx)
                        if space_idx > start_idx:
                            end_idx = space_idx + 1

                chunk_text = normalized_text[start_idx:end_idx].strip()
                if chunk_text:
                    chunk_obj = ChunkMetadata(
                        chunk_id=f"chunk_{chunk_counter}",
                        page=page_num,
                        text=chunk_text,
                        token_count=len(chunk_text.split())
                    )
                    chunks.append(chunk_obj)
                    chunk_counter += 1

                if end_idx >= text_len:
                    break

                # Advance start_idx by step
                start_idx = max(start_idx + step, end_idx - chunk_overlap)

        return chunks
