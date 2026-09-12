import os
from dotenv import load_dotenv
load_dotenv()
import time
import httpx
from typing import List, Tuple
from app.models.schemas import RetrievedSource

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


class GeneratorError(Exception):
    pass


class GeneratorService:
    @staticmethod
    def get_api_key() -> str:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        return api_key

    @staticmethod
    def format_context(sources: List[RetrievedSource]) -> str:
        if not sources:
            return "No relevant context found in document."

        context_blocks = []
        for i, source in enumerate(sources, 1):
            context_blocks.append(
                f"[Source {i} - Page {source.page} (Chunk {source.chunk_id})]:\n{source.text}"
            )
        return "\n\n".join(context_blocks)

    @classmethod
    def generate_answer(
        cls,
        question: str,
        sources: List[RetrievedSource]
    ) -> Tuple[str, float]:
        """
        Generate answer using Groq Openai/gpt-oss-120 with strict context prompt.
        Returns:
            Tuple of (answer_text, llm_time_ms)
        """
        api_key = cls.get_api_key()
        if not api_key:
            raise GeneratorError(
                "GROQ_API_KEY is not configured. Please set the GROQ_API_KEY environment variable in .env to enable Llama 3.3 70B generation."
            )

        context_str = cls.format_context(sources)
        system_prompt = (
    "You are a document question-answering assistant.\n"
    "Answer using only the provided context.\n"
    "If the answer cannot be found in the context, say that the information is not available.\n"
    "Do not extrapolate or assume information beyond what is explicitly stated.\n"
    "\n"
    "Formatting rules — the answer is displayed as plain text, not rendered markdown, "
    "so raw markdown syntax would show up as literal symbols instead of formatting:\n"
    "- Do not use markdown tables (no '|' pipe characters or '---' separator rows).\n"
    "- Do not use markdown emphasis syntax (no '**bold**', '*italic*', or '#' headers).\n"
    "- If you need to present multiple items, list them as plain numbered lines "
    "(e.g. '1. Item one') or short sentences — never as a table.\n"
    "- Write in plain prose sentences by default; only use a numbered list when the "
    "content is genuinely a sequence or enumeration."
)

        user_content = f"Context:\n{context_str}\n\nQuestion:\n{question}"

        payload = {
            "model": DEFAULT_GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.1,
            "max_tokens": 1024
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(GROQ_API_URL, json=payload, headers=headers)

                if response.status_code == 401:
                    raise GeneratorError("Groq API authentication failed: Invalid API key provided.")
                if response.status_code == 429:
                    raise GeneratorError("Groq API rate limit exceeded. Please try again in a few moments.")
                if response.status_code != 200:
                    raise GeneratorError(
                        f"Groq API error ({response.status_code}): {response.text[:200]}"
                    )

                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    raise GeneratorError("Groq API returned an empty response.")

                answer = choices[0].get("message", {}).get("content", "").strip()
                llm_time_ms = (time.perf_counter() - t0) * 1000.0
                return answer, llm_time_ms

        except httpx.RequestError as e:
            raise GeneratorError(f"Network error connecting to Groq API: {str(e)}")
        except GeneratorError:
            raise
        except Exception as e:
            raise GeneratorError(f"Unexpected error during answer generation: {str(e)}")
