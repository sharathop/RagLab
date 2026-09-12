import os
import re
import time
import requests
from typing import List, Tuple, Optional
from app.models.schemas import EvaluationMetrics, RetrievedSource

# Your standalone hallucination-detection framework, deployed separately on
# Hugging Face Spaces. Configurable via env var so a local/self-hosted copy
# can be swapped in later without a code change.
EVAL_FRAMEWORK_URL = os.getenv(
    "EVAL_FRAMEWORK_URL", "https://sha6th-llm-eval-ap.hf.space/evaluate"
)

# Free-tier HF Spaces sleep after inactivity and can take 30-60s to wake on
# the first request after a period of no traffic. Timeout is generous on
# purpose so a cold-start doesn't look like a hard failure.
EVAL_FRAMEWORK_TIMEOUT_SECONDS = 75

# Tuning knobs for the cosine length/format-bias warning.
# Not scientifically calibrated — starting points to refine against real
# test questions, same as the rest of this project's documented limitations.
COSINE_LENGTH_RATIO_THRESHOLD = 4.0
LIST_FORMAT_PATTERN = re.compile(r'^\s*(\d+[\.\)]|[-*•])\s+')


class EvaluatorError(Exception):
    """Raised when the external hallucination-detection framework can't be
    reached or returns something unexpected.

    No local fallback scoring here on purpose: silently substituting a
    heuristic approximation when the real framework is unreachable would
    mean the displayed NLI/cosine/BERTScore numbers sometimes come from a
    completely different (and much weaker) method with no indication to
    the user which one they're looking at. Fail loudly instead.
    """
    pass


class EvaluatorService:
    """
    Calls the standalone LLM Evaluation & Hallucination Detection Framework
    (a separate project, deployed at EVAL_FRAMEWORK_URL) for NLI, cosine,
    and BERTScore. Fluency is intentionally not used in this project.

    Design: Option B — no aggregate verdict is computed or surfaced here.
    The framework's own `final_verdict` field is deliberately ignored: it's
    built from the same kind of fused-signal logic (see its aggregator)
    that this project's own testing showed can be misleading — e.g. a low
    cosine score alone can look damning even when it's explained by answer
    length/format rather than a real faithfulness problem. Each metric's
    own single-signal verdict label (e.g. cosine's "Relevant"/"Irrelevant")
    is still shown, since that's just a categorization of one number, not
    a fused system judgment.
    """

    @classmethod
    def evaluate(
        cls,
        question: str,
        answer: str,
        sources: List[RetrievedSource],
        embedding_model: str = "all-MiniLM-L6-v2"  # unused here; kept for call-site compatibility
    ) -> Tuple[EvaluationMetrics, float]:
        t0 = time.perf_counter()

        context_text = " ".join([s.text for s in sources]) if sources else ""

        if not context_text.strip():
            # The framework's /evaluate rejects empty context with a 400 —
            # fail the same way locally rather than sending a request we
            # already know will be rejected.
            raise EvaluatorError("No retrieved context available to evaluate against.")

        payload = {
            "context": context_text,
            "question": question,
            "llm_response": answer,
        }

        try:
            resp = requests.post(EVAL_FRAMEWORK_URL, json=payload, timeout=EVAL_FRAMEWORK_TIMEOUT_SECONDS)
        except requests.exceptions.Timeout:
            raise EvaluatorError(
                f"Evaluation framework timed out after {EVAL_FRAMEWORK_TIMEOUT_SECONDS}s. "
                "If it's hosted on a free Hugging Face Space, it may be asleep — "
                "try again in about a minute."
            )
        except requests.exceptions.RequestException as e:
            raise EvaluatorError(f"Could not reach evaluation framework: {e}")

        if resp.status_code != 200:
            raise EvaluatorError(
                f"Evaluation framework returned {resp.status_code}: {resp.text[:300]}"
            )

        try:
            data = resp.json()
            nli = data["nli"]
            cosine = data["cosine"]
            bert = data["bert_score"]
        except (ValueError, KeyError) as e:
            raise EvaluatorError(f"Unexpected response shape from evaluation framework: {e}")

        cosine_score = float(cosine["score"])
        cosine_warning = cls.check_cosine_bias(question, answer, cosine_score)

        eval_time_ms = (time.perf_counter() - t0) * 1000.0

        metrics = EvaluationMetrics(
            nli_score=round(float(nli["score"]), 4),
            nli_label=nli.get("verdict"),
            cosine_score=round(cosine_score, 4),
            cosine_label=cosine.get("verdict"),
            bertscore=round(float(bert["score"]), 4),
            bertscore_label=bert.get("verdict"),
            cosine_warning=cosine_warning,
        )

        return metrics, eval_time_ms

    @classmethod
    def check_cosine_bias(cls, question: str, answer: str, cosine_score: float,
                           threshold: float = 0.6) -> Optional[str]:
        """
        Flags the known cosine failure mode, confirmed present in the real
        framework: cosine here compares QUESTION vs RESPONSE embeddings
        directly, so a long and/or list-formatted answer naturally drifts
        away from a short question in embedding space, regardless of
        factual accuracy. This does NOT change the cosine score or its
        verdict from the framework — it only attaches a note so a low
        score isn't misread as evidence the answer is wrong.

        `threshold` here is a display-triage cutoff (does this score even
        need a warning), independent of whatever threshold the user has
        set in the UI for their own reading of the score.
        """
        if cosine_score >= threshold:
            return None

        q_words = len(question.split())
        a_words = len(answer.split())
        length_ratio = a_words / max(q_words, 1)

        is_list_format = bool(LIST_FORMAT_PATTERN.match(answer.strip()))

        if length_ratio > COSINE_LENGTH_RATIO_THRESHOLD or is_list_format:
            return (
                f"Cosine score is low ({cosine_score:.2f}), but the answer is "
                f"~{length_ratio:.1f}x longer than the question"
                + (" and list-formatted" if is_list_format else "")
                + ". Cosine here compares the question directly to the answer, so this "
                "commonly happens even for correct, well-grounded answers — check "
                "NLI/BERTScore before treating this score as a sign the answer is wrong."
            )

        return (
            f"Cosine score is low ({cosine_score:.2f}) and the answer's length/format "
            "doesn't obviously explain it — worth reviewing alongside NLI/BERTScore."
        )
