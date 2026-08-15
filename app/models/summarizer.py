"""
Meeting Summarization Module
Uses Groq API (FREE - OpenAI GPT-OSS 20B)

Note: gpt-oss models are reasoning models. They spend completion tokens on
internal chain-of-thought before writing the final answer. Using a low
max_tokens budget causes empty message.content (finish_reason=length).
Always use max_completion_tokens with enough headroom, and prefer
reasoning_effort="low" for simple extraction/summarization tasks.
"""
import os
import re
import time
from typing import Dict, Any, List, Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def _message_text(message) -> str:
    """Safely pull final answer text from a Groq chat message."""
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    # Some SDK versions expose reasoning separately; content should hold the answer,
    # but fall back if content is empty and reasoning looks like the final output.
    reasoning = getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return ""


class MeetingSummarizer:

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "openai/gpt-oss-20b"
        # Low effort keeps reasoning short so the final answer fits in the budget
        self.reasoning_effort = "low"
        print(f"Summarizer initialized with Groq ({self.model_name})")

    def load_model(self):
        """API is always ready"""
        pass

    def _chat(
        self,
        messages: List[Dict[str, str]],
        max_completion_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """Call Groq chat completions with gpt-oss-safe settings."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=self.reasoning_effort,
        )
        choice = response.choices[0]
        text = _message_text(choice.message)
        if not text:
            finish = getattr(choice, "finish_reason", None)
            usage = getattr(response, "usage", None)
            print(
                f"Groq empty content (finish_reason={finish}, usage={usage}). "
                "Increase max_completion_tokens if finish_reason=length."
            )
        return text

    def summarize(self, transcript: str) -> Dict[str, Any]:
        """Generate summary - auto-detects if chunking needed"""
        if len(transcript) > 4000:
            return self.summarize_long(transcript)

        print("Generating summary with Groq (GPT-OSS 20B)...")
        start_time = time.time()

        try:
            full_summary = self._chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a meeting summarizer. Summarize accurately in "
                            "2-3 sentences. Only use facts from the transcript. "
                            "Do not add information. Reply with the summary only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Summarize this conversation:\n\n{transcript[:3000]}",
                    },
                ],
                max_completion_tokens=1024,
                temperature=0.3,
            )

            if not full_summary:
                print("Empty summary from Groq, using extractive fallback")
                return self._extractive_fallback(transcript)

            key_points = self._extract_key_points(transcript)
            decisions = self._extract_decisions(transcript)
            title = self._infer_title(full_summary)

            processing_time = time.time() - start_time
            print(f"Summary generated in {processing_time:.2f}s")

            return {
                "title": title,
                "summary": full_summary,
                "key_points": key_points,
                "decisions": decisions,
                "processing_time": processing_time,
            }

        except Exception as e:
            print(f"Groq error: {str(e)}, using extractive fallback")
            return self._extractive_fallback(transcript)

    def _extract_key_points(self, transcript: str) -> List[str]:
        """Extract key points using Groq"""
        try:
            text = self._chat(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "List the key points from this conversation as bullet points "
                            "(be brief and accurate). Reply with bullets only.\n\n"
                            f"{transcript[:3000]}"
                        ),
                    }
                ],
                max_completion_tokens=1024,
                temperature=0.3,
            )
            if not text:
                return self._extractive_key_points(transcript)
            points = []
            for line in text.split("\n"):
                line = line.strip().lstrip("•-*0123456789.").strip()
                if len(line) > 10:
                    points.append(line)
            return points[:8] if points else self._extractive_key_points(transcript)
        except Exception as e:
            print(f"Key points error: {e}")
            return self._extractive_key_points(transcript)

    def _extract_decisions(self, transcript: str) -> List[str]:
        """Extract decisions using Groq"""
        try:
            text = self._chat(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "List any decisions, commitments, or action items from this "
                            "conversation as bullet points. If none, reply with exactly: None\n\n"
                            f"{transcript[:2000]}"
                        ),
                    }
                ],
                max_completion_tokens=1024,
                temperature=0.3,
            )
            if not text or "none" in text.lower().strip()[:20]:
                if text and "none" in text.lower() and len(text.strip()) < 30:
                    return []
            if not text:
                return []
            if text.strip().lower() == "none":
                return []
            decisions = []
            for line in text.split("\n"):
                line = line.strip().lstrip("•-*0123456789.").strip()
                if len(line) > 10 and line.lower() != "none":
                    decisions.append(line)
            return decisions[:5]
        except Exception as e:
            print(f"Decisions error: {e}")
            return []

    def _infer_title(self, text: str) -> str:
        """Infer a title from text"""
        sentences = re.split(r"[.!?]+", text)
        for s in sentences:
            s = s.strip()
            if len(s) > 10:
                words = s.split()[:8]
                return " ".join(words)
        return "Meeting Summary"

    def _extractive_key_points(self, transcript: str) -> List[str]:
        """Extractive key points fallback"""
        sentences = re.split(r"[.!?]+", transcript)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        skip = ["hello", "good morning", "thank you", "bye", "how may i"]
        content = [s for s in sentences if not any(sk in s.lower() for sk in skip)]

        keywords = [
            "topic",
            "problem",
            "issue",
            "need",
            "want",
            "api",
            "product",
            "order",
            "customer",
            "feature",
            "action",
            "plan",
            "will",
        ]

        scored = [(s, sum(1 for kw in keywords if kw in s.lower())) for s in content]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:8]]

    def _extractive_fallback(self, transcript: str) -> Dict[str, Any]:
        """Complete extractive fallback"""
        sentences = re.split(r"[.!?]+", transcript)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        skip = [
            "hello",
            "good morning",
            "thank you",
            "bye",
            "how may i",
            "you're welcome",
            "welcome",
            "nice to meet",
            "how are you",
        ]

        content = [s for s in sentences if not any(sk in s.lower() for sk in skip)]

        keywords = [
            "topic",
            "problem",
            "issue",
            "solution",
            "need",
            "want",
            "api",
            "product",
            "order",
            "customer",
            "feature",
            "plan",
        ]

        scored = [(s, sum(1 for kw in keywords if kw in s.lower())) for s in content]
        scored.sort(key=lambda x: x[1], reverse=True)

        summary = ". ".join([s for s, _ in scored[:3]]) + "."
        key_points = [s for s, _ in scored[:8]]

        return {
            "title": self._infer_title(summary if summary else transcript),
            "summary": summary if summary else (content[0] if content else "No summary"),
            "key_points": key_points if key_points else content[:5],
            "decisions": [],
            "processing_time": 0,
        }

    def summarize_long(self, transcript: str) -> Dict[str, Any]:
        """
        Summarize long transcripts by chunking.
        Splits into 2000-char chunks, summarizes each, then combines.
        """
        print("Long transcript detected - using chunked summarization...")
        start_time = time.time()

        chunk_size = 2000
        words = transcript.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        print(f"Split into {len(chunks)} chunks")

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i+1}/{len(chunks)}...")
            try:
                text = self._chat(
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Summarize this conversation segment in 2-3 sentences. "
                                f"Reply with the summary only.\n\n{chunk}"
                            ),
                        }
                    ],
                    max_completion_tokens=768,
                    temperature=0.3,
                )
                chunk_summaries.append(text if text else chunk[:200])
            except Exception as e:
                print(f"Chunk {i+1} error: {str(e)}")
                chunk_summaries.append(chunk[:200])

        combined = " ".join(chunk_summaries)

        try:
            final_summary = self._chat(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Create a final 3-4 sentence summary from these segment "
                            f"summaries. Reply with the summary only.\n\n{combined}"
                        ),
                    }
                ],
                max_completion_tokens=1024,
                temperature=0.3,
            )
            if not final_summary:
                final_summary = combined[:500]
        except Exception:
            final_summary = combined[:500]

        key_points = self._extract_key_points(combined[:3000])

        processing_time = time.time() - start_time
        print(f"Chunked summary complete in {processing_time:.2f}s")

        return {
            "title": self._infer_title(final_summary),
            "summary": final_summary,
            "key_points": key_points,
            "decisions": [],
            "processing_time": processing_time,
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "type": "Cloud API (FREE - Groq)",
            "provider": "Groq / OpenAI GPT-OSS 20B",
            "cost": "$0.00 (Free tier)",
        }
