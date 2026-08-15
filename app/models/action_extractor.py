"""
Action Item Extraction Module
Uses Groq API (FREE - OpenAI GPT-OSS 20B) to extract tasks,
assignees, and deadlines from meeting transcripts.

gpt-oss is a reasoning model: low max_tokens often yields empty content.
Use max_completion_tokens + reasoning_effort="low".
"""
import re
import time
import json
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def _message_text(message) -> str:
    """Safely pull final answer text from a Groq chat message."""
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    reasoning = getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return ""


def _parse_json_array(text: str) -> List[Dict[str, Any]]:
    """Parse a JSON array from model output, tolerating markdown fences."""
    if not text:
        return []

    cleaned = text.strip()
    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Sometimes models wrap as {"items": [...]}
            for key in ("items", "action_items", "actions", "tasks"):
                if isinstance(data.get(key), list):
                    return data[key]
            return [data]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return []


class ActionExtractor:
    """
    Extracts action items, assignees, and deadlines from meeting text.
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "openai/gpt-oss-20b"
        self.reasoning_effort = "low"
        print("ActionExtractor initialized with Groq (Free)")

    def load_model(self):
        """API is always ready — kept for interface compatibility."""
        pass

    def extract(self, transcript: str, summary: str = "") -> List[Dict[str, Any]]:
        print("Extracting action items with Groq (GPT-OSS 20B)...")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Analyze this conversation and extract ALL action items, tasks, commitments, and things people said they would do.

Look for phrases like:
- "I will..." / "We will..."
- "Let me..." / "I'll..."
- "I'm going to..."
- "You need to..."
- Promises to do something
- Arrangements being made

Return ONLY a valid JSON array (no markdown, no explanation):
[
  {{"task": "clear description", "assignee": "person name or Not specified", "deadline": "when or Not specified", "priority": "High/Medium/Low"}}
]

If there are no action items, return: []

Transcript:
{transcript[:3000]}""",
                    }
                ],
                temperature=0.3,
                max_completion_tokens=2048,
                reasoning_effort=self.reasoning_effort,
            )

            choice = response.choices[0]
            text = _message_text(choice.message)

            if not text:
                finish = getattr(choice, "finish_reason", None)
                print(
                    f"Groq extraction empty content (finish_reason={finish}). "
                    "Returning no action items."
                )
                return []

            items = _parse_json_array(text)
            normalized = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                task = item.get("task") or item.get("description") or item.get("action")
                if not task or not str(task).strip():
                    continue
                priority = item.get("priority", "Medium")
                if priority not in ("High", "Medium", "Low"):
                    priority = "Medium"
                normalized.append(
                    {
                        "task": str(task).strip(),
                        "assignee": str(
                            item.get("assignee") or "Not specified"
                        ).strip(),
                        "deadline": str(
                            item.get("deadline") or "Not specified"
                        ).strip(),
                        "priority": priority,
                        "confidence": float(item.get("confidence", 0.9)),
                        "source": "groq-gpt-oss",
                    }
                )

            print(f"Extracted {len(normalized)} action items")
            return normalized

        except Exception as e:
            print(f"Groq extraction error: {str(e)}")
            return []

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the extraction approach"""
        return {
            "model_name": self.model_name,
            "type": "Cloud API (FREE - Groq)",
            "provider": "Groq / OpenAI GPT-OSS 20B",
            "capabilities": [
                "Task extraction",
                "Assignee detection",
                "Deadline recognition",
                "Priority classification",
                "Confidence scoring",
            ],
            "advantages": [
                "No API costs",
                "Fast processing",
                "Combines ML with rule-based accuracy",
            ],
        }


if __name__ == "__main__":
    extractor = ActionExtractor()

    test_transcript = """
    John needs to prepare the sales report by next Friday. Sarah will handle
    the client presentation, it's urgent. We should review the budget when possible.
    Action item: Mike must schedule the team meeting for next week.
    The deadline for the project proposal is end of month, assigned to David.
    """

    test_summary = "The team discussed sales reports, client presentations, and project proposals."

    items = extractor.extract(test_transcript, test_summary)

    print("\n--- Extracted Action Items ---")
    for i, item in enumerate(items, 1):
        print(f"\n{i}. Task: {item['task'][:100]}")
        print(f"   Assignee: {item['assignee']}")
        print(f"   Deadline: {item['deadline']}")
        print(f"   Priority: {item['priority']}")
        print(f"   Confidence: {item['confidence']:.0%}")
