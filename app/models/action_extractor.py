"""
Action Item Extraction Module
Uses HuggingFace transformers + regex patterns to extract tasks, 
assignees, and deadlines from meeting transcripts.
100% free, no API keys needed.
"""
import re
import time
import json
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class ActionExtractor:
    """
    Extracts action items, assignees, and deadlines from meeting text.
    Uses combination of transformer model and regex patterns.
    """
    
    def __init__(self):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.model_name = "llama-3.1-8b-instant"
        print("ActionExtractor initialized with Groq (Free)")
    
    def load_model(self):
        """Lazy load the summarization model for action extraction"""
        if self.extractor is None:
            print("Loading action extraction model...")
            self.extractor = pipeline(
                "text-generation",
                model="facebook/bart-large-cnn",
                max_new_tokens=50,
            )
            print("Action extraction model loaded")
    
    def extract(self, transcript: str, summary: str = "") -> List[Dict[str, Any]]:
        print("Extracting action items with Groq...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": f"""Analyze this conversation and extract ALL action items, tasks, commitments, and things people said they would do.

Look for phrases like:
- "I will..." / "We will..."
- "Let me..." / "I'll..."
- "I'm going to..."
- "You need to..."
- Promises to do something
- Arrangements being made

Return a JSON array:
[
  {{"task": "clear description", "assignee": "person name or 'Not specified'", "deadline": "when or 'Not specified'", "priority": "High/Medium/Low"}}
]

Transcript:
{transcript[:3000]}"""
                }],
                temperature=0.1,
                max_tokens=500
            )
            
            import json
            text = response.choices[0].message.content
            
            try:
                items = json.loads(text)
            except:
                match = re.search(r'\[.*\]', text, re.DOTALL)
                items = json.loads(match.group()) if match else []
            
            for item in items:
                if item.get('priority') not in ['High', 'Medium', 'Low']:
                    item['priority'] = 'Medium'
                item['confidence'] = 0.9
                item['source'] = 'groq-llama'
            
            return items
            
        except Exception as e:
            print(f"Groq extraction error: {str(e)}")
            return []

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the extraction approach"""
        return {
            "model_name": "facebook/bart-large-cnn + Custom Regex",
            "type": "Hybrid (Local NLP + Pattern Matching)",
            "capabilities": [
                "Task extraction",
                "Assignee detection",
                "Deadline recognition",
                "Priority classification",
                "Confidence scoring"
            ],
            "advantages": [
                "No API costs",
                "Fast processing",
                "Combines ML with rule-based accuracy"
            ]
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