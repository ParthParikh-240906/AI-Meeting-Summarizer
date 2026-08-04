"""
Meeting Summarization Module
Uses Groq API (FREE - Llama 3.1, Mixtral, etc.)
"""
import os
import re
import time
from typing import Dict, Any, List
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class MeetingSummarizer:

    def summarize(self, transcript: str) -> Dict[str, Any]:
        """Generate summary - auto-detects if chunking needed"""
        
        # If transcript is very long, use chunking
        if len(transcript) > 4000:
            return self.summarize_long(transcript)
        
        # Rest of existing summarize method...
    
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "llama-3.1-8b-instant"
        print(f"Summarizer initialized with Groq ({self.model_name})")
    
    def load_model(self):
        """API is always ready"""
        pass
    
    def summarize(self, transcript: str) -> Dict[str, Any]:
        """Generate summary using Groq (free Llama 3.1)"""
        print("Generating summary with Groq (Llama 3.1)...")
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a meeting summarizer. Summarize accurately in 2-3 sentences. Only use facts from the transcript. Do not add information."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize this conversation:\n\n{transcript[:3000]}"
                    }
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            full_summary = response.choices[0].message.content.strip()
            
            # Get key points
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
                "processing_time": processing_time
            }
            
        except Exception as e:
            print(f"Groq error: {str(e)}, using extractive fallback")
            return self._extractive_fallback(transcript)
    
    def _extract_key_points(self, transcript: str) -> List[str]:
        """Extract key points using Groq"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": f"List the key points from this conversation as bullet points (be brief and accurate):\n\n{transcript[:3000]}"
                }],
                temperature=0.1,
                max_tokens=500
            )
            text = response.choices[0].message.content
            points = []
            for line in text.split('\n'):
                line = line.strip().lstrip('•-*0123456789.').strip()
                if len(line) > 10:
                    points.append(line)
            return points[:8]
        except:
            return self._extractive_key_points(transcript)
    
    def _extract_decisions(self, transcript: str) -> List[str]:
        """Extract decisions using Groq"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": f"List any decisions, commitments, or action items from this conversation. If none, just say 'None':\n\n{transcript[:2000]}"
                }],
                temperature=0.1,
                max_tokens=300
            )
            text = response.choices[0].message.content
            if "none" in text.lower():
                return []
            decisions = []
            for line in text.split('\n'):
                line = line.strip().lstrip('•-*0123456789.').strip()
                if len(line) > 10:
                    decisions.append(line)
            return decisions[:5]
        except:
            return []
    
    def _infer_title(self, text: str) -> str:
        """Infer a title from text"""
        sentences = re.split(r'[.!?]+', text)
        for s in sentences:
            s = s.strip()
            if len(s) > 10:
                words = s.split()[:8]
                return " ".join(words)
        return "Meeting Summary"
    
    def _extractive_key_points(self, transcript: str) -> List[str]:
        """Extractive key points fallback"""
        sentences = re.split(r'[.!?]+', transcript)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        skip = ['hello', 'good morning', 'thank you', 'bye', 'how may i']
        content = [s for s in sentences if not any(sk in s.lower() for sk in skip)]
        
        keywords = ['topic', 'problem', 'issue', 'need', 'want', 'api', 'product',
                    'order', 'customer', 'feature', 'action', 'plan', 'will']
        
        scored = [(s, sum(1 for kw in keywords if kw in s.lower())) for s in content]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:8]]
    
    def _extractive_fallback(self, transcript: str) -> Dict[str, Any]:
        """Complete extractive fallback"""
        sentences = re.split(r'[.!?]+', transcript)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        skip = ['hello', 'good morning', 'thank you', 'bye', 'how may i', 
                'you\'re welcome', 'welcome', 'nice to meet', 'how are you']
        
        content = [s for s in sentences if not any(sk in s.lower() for sk in skip)]
        
        keywords = ['topic', 'problem', 'issue', 'solution', 'need', 'want', 
                    'api', 'product', 'order', 'customer', 'feature', 'plan']
        
        scored = [(s, sum(1 for kw in keywords if kw in s.lower())) for s in content]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        summary = ". ".join([s for s, _ in scored[:3]]) + "."
        key_points = [s for s, _ in scored[:8]]
        
        return {
            "title": self._infer_title(summary if summary else transcript),
            "summary": summary if summary else (content[0] if content else "No summary"),
            "key_points": key_points if key_points else content[:5],
            "decisions": [],
            "processing_time": 0
        }

    def summarize_long(self, transcript: str) -> Dict[str, Any]:
        """
        Summarize long transcripts by chunking.
        Splits into 2000-char chunks, summarizes each, then combines.
        """
        print("Long transcript detected - using chunked summarization...")
        start_time = time.time()
        
        # Split into chunks of ~2000 characters
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
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i+1}/{len(chunks)}...")
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{
                        "role": "user",
                        "content": f"Summarize this conversation segment in 2-3 sentences:\n\n{chunk}"
                    }],
                    temperature=0.1,
                    max_tokens=200
                )
                chunk_summaries.append(response.choices[0].message.content.strip())
            except Exception as e:
                print(f"Chunk {i+1} error: {str(e)}")
                chunk_summaries.append(chunk[:200])  # Fallback: first 200 chars
        
        # Combine all chunk summaries
        combined = " ".join(chunk_summaries)
        
        # Final summary of summaries
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": f"Create a final 3-4 sentence summary from these segment summaries:\n\n{combined}"
                }],
                temperature=0.1,
                max_tokens=300
            )
            final_summary = response.choices[0].message.content.strip()
        except:
            final_summary = combined[:500]
        
        key_points = self._extract_key_points(combined[:3000])
        
        processing_time = time.time() - start_time
        print(f"Chunked summary complete in {processing_time:.2f}s")
        
        return {
            "title": self._infer_title(final_summary),
            "summary": final_summary,
            "key_points": key_points,
            "decisions": [],
            "processing_time": processing_time
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "type": "Cloud API (FREE - Groq)",
            "provider": "Groq / Meta Llama 3.1",
            "cost": "$0.00 (Free tier)"
        }