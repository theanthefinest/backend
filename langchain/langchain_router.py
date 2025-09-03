from langchain.llms.base import LLM
from langchain.schema import Generation, LLMResult
from typing import Any, Dict, List, Optional
import requests
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

class Tourism(LLM):

    api_key: str = ""
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    timeout: int = 200

    @property 
    def _llm_type(self) -> str:
        return "Cambodia Tourism"

    def _call(self, prompt: str, **kwargs: Any) -> str:
        payload = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p)
        }

        try:
            logger.info(f"Calling Cambodia Chatbot with prompt: {prompt[:50]}...")
            response = requests.post(self.api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json().get("response", "")
            logger.info(f"Cambodia Chatbot response: {result[:50]}...")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Cambodia Chatbot: {e}")
            raise Exception(f"Error calling Cambodia Chatbot: {e}")

class Gemini(LLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        self.client = genai.Client(api_key=self.api_key)

    @property
    def _llm_type(self) -> str:
        return "gemini-llm"
    
    def _call(self, prompt: str, **kwargs: Any) -> str:
        try:
            logger.info(f"Calling Gemini with prompt: {prompt[:50]}...")
            response = self.client.models.generate_content(
                model='gemini-pro',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=kwargs.get("temperature", 0.7),
                    max_output_tokens=kwargs.get("max_tokens", 1024),
                )
            )
            result = response.text
            logger.info(f"Gemini response: {result[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Error calling Gemini: {e}")
            raise Exception(f"Error calling Gemini: {e}")

class Router(LLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tourism = Tourism()

        try:
            self.gemini = Gemini()
            self.use_gemini = True
        except ValueError:
            logger.warning("Gemini API key not found. Using Cambodia model for all queries.")
            self.gemini_available = False

        self.cambodia_keywords = [
            'cambodia', 'angkor wat', 'phnom penh', 'siem reap', 'khmer',
            'cambodian', 'tonle sap', 'killing fields', 'tuol sleng',
            'preah vihear', 'battambang', 'sihanoukville', 'kampot', 'kep',
            'cambodge', 'ខ្មែរ', 'tourism cambodia', 'visit cambodia',
            'travel cambodia', 'cambodia travel', 'cambodia tour',
            'cambodia temple', 'cambodia culture', 'cambodia history',
            'cambodia food', 'cambodian cuisine', 'apsara dance',
            'mondulkiri', 'rattanakiri', 'koh rong', 'koh rong samloem',
            'history', 'cambodia history'
        ]

    @property
    def _llm_type(self) -> str:
        return "Router"

    def _is_cambodia_tourism_question(self, query: str) -> bool:
        query_lower = query.lower()
        
        for keyword in self.cambodia_keywords:
            if keyword in query_lower:
                return True

        tourism_words = ['tour', 'travel', 'visit', 'vacation', 'holiday', 'tourism', 'hotel', 'restaurant']
        question_words = ['where', 'what', 'how', 'when', 'which', 'recommend', 'best', 'good']
        
        has_tourism_word = any(word in query_lower for word in tourism_words)
        has_question_word = any(word in query_lower for word in question_words)
        
        if has_tourism_word and has_question_word:
            other_countries = ['thailand', 'vietnam', 'laos', 'myanmar', 'indonesia', 'malaysia', 'singapore']
            no_other_country = not any(country in query_lower for country in other_countries)
            
            if no_other_country:
                return True
        
        return False
    
    def _call(self, prompt: str, **kwargs: Any) -> str:
        if self._is_cambodia_tourism_question(prompt):
            logger.info("Routing to Cambodia Tourism model.")
            return self.tourism._call(prompt, **kwargs)
        elif self.use_gemini:
            logger.info("Routing to Gemini model.")
            return self.gemini._call(prompt, **kwargs)
        else:
            logger.info("Gemini not available. Routing to Cambodia Tourism model.")
            return self.tourism._call(prompt, **kwargs)