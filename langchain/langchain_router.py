from langchain.llms.base import LLM
from typing import Any, List
import requests
import os
import logging
import google.generativeai as genai  
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Tourism(LLM):
    api_url: str = "http://127.0.0.1:8000/chat"  
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    timeout: int = 200

    @property
    def _llm_type(self) -> str:
        return "cambodia-tourism"

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
            result = response.json().get("response", "").strip()
            logger.info(f"Cambodia Chatbot response: {result[:50]}...")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Cambodia Chatbot: {e}")
            raise Exception(f"Error calling Cambodia Chatbot: {e}")

    @property
    def _identifying_params(self) -> dict:
        return {
            "api_url": self.api_url,
            "model": "cambodia-custom"
        }

class Gemini(LLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        genai.configure(api_key=self.api_key)
        self.model_name = "gemini-pro"

    @property
    def _llm_type(self) -> str:
        return "gemini-llm"

    def _call(self, prompt: str, **kwargs: Any) -> str:
        try:
            logger.info(f"Calling Gemini with prompt: {prompt[:50]}...")
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                contents=prompt,
                generation_config={
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_output_tokens": kwargs.get("max_tokens", 1024),
                    "top_p": kwargs.get("top_p", 0.9),
                }
            )
            result = response.text.strip()
            logger.info(f"Gemini response: {result[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Error calling Gemini: {e}")
            raise Exception(f"Error calling Gemini: {e}")

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model_name}


class Router(LLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tourism = Tourism()

        self.use_gemini = False
        self.gemini = None

        try:
            self.gemini = Gemini()
            self.use_gemini = True
            logger.info("Gemini model initialized successfully.")
        except ValueError as e:
            logger.warning(f"Gemini API key not found: {e}. Using Cambodia model for all queries.")

        self.cambodia_keywords = [
            'cambodia', 'angkor wat', 'phnom penh', 'siem reap', 'khmer',
            'cambodian', 'tonle sap', 'killing fields', 'tuol sleng',
            'preah vihear', 'battambang', 'sihanoukville', 'kampot', 'kep',
            'cambodge', 'ខ្មែរ', 'tourism cambodia', 'visit cambodia',
            'travel cambodia', 'cambodia travel', 'cambodia tour',
            'cambodia temple', 'cambodia culture', 'cambodia history',
            'cambodia food', 'cambodian cuisine', 'apsara dance',
            'mondulkiri', 'rattanakiri', 'koh rong', 'koh rong samloem',
            'history', 'cambodia history', 'visa to cambodia'
        ]

    @property
    def _llm_type(self) -> str:
        return "router-llm"

    def _is_cambodia_tourism_question(self, query: str) -> bool:
        query_lower = query.lower()

        for keyword in self.cambodia_keywords:
            if keyword in query_lower:
                return True

        tourism_words = ['tour', 'travel', 'visit', 'vacation', 'holiday', 'tourism', 'hotel', 'restaurant', 'food', 'cuisine', 'eat', 'stay']
        question_words = ['where', 'what', 'how', 'when', 'which', 'recommend', 'best', 'good', 'suggest']

        has_tourism_word = any(word in query_lower for word in tourism_words)
        has_question_word = any(word in query_lower for word in question_words)

        other_countries = ['thailand', 'vietnam', 'laos', 'myanmar', 'indonesia', 'malaysia', 'singapore', 'philippines']
        no_other_country = not any(country in query_lower for country in other_countries)

        if has_tourism_word and has_question_word and no_other_country:
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

    @property
    def _identifying_params(self) -> dict:
        return {
            "use_gemini": self.use_gemini,
            "cambodia_keywords_count": len(self.cambodia_keywords)
        }