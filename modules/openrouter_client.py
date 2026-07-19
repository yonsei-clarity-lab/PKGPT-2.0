# openrouter_client.py
"""
OpenRouter API Client for NONMEM optimization
Supports Claude, Gemini, GPT via unified OpenRouter endpoint
"""

import os
import time
from typing import Optional, List
from openai import OpenAI


class OpenRouterClient:
    """Client for interacting with OpenRouter API"""

    MODELS = {
        # Claude
        'claude-sonnet': 'anthropic/claude-sonnet-4.6',
        'claude-opus':   'anthropic/claude-opus-4.8',
        # Gemini
        'gemini-flash':      'google/gemini-2.5-flash',
        'gemini-flash-lite': 'google/gemini-2.5-flash-lite',
        'gemini-pro': 'google/gemini-2.5-pro',   
        # GPT
        'gpt-4.1':   'openai/gpt-4.1',
        'gpt-5.5':   'openai/gpt-5.5',
    }

    def __init__(self, api_key: Optional[str] = None, model_type: str = 'claude-sonnet'):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not provided. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key parameter"
            )
        if model_type not in self.MODELS:
            raise ValueError(f"Invalid model type: {model_type}. Valid: {list(self.MODELS.keys())}")

        self.model_type = model_type
        self.model_name = self.MODELS[model_type]

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        print(f"[OK] Initialized OpenRouter client with model: {self.model_name}")

    def generate(self, prompt: str, retry_attempts: int = 3, timeout: int = 120) -> str:
        for attempt in range(retry_attempts):
            try:
                print(f"  [INFO] Calling OpenRouter (attempt {attempt+1}/{retry_attempts})...")
                start = time.time()

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=8192,
                    temperature=0.7,
                    timeout=timeout,
                )

                elapsed = time.time() - start
                text = response.choices[0].message.content

                if text:
                    print(f"  [OK] Response received ({len(text)} chars, {elapsed:.1f}s)")
                    return text
                else:
                    print(f"[WARNING] Empty response (attempt {attempt+1})")

            except Exception as e:
                elapsed = time.time() - start
                error_msg = str(e)

                if '429' in error_msg or 'rate limit' in error_msg.lower():
                    wait = 30
                    print(f"[WARNING] Rate limit — waiting {wait}s...")
                    time.sleep(wait)
                    continue
                elif '500' in error_msg or '503' in error_msg:
                    print(f"[WARNING] Server error: {error_msg[:100]}")
                else:
                    print(f"[WARNING] Error: {error_msg[:200]}")

                if attempt < retry_attempts - 1:
                    wait = 2 ** attempt
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise Exception(f"Failed after {retry_attempts} attempts: {e}")

        raise Exception("Failed to generate valid response")

    def switch_model(self, model_type: str):
        if model_type not in self.MODELS:
            raise ValueError(f"Invalid model type: {model_type}")
        self.model_type = model_type
        self.model_name = self.MODELS[model_type]
        # client 재사용 (base_url, api_key 동일)
        print(f"[OK] Switched to model: {self.model_name}")

    def get_current_model(self) -> str:
        return self.model_name


class MultiModelOpenRouterClient:
    """기존 MultiModelGeminiClient와 동일한 인터페이스 유지"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        # OpenRouter는 단일 client로 모델만 바꾸면 되므로
        # 기존처럼 model_type별 client dict를 만들되 실제 HTTP client는 공유
        self.clients = {}
        for model_type in OpenRouterClient.MODELS:
            try:
                self.clients[model_type] = OpenRouterClient(self.api_key, model_type)
            except Exception as e:
                print(f"[WARNING] Could not register {model_type}: {e}")

        if not self.clients:
            raise Exception("No models could be registered")

        self.current_model_type = 'claude-sonnet'  # 기본값

    def generate(self, prompt: str, model_type: Optional[str] = None) -> str:
        model_type = model_type or self.current_model_type
        if model_type not in self.clients:
            raise ValueError(f"Model {model_type} not available")
        return self.clients[model_type].generate(prompt)

    def rotate_model(self) -> str:
        available = list(self.clients.keys())
        idx = available.index(self.current_model_type)
        self.current_model_type = available[(idx + 1) % len(available)]
        print(f"[ROTATE] → {self.clients[self.current_model_type].get_current_model()}")
        return self.current_model_type

    def get_available_models(self) -> List[str]:
        return list(self.clients.keys())
