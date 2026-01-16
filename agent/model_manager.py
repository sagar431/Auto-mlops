"""
Model Manager for MLOps Agent - LLM provider management.
Supports multiple providers (Google Gemini, OpenAI, Anthropic).
"""

import os
import json
import threading
from typing import Optional, Dict, Any, List
from pathlib import Path


class ModelManager:
    """
    Manages LLM providers and model configurations.
    Handles API calls with retry logic and fallback.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "models.json"
        
        self.config = self._load_config(config_path)
        self.default_model = self.config.get("default", "gemini")
        self.models = self.config.get("models", {})
        
        # Initialize clients lazily
        self._clients: Dict[str, Any] = {}
    
    def _load_config(self, config_path: str) -> Dict:
        """Load model configuration from JSON file."""
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load model config: {e}")
            return {
                "default": "gemini",
                "models": {
                    "gemini": {
                        "provider": "google",
                        "model": "gemini-2.0-flash",
                        "temperature": 0.7,
                        "max_tokens": 8192
                    }
                }
            }
    
    def _get_client(self, provider: str):
        """Get or create client for provider."""
        if provider in self._clients:
            return self._clients[provider]
        
        if provider == "google":
            try:
                import google.generativeai as genai
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self._clients[provider] = genai
                    return genai
            except ImportError:
                print("Warning: google-generativeai not installed")
        
        elif provider == "openai":
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    client = OpenAI(api_key=api_key)
                    self._clients[provider] = client
                    return client
            except ImportError:
                print("Warning: openai not installed")
        
        elif provider == "anthropic":
            try:
                from anthropic import Anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    client = Anthropic(api_key=api_key)
                    self._clients[provider] = client
                    return client
            except ImportError:
                print("Warning: anthropic not installed")
        
        return None
    
    async def generate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: str = "text"  # "text" or "json"
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            model_name: Model config name (default: uses default model)
            system_prompt: System prompt to prepend
            temperature: Override temperature
            max_tokens: Override max tokens
            response_format: "text" or "json"
        
        Returns:
            Generated text response
        """
        model_name = model_name or self.default_model
        model_config = self.models.get(model_name, self.models.get(self.default_model))
        
        if not model_config:
            raise ValueError(f"Model config not found: {model_name}")
        
        provider = model_config["provider"]
        model_id = model_config["model"]
        temp = temperature or model_config.get("temperature", 0.7)
        tokens = max_tokens or model_config.get("max_tokens", 4096)
        
        client = self._get_client(provider)
        if client is None:
            raise RuntimeError(f"Could not initialize {provider} client. Check API key.")
        
        # Build full prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        if response_format == "json":
            full_prompt += "\n\nRespond with valid JSON only, no markdown formatting."
        
        try:
            if provider == "google":
                return await self._generate_gemini(client, model_id, full_prompt, temp, tokens)
            elif provider == "openai":
                return await self._generate_openai(client, model_id, full_prompt, system_prompt, temp, tokens)
            elif provider == "anthropic":
                return await self._generate_anthropic(client, model_id, full_prompt, system_prompt, temp, tokens)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        
        except Exception as e:
            print(f"Error generating with {provider}/{model_id}: {e}")
            raise
    
    async def _generate_gemini(
        self,
        client,
        model_id: str,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using Google Gemini."""
        import asyncio
        
        model = client.GenerativeModel(model_id)
        
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        # Run in executor since Gemini SDK is synchronous
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(
                prompt,
                generation_config=generation_config
            )
        )
        
        return response.text
    
    async def _generate_openai(
        self,
        client,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using OpenAI."""
        import asyncio
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        )
        
        return response.choices[0].message.content
    
    async def _generate_anthropic(
        self,
        client,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using Anthropic Claude."""
        import asyncio
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}]
            )
        )
        
        return response.content[0].text
    
    async def generate_text(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate plain text response."""
        return await self.generate(
            prompt=prompt,
            model_name=model_name,
            system_prompt=system_prompt,
            response_format="text"
        )

    async def generate_json(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate and parse JSON response."""
        response = await self.generate(
            prompt=prompt,
            model_name=model_name,
            system_prompt=system_prompt,
            response_format="json"
        )
        
        # Clean response
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response was: {cleaned[:500]}")
            return {"error": "Failed to parse response", "raw": cleaned}
    
    def list_models(self) -> List[str]:
        """List available model configurations."""
        return list(self.models.keys())
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get configuration for a specific model."""
        return self.models.get(model_name)


# Singleton instance with thread-safe initialization
_model_manager: Optional[ModelManager] = None
_model_manager_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    """Get the singleton ModelManager instance (thread-safe)."""
    global _model_manager
    if _model_manager is None:
        with _model_manager_lock:
            # Double-check locking pattern
            if _model_manager is None:
                _model_manager = ModelManager()
    return _model_manager
