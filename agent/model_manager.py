"""
Model Manager for MLOps Agent - LLM provider management.
Supports multiple providers (Google Gemini, OpenAI, Anthropic).
Includes fallback chain support (Gemini → OpenAI → Gemini Flash).
"""

import json
import os
from pathlib import Path
from typing import Any

from observability import get_logger

logger = get_logger("agent.model_manager")

# Default fallback chain: Gemini → OpenAI → Gemini Flash
DEFAULT_FALLBACK_CHAIN = ["gemini", "gpt4", "gemini_flash"]


class ModelManager:
    """
    Manages LLM providers and model configurations.
    Handles API calls with retry logic and fallback chain.

    Fallback Chain:
        When a model fails (API error, rate limit, unavailable), the manager
        automatically tries the next model in the fallback chain:
        1. Gemini (primary)
        2. OpenAI GPT-4 (first fallback)
        3. Gemini Flash (final fallback)
    """

    def __init__(self, config_path: str | None = None):
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "models.json"

        self.config = self._load_config(config_path)
        self.default_model = self.config.get("default", "gemini")
        self.models = self.config.get("models", {})
        self.fallback_chain = self.config.get("fallback_chain", DEFAULT_FALLBACK_CHAIN)

        # Initialize clients lazily
        self._clients: dict[str, Any] = {}

    def _load_config(self, config_path: str) -> dict:
        """Load model configuration from JSON file."""
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(
                "Could not load model config, using defaults", error=e, config_path=str(config_path)
            )
            return {
                "default": "gemini",
                "models": {
                    "gemini": {
                        "provider": "google",
                        "model": "gemini-2.0-flash",
                        "temperature": 0.7,
                        "max_tokens": 8192,
                    }
                },
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
                logger.warning("google-generativeai not installed", provider=provider)

        elif provider == "openai":
            try:
                from openai import OpenAI

                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    client = OpenAI(api_key=api_key)
                    self._clients[provider] = client
                    return client
            except ImportError:
                logger.warning("openai not installed", provider=provider)

        elif provider == "anthropic":
            try:
                from anthropic import Anthropic

                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    client = Anthropic(api_key=api_key)
                    self._clients[provider] = client
                    return client
            except ImportError:
                logger.warning("anthropic not installed", provider=provider)

        return None

    async def generate(
        self,
        prompt: str,
        model_name: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str = "text",  # "text" or "json"
        use_fallback: bool = True,
    ) -> str:
        """
        Generate a response from the LLM with fallback chain support.

        Args:
            prompt: User prompt
            model_name: Model config name (default: uses default model)
            system_prompt: System prompt to prepend
            temperature: Override temperature
            max_tokens: Override max tokens
            response_format: "text" or "json"
            use_fallback: If True, try fallback models on failure (default: True)

        Returns:
            Generated text response

        Raises:
            RuntimeError: If all models in the fallback chain fail
        """
        model_name = model_name or self.default_model

        # Build the chain of models to try
        if use_fallback:
            models_to_try = self._build_fallback_chain(model_name)
        else:
            models_to_try = [model_name]

        last_error: Exception | None = None
        errors_by_model: dict[str, str] = {}

        for current_model in models_to_try:
            model_config = self.models.get(current_model)
            if not model_config:
                logger.warning("Model config not found, skipping", model_name=current_model)
                errors_by_model[current_model] = "Model config not found"
                continue

            provider = model_config["provider"]
            model_id = model_config["model"]
            temp = temperature or model_config.get("temperature", 0.7)
            tokens = max_tokens or model_config.get("max_tokens", 4096)

            client = self._get_client(provider)
            if client is None:
                logger.warning(
                    "Could not initialize client, trying next model",
                    provider=provider,
                    model_name=current_model,
                )
                errors_by_model[current_model] = f"Could not initialize {provider} client"
                continue

            # Build full prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            if response_format == "json":
                full_prompt += "\n\nRespond with valid JSON only, no markdown formatting."

            try:
                if provider == "google":
                    result = await self._generate_gemini(
                        client, model_id, full_prompt, temp, tokens
                    )
                elif provider == "openai":
                    result = await self._generate_openai(
                        client, model_id, full_prompt, system_prompt, temp, tokens
                    )
                elif provider == "anthropic":
                    result = await self._generate_anthropic(
                        client, model_id, full_prompt, system_prompt, temp, tokens
                    )
                else:
                    errors_by_model[current_model] = f"Unknown provider: {provider}"
                    continue

                # Success - log if we used a fallback
                if current_model != models_to_try[0]:
                    logger.info(
                        "Fallback successful",
                        original_model=models_to_try[0],
                        fallback_model=current_model,
                    )
                return result

            except Exception as e:
                last_error = e
                error_msg = str(e)
                errors_by_model[current_model] = error_msg
                logger.warning(
                    "Model generation failed, trying fallback",
                    model_name=current_model,
                    provider=provider,
                    error=error_msg,
                    remaining_fallbacks=len(models_to_try) - models_to_try.index(current_model) - 1,
                )

        # All models failed
        logger.error(
            "All models in fallback chain failed",
            errors=errors_by_model,
            fallback_chain=models_to_try,
        )
        raise RuntimeError(
            f"All models in fallback chain failed. Errors: {errors_by_model}"
        ) from last_error

    def _build_fallback_chain(self, starting_model: str) -> list[str]:
        """
        Build the ordered list of models to try, starting from the specified model.

        If the starting model is in the fallback chain, start from that position.
        Otherwise, prepend it to the chain.

        Args:
            starting_model: The initial model to try

        Returns:
            Ordered list of model names to try
        """
        if starting_model in self.fallback_chain:
            # Start from this model's position in the chain
            start_idx = self.fallback_chain.index(starting_model)
            return self.fallback_chain[start_idx:]
        else:
            # Model not in chain - try it first, then the full chain
            return [starting_model] + self.fallback_chain

    async def _generate_gemini(
        self, client, model_id: str, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        """Generate using Google Gemini."""
        import asyncio

        model = client.GenerativeModel(model_id)

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # Run in executor since Gemini SDK is synchronous
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, lambda: model.generate_content(prompt, generation_config=generation_config)
        )

        return response.text

    async def _generate_openai(
        self,
        client,
        model_id: str,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate using OpenAI."""
        import asyncio

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model_id, messages=messages, temperature=temperature, max_tokens=max_tokens
            ),
        )

        return response.choices[0].message.content

    async def _generate_anthropic(
        self,
        client,
        model_id: str,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate using Anthropic Claude."""
        import asyncio

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
            ),
        )

        return response.content[0].text

    async def generate_text(
        self,
        prompt: str,
        model_name: str | None = None,
        system_prompt: str | None = None,
        use_fallback: bool = True,
    ) -> str:
        """Generate plain text response with fallback chain support."""
        return await self.generate(
            prompt=prompt,
            model_name=model_name,
            system_prompt=system_prompt,
            response_format="text",
            use_fallback=use_fallback,
        )

    async def generate_json(
        self,
        prompt: str,
        model_name: str | None = None,
        system_prompt: str | None = None,
        use_fallback: bool = True,
    ) -> dict[str, Any]:
        """Generate and parse JSON response with fallback chain support."""
        response = await self.generate(
            prompt=prompt,
            model_name=model_name,
            system_prompt=system_prompt,
            response_format="json",
            use_fallback=use_fallback,
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
            logger.error("Failed to parse JSON response", error=e, response_preview=cleaned[:500])
            return {"error": "Failed to parse response", "raw": cleaned}

    def list_models(self) -> list[str]:
        """List available model configurations."""
        return list(self.models.keys())

    def get_model_info(self, model_name: str) -> dict | None:
        """Get configuration for a specific model."""
        return self.models.get(model_name)


# Singleton instance
_model_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    """Get the singleton ModelManager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
