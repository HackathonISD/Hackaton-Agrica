import os
import re
import pandas as pd
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from typing import Any, Optional, List, Dict
from datetime import datetime


class LLmClient:
    """A client class for interacting with OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_TOKEN env variable.
            base_url: Base URL for the API. If None, reads from PROVIDER_URL env variable.
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_TOKEN", "")
        self.base_url = base_url or os.getenv("PROVIDER_URL", "")
        self.client = self._initialize_client()
        self.model = "openai-gpt-5.2"

    def _initialize_client(self) -> OpenAI:
        """Initialize and return the OpenAI client."""
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_prompt(self, prompt_template: str, **kwargs: Any) -> str:
        """
        Generate a formatted prompt from a template.

        Args:
            prompt_template: Template string with placeholders.
            **kwargs: Values to fill the template placeholders.

        Returns:
            Formatted prompt string.
        """
        return prompt_template.format(**kwargs)

    def generate_response(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.5,
    ) -> str:
        """
        Generate a response from OpenAI.

        Args:
            user_prompt: The user's input prompt.
            system_prompt: The system prompt to set context.
            temperature: Controls randomness (0-2). Default is 0.5.

        Returns:
            The generated response content.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )

            if response.choices[0].message.content is not None:
                return response.choices[0].message.content
            else:
                return "No content in response."

        except OpenAIError as e:
            print(f"An error occurred: {e}")
            return "An error occurred while generating the response."

    def generate_response_with_history(
        self,
        user_prompt: str,
        history: List[Dict[str, str]],
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.5,
    ) -> str:
        """
        Generate a response from OpenAI with conversation history.

        Args:
            user_prompt: The user's current input prompt.
            history: List of previous messages [{"role": "user"|"assistant", "content": "..."}].
            system_prompt: The system prompt to set context.
            temperature: Controls randomness (0-2). Default is 0.5.

        Returns:
            The generated response content.
        """
        try:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": user_prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )

            if response.choices[0].message.content is not None:
                return response.choices[0].message.content
            else:
                return "No content in response."

        except OpenAIError as e:
            print(f"An error occurred: {e}")
            return "An error occurred while generating the response."
