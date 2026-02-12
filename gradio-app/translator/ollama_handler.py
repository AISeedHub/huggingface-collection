import logging
import os
import sys
from collections.abc import Generator
from datetime import datetime

import ollama
import yaml

# Configure logging to go to stderr
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaLLMManager:
    def __init__(self, host: str = "localhost:11434"):
        self.client = ollama.Client(host=host)
        self.current_model = None
        self.AVAILABLE_MODELS = [
            "translategemma:4b",
            "translategemma:12b",
            "translategemma:27b",
        ]
        self.system_message = self.load_system_message()
        # Set default model
        if self.AVAILABLE_MODELS:
            self.current_model = self.AVAILABLE_MODELS[0]

        # Create logs directory
        self.logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(self.logs_dir, exist_ok=True)

    def load_system_message(self):
        try:
            path = os.path.join(os.path.dirname(__file__), "./system.yaml")
            with open(path) as f:
                data = yaml.safe_load(f)
                return data["system_prompt"]
        except Exception:
            return "You are a professional translator."

    def save_analysis_log(self, prompt: str, output: str) -> str:
        """Save analysis log to markdown file

        Args:
            prompt (str): Input prompt
            output (str): Generated output

        Returns:
            str: Path to saved log file
        """
        try:
            # Generate filename: modelname_timestamp.md
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Clean model name for filename (replace : with -)
            model_name = self.current_model.replace(":", "-")
            filename = f"{model_name}_{timestamp}.md"
            filepath = os.path.join(self.logs_dir, filename)

            # Create markdown content
            content = f"""# Analysis Log

**Model:** {self.current_model}
**Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Input Prompt

```
{prompt}
```

---

## Output

{output}
"""

            # Save to file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            return filepath
        except Exception as e:
            logger.error(f"Error saving analysis log: {e}")
            return None

    def load_model(self, model_id: str) -> str:
        """Set the current model for Ollama inference"""
        logger.info(f"Setting Ollama model: {model_id}...")

        # Check if model is available locally
        try:
            available_models = self.get_available_models()
            if model_id not in available_models:
                logger.info(
                    f"Model {model_id} not found locally. Attempting to pull..."
                )
                pull_success = self._pull_model(model_id)
                if not pull_success:
                    return f"Failed to pull model {model_id}"

            self.current_model = model_id
            return f"Successfully set Ollama model to {model_id}"

        except Exception as e:
            if "connection" in str(e).lower():
                return "Error: Could not connect to Ollama server"
            return f"Error setting model {model_id}: {str(e)}"

    def get_available_models(self) -> list:
        """Get list of locally available models"""
        try:
            models_response = self.client.list()
            if hasattr(models_response, "models"):
                return [model.model for model in models_response.models]
            return []
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []

    def is_server_available(self) -> bool:
        """Check if Ollama server is accessible"""
        try:
            self.client.list()
            return True
        except Exception as e:
            logger.error(f"Error checking server availability: {e}")
            return False

    def _pull_model(self, model_id: str) -> bool:
        """Pull a model from Ollama registry"""
        try:
            logger.info(f"Pulling model {model_id}... This may take a while.")

            # Use the pull method with progress tracking
            for progress in self.client.pull(model_id, stream=True):
                if "status" in progress:
                    logger.debug(f"Pull status: {progress['status']}")
                    if "total" in progress and "completed" in progress:
                        percentage = (progress["completed"] / progress["total"]) * 100
                        logger.debug(f"Progress: {percentage:.1f}%")

            logger.info(f"Successfully pulled model {model_id}")
            return True

        except Exception as e:
            if "connection" in str(e).lower():
                logger.error(f"Connection error while pulling model {model_id}")
            else:
                logger.error(f"Error pulling model {model_id}: {e}")
            return False

    def generate_analysis(
        self, prompt: str, source_lang: str = "English", target_lang: str = "Korean"
    ) -> Generator[str, None, None]:
        """Generate translation using Ollama Python library with streaming"""
        if not self.current_model:
            yield "Error: No model loaded"
            return

        # Check if server is available
        if not self.is_server_available():
            yield "Error: Cannot connect to Ollama server"
            return

        # Prepare system prompt with languages
        lang_codes = {
            "Korean": "ko",
            "English": "en",
            "Chinese": "zh",
            "Japanese": "ja",
            "Vietnamese": "vi",
        }

        system_content = self.system_message.format(
            SOURCE_LANG=source_lang,
            SOURCE_CODE=lang_codes.get(source_lang, "auto"),
            TARGET_LANG=target_lang,
            TARGET_CODE=lang_codes.get(target_lang, "auto"),
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

        try:
            # Use the chat method with streaming
            stream = self.client.chat(
                model=self.current_model,
                messages=messages,
                stream=True,
                options={"temperature": 0.1},
            )

            partial_text = ""
            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    partial_text += chunk["message"]["content"]
                    yield partial_text

                if chunk.get("done", False):
                    # Save log after completion
                    log_path = self.save_analysis_log(prompt, partial_text)
                    if log_path:
                        logger.info(f"Analysis log saved to: {log_path}")
                    break

        except Exception as e:
            if "connection" in str(e).lower():
                yield "Error: Could not connect to Ollama server"
            else:
                yield f"Error during generation: {str(e)}"
