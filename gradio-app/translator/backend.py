# from minimal_agentic_ollama_handler import AgenticOllamaHandler
from ollama_handler import OllamaLLMManager


class Backend:
    def __init__(self):
        self.llm_manager = OllamaLLMManager()

    @property
    def available_models(self):
        return self.llm_manager.AVAILABLE_MODELS

    def load_model(self, model_id, progress_callback=None):
        # Update all handlers
        msg = self.llm_manager.load_model(model_id, progress_callback)
        return msg

    def generate_analysis(self, prompt, source_lang="English", target_lang="Korean"):
        """Generate translation using Ollama"""
        return self.llm_manager.generate_analysis(prompt, source_lang, target_lang)


# Singleton instance
backend = Backend()
