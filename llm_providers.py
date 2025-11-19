# llm_providers.py

from abc import ABC, abstractmethod
import time
import re
from typing import Dict, List, Optional, Any

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from transformers import StoppingCriteria, StoppingCriteriaList


# import openai
# import google.generativeai as genai
# from anthropic import Anthropic

class BaseLLM(ABC):
    def __init__(self, config: Dict):
        self.temperature = config.get('temperature', 0.7)
        self.last_request_time = 0
        self.min_delay = 0.5

    def _rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_delay:
            time.sleep(self.min_delay - time_since_last)
        self.last_request_time = time.time()

    @abstractmethod
    def generate(self, messages: List[Dict], max_tokens: int) -> str:
        """Generate a response from the model"""
        pass

class OpenAILLM(BaseLLM):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.client = openai.OpenAI(api_key=config['api_key'])
        self.model_name = config['model_name']

    def generate(self, messages: List[Dict], max_tokens: int) -> str:
        self._rate_limit()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

# in llm_providers.py, update the GeminiLLM class

class GeminiLLM(BaseLLM):
    def __init__(self, config: Dict):
        super().__init__(config)
        genai.configure(api_key=config['api_key'])
        self.client = genai.GenerativeModel(config['model_name'])
        self.generation_config = genai.types.GenerationConfig(
            temperature=self.temperature,
            candidate_count=1,
            stop_sequences=None,
            max_output_tokens=1024,
            top_p=0.8,
            top_k=40,
        )

    def generate(self, messages: List[Dict], max_tokens: int) -> str:
        self._rate_limit()
        try:
            # Convert OpenAI-style messages to Gemini format
            prompt = self._convert_messages(messages)
            safety_settings = {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            }
            
            response = self.client.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=safety_settings
            )
            
            if hasattr(response, 'text'):
                return response.text.strip()
            # Handle blocked response
            return "I cannot provide an answer."
            
        except Exception as e:
            if 'Resource has been exhausted' in str(e):
                # Add longer delay for quota errors
                time.sleep(5)
                raise Exception("Quota exceeded, please wait")
            raise e

    def _convert_messages(self, messages: List[Dict]) -> str:
        prompt = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                prompt += f"Instructions: {content}\n\n"
            elif role == 'user':
                prompt += f"User: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n"
        return prompt.strip()

class ClaudeLLM(BaseLLM):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.client = Anthropic(api_key=config['api_key'])
        self.model_name = config['model_name']

    def generate(self, messages: List[Dict], max_tokens: int) -> str:
        self._rate_limit()
        # Convert OpenAI-style messages to Claude format
        system_prompt = next((m['content'] for m in messages if m['role'] == 'system'), "")
        conversation = [m for m in messages if m['role'] != 'system']
        
        response = self.client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=[{
                'role': 'user' if m['role'] == 'user' else 'assistant',
                'content': m['content']
            } for m in conversation],
            max_tokens=max_tokens,
            temperature=self.temperature
        )
        return response.content[0].text.strip()

class StopAfterSequence(StoppingCriteria):
    """Stops generation when any of the provided sequences is found."""
    def __init__(self, stop_sequences: List[str], tokenizer: Any, device: torch.device):
        super().__init__()
        self.tokenizer = tokenizer
        
        # Tokenize stop sequences and move to the provided device
        self.stop_ids = [
            tokenizer.encode(seq, add_special_tokens=False, return_tensors='pt')[0].to(device)
            for seq in stop_sequences
        ]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if any stop sequence is present in the generated output (input_ids)
        for stop_id in self.stop_ids:
            # Check if the generated sequence ends with the stop sequence
            # Use .tolist() for safe comparison of tensor elements
            if input_ids[0][-len(stop_id):].tolist() == stop_id.tolist():
                return True
        return False
    
class LocalHFLLM(BaseLLM):
    """Generic local Hugging Face LLM with support for different model families."""
    def __init__(self, config: Dict):
        super().__init__(config)
        model_name = config.get("model_name", "meta-llama/Llama-2-7b-chat-hf")  # default LLaMA
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Detect model family for potential fallback handling
        model_lower = model_name.lower()
        if "llama" in model_lower:
            self.model_family = "llama"
        elif "qwen" in model_lower:
            self.model_family = "qwen"
        elif "mistral" in model_lower:
            self.model_family = "mistral"
        elif "gemma" in model_lower:
            self.model_family = "gemma"
        else:
            self.model_family = "generic"

        if config.get("load_in_4bit", False):
            # Configuration for 4-bit loading
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4", # Recommended for better performance
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            bnb_config = None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb_config
        )

    # llm_agent.py (inside LLMAgent or LLMWrapper)

    def generate(self, messages: List[Dict], max_tokens: int) -> str:

        prompt = self._convert_messages(messages)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=self.temperature
        )

        text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # For Qwen models, strip <think> tags after generation instead of using stopping criteria
        if self.model_family == "qwen":
            # Remove complete think blocks
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            # Remove incomplete think blocks (if generation stopped mid-think)
            text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
            text = text.strip()

        return text.strip()

    def _convert_messages(self, messages: List[Dict]) -> str:
        """Convert messages to proper chat format using tokenizer's chat template."""
        # Try to use the tokenizer's built-in chat template if available
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            try:
                # Use the model's native chat template
                # add_generation_prompt=True adds the assistant prefix for generation
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return prompt
            except Exception as e:
                # Special handling for Gemma: merge system message into first user message
                if "System role not supported" in str(e):
                    try:
                        converted_messages = self._convert_system_to_user(messages)
                        prompt = self.tokenizer.apply_chat_template(
                            converted_messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        return prompt
                    except Exception as e2:
                        print(f"Warning: Failed to use chat template after Gemma conversion, falling back to manual formatting: {e2}")
                else:
                    # For other exceptions, print warning and fall through to manual formatting
                    print(f"Warning: Failed to use chat template, falling back to manual formatting: {e}")

        # Fallback: Manual formatting for models without chat templates
        prompt = ""
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        return prompt.strip()

    def _convert_system_to_user(self, messages: List[Dict]) -> List[Dict]:
        """Convert system message to user message for models that don't support system role (e.g., Gemma)."""
        converted = []
        system_content = None

        for msg in messages:
            if msg["role"] == "system":
                # Store system message to prepend to first user message
                system_content = msg["content"]
            elif msg["role"] == "user":
                if system_content:
                    # Prepend system instructions to first user message
                    converted.append({
                        "role": "user",
                        "content": f"{system_content}\n\n{msg['content']}"
                    })
                    system_content = None  # Only prepend once
                else:
                    converted.append(msg)
            else:
                # Keep assistant messages as-is
                converted.append(msg)

        return converted

# Factory function to create LLM instances
def create_llm(config: Dict) -> BaseLLM:
    llm_type = config['type'].lower()
    if llm_type == 'openai':
        return OpenAILLM(config)
    elif llm_type == 'gemini':
        return GeminiLLM(config)
    elif llm_type == 'claude':
        return ClaudeLLM(config)
    elif llm_type == 'local_hf':
        return LocalHFLLM(config)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")