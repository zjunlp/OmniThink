import random
import threading
import time
import dspy
import os
from typing import Optional, Literal, Any
from dashscope import Generation
import backoff
from dsp import ERRORS, backoff_hdlr, giveup_hdlr
import requests
# This code is originally sourced from Repository STORM
# URL: [https://github.com/stanford-oval/storm]


class OpenAIModel_dashscope(dspy.OpenAI):
    """A wrapper class for dspy.OpenAI."""

    def __init__(
            self,
            model: str = "gpt-4",
            max_tokens: int = 2000,
            api_key: Optional[str] = None,
            **kwargs
    ):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.model = model
        self._token_usage_lock = threading.Lock()
        self.max_tokens = max_tokens
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, response):
        """Log the total tokens from the OpenAI API response."""
        usage_data = response.get('usage')
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get('input_tokens', 0)
                self.completion_tokens += usage_data.get('output_tokens', 0)

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        usage = {
            self.kwargs.get('model') or self.kwargs.get('engine'):
                {'prompt_tokens': self.prompt_tokens, 'completion_tokens': self.completion_tokens}
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0

        return usage

    def __call__(
            self,
            prompt: str,
            only_completed: bool = True,
            return_sorted: bool = False,
            **kwargs,
    ) -> list[dict[str, Any]]:
        """Copied from dspy/dsp/modules/gpt3.py with the addition of tracking token usage."""

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        CALL_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
        DASHSCOPE_KEY = os.getenv('DASHSCOPE_KEY')
        HEADERS = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {DASHSCOPE_KEY}"
        }

        kwargs = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=self.max_tokens,
            stream=False,
        )
        import requests
        max_try = 3
        for i in range(max_try):
            try:
                ret = requests.post(CALL_URL, json=kwargs,
                                    headers=HEADERS, timeout=180)
                if ret.status_code != 200:
                    raise Exception(f"http status_code: {ret.status_code}\n{ret.content}")
                ret_json = ret.json()
                for output in ret_json['choices']:
                    if output['finish_reason'] not in ['stop', 'function_call']:
                        raise Exception(f'openai finish with error...\n{ret_json}')
                return [ret_json['choices'][0]['message']['content']]
            except Exception as e:
                print(f"请求失败: {e}. 尝试重新请求...")    
                time.sleep(1)





class QwenModel(dspy.OpenAI):
    """A wrapper class for dspy.OpenAI."""

    def __init__(
            self,
            model: str = "qwen-max-allinone",
            api_key: Optional[str] = None,
            **kwargs
    ):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.model = model
        self.api_key = api_key
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, response):
        """Log the total tokens from the OpenAI API response."""
        usage_data = response.get('usage')
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get('input_tokens', 0)
                self.completion_tokens += usage_data.get('output_tokens', 0)

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        usage = {
            self.kwargs.get('model') or self.kwargs.get('engine'):
                {'prompt_tokens': self.prompt_tokens, 'completion_tokens': self.completion_tokens}
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0

        return usage

    def __call__(
            self,
            prompt: str,
            only_completed: bool = True,
            return_sorted: bool = False,
            **kwargs,
    ) -> list[dict[str, Any]]:
        """Copied from dspy/dsp/modules/gpt3.py with the addition of tracking token usage."""

        assert only_completed, "for now"
        assert return_sorted is False, "for now"



        messages = [{'role': 'user', 'content': prompt}]
        max_retries = 3
        attempt = 0
        while attempt < max_retries:
            try:
                response = Generation.call(
                    model=self.model, 
                    messages=messages,
                    result_format='message',
                ) 
                choices = response["output"]["choices"]
                break

            except Exception as e:
                delay = random.uniform(0, 10)
                time.sleep(delay)
                attempt += 1

        self.log_usage(response)

        completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [c['message']['content'] for c in choices]

        return completions

class DeepSeekModel(dspy.OpenAI):
    """A wrapper class for DeepSeek API, compatible with dspy.OpenAI."""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        api_base: str = "https://api.deepseek.com",
        **kwargs,
    ):
        super().__init__(model=model, api_key=api_key, api_base=api_base, **kwargs)
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_base = api_base
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key must be provided either as an argument or as an environment variable DEEPSEEK_API_KEY"
            )
        self.transforms = kwargs.get("transforms", None)

    def log_usage(self, response):
        """Log the total tokens from the DeepSeek API response."""
        usage_data = response.get("usage")
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get("prompt_tokens", 0)
                self.completion_tokens += usage_data.get("completion_tokens", 0)

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        usage = {
            self.model: {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0
        return usage

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=1000,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def _create_completion(self, prompt: str, **kwargs):
        """Create a completion using the DeepSeek API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        response = requests.post(
            f"{self.api_base}/v1/chat/completions", headers=headers, json=data
        )
        response.raise_for_status()
        return response.json()

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Call the DeepSeek API to generate completions."""
        assert only_completed, "for now"
        assert return_sorted is False, "for now"
        if self.transforms is not None:
            kwargs.update(dict(transforms=self.transforms))
        response = self._create_completion(prompt, **kwargs)

        # Log the token usage from the DeepSeek API response.
        self.log_usage(response)

        if not "choices" in response:
            print("-------------------DeepSeekModel start-----------------")
            print("prompt:",prompt)
            print("response:",response)
            print("kwargs", kwargs)
            print("-------------------DeepSeekModel end-----------------")
        choices = response["choices"]
        completions = [choice["message"]["content"] for choice in choices]

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
        }
        self.history.append(history)

        return completions