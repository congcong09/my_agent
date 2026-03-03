import os

from openai import OpenAI

from hello_agents.core.llm import HelloAgentsLLM


class MyLLM(HelloAgentsLLM):
    """
    一个自定义的 LLM 客户端，通过继承增加了对 model scope 的支持
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        provider: str | None = "auto",
        **kwargs,
    ):
        if provider == "modelscope":
            print("正在使用自定义的 ModelScope provider")
            self.provider = provider

            self.api_key = api_key or os.getenv("MODELSCOPE_API_KEY")
            self.base_url = base_url or "https://api-inference.modelscope.cn/v1/"

            if not self.api_key:
                raise ValueError("ModelScope API key not found.")

            self.model = (
                model or os.getenv("LLM_MODEL_ID") or "Qwen/Qwen2.5-VL-72B-Instruct"
            )

            self.temperature = kwargs.get("temperature", 0.7)
            self.max_tokens = kwargs.get("max_tokens")
            self.timeout = kwargs.get("timeout", 60)

            self._client = OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )
        else:
            super().__init__(
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                **kwargs,
            )
