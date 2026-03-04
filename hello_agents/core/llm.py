import os
from typing import Iterator, Literal

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .exceptions import HelloAgentsException

SUPPORTED_PROVIDERS = Literal[
    "openai",
    "deepseek",
    "qwen",
    "modelscope",
    "kimi",
    "zhipu",
    "ollama",
    "vllm",
    "local",
    "auto",
]


class HelloAgentsLLM:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: SUPPORTED_PROVIDERS | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: int | None = None,
        **kwargs,
    ):
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))
        self.kwargs = kwargs

        self.provider = provider or self._auto_detect_provider(api_key, base_url)
        self.api_key, self.base_url = self._resolve_credentials(api_key, base_url)

        if not self.model:
            self.model = self._get_default_model()

        if not all([self.model, self.api_key, self.base_url]):
            raise HelloAgentsException("必须提供 api_key 和 base_url")

        self._client = self._create_client()

    def _auto_detect_provider(self, api_key: str | None, base_url: str | None):
        # 1. 根据环境变量判断
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        elif os.getenv("DEEPSEEK_API_KEY"):
            return "deepseek"
        elif os.getenv("DASHSCOPE_API_KEY"):
            return "qwen"
        elif os.getenv("MODELSCOPE_API_KEY"):
            return "modelscope"
        elif os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY"):
            return "kimi"
        elif os.getenv("ZHIPU_API_KEY") or os.getenv("GLM_API_KEY"):
            return "zhipu"
        elif os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_HOST"):
            return "ollama"
        elif os.getenv("VLLM_API_KEY") or os.getenv("VLLM_HOST"):
            return "vllm"

        # 2. 根据 api 密钥格式判断
        actual_api_key = api_key or os.getenv("LLM_API_KEY") or ""
        if actual_api_key:
            actual_key_lower = actual_api_key.lower()
            if actual_key_lower.startswith("ms-"):
                return "modelscope"
            elif actual_key_lower == "ollama":
                return "ollama"
            elif actual_key_lower == "vllm":
                return "vllm"
            elif actual_key_lower == "local":
                return "local"
            elif actual_api_key.endswith(".") or "." in actual_api_key[-20]:
                return "zhipu"
            elif actual_key_lower.startswith("sk-"):
                pass

        # 3. 根据 base_url 判断
        actual_base_url = base_url or os.getenv("LLM_BASE_URL") or ""
        if actual_base_url:
            base_url_lower = actual_base_url.lower()
            if "api.openai.com" in base_url_lower:
                return "openai"
            elif "api.deepseek.com" in base_url_lower:
                return "deepseek"
            elif "dashscope.aliyuncs.com" in base_url_lower:
                return "qwen"
            elif "api-inference.modelscope.cn" in base_url_lower:
                return "modelscope"
            elif "api.moonshot.cn" in base_url_lower:
                return "kimi"
            elif "open.bigmodel.cn" in base_url_lower:
                return "zhipu"
            elif "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
                # 本地部署检测 - 优先检查特定服务
                if ":11434" in base_url_lower or "ollama" in base_url_lower:
                    return "ollama"
                elif ":8000" in base_url_lower and "vllm" in base_url_lower:
                    return "vllm"
                elif ":8080" in base_url_lower or ":7860" in base_url_lower:
                    return "local"
                else:
                    # 根据API密钥进一步判断
                    if actual_api_key and actual_api_key.lower() == "ollama":
                        return "ollama"
                    elif actual_api_key and actual_api_key.lower() == "vllm":
                        return "vllm"
                    else:
                        return "local"
            elif any(port in base_url_lower for port in [":8080", ":7860", ":5000"]):
                # 常见的本地部署端口
                return "local"

        # 4. 默认返回 auto
        return "auto"

    def _resolve_credentials(
        self, api_key: str | None, base_url: str | None
    ) -> tuple[str, str]:
        """根据provider解析API密钥和base_url"""
        if self.provider == "openai":
            resolved_api_key = (
                api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or ""
            )
            resolved_base_url = (
                base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
            )
            return resolved_api_key, resolved_base_url

        elif self.provider == "deepseek":
            resolved_api_key = (
                api_key
                or os.getenv("DEEPSEEK_API_KEY")
                or os.getenv("LLM_API_KEY")
                or ""
            )
            resolved_base_url = (
                base_url or os.getenv("LLM_BASE_URL") or "https://api.deepseek.com"
            )
            return resolved_api_key, resolved_base_url

        elif self.provider == "qwen":
            resolved_api_key = (
                api_key
                or os.getenv("DASHSCOPE_API_KEY")
                or os.getenv("LLM_API_KEY")
                or ""
            )
            resolved_base_url = (
                base_url
                or os.getenv("LLM_BASE_URL")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            return resolved_api_key, resolved_base_url

        elif self.provider == "modelscope":
            resolved_api_key = (
                api_key
                or os.getenv("MODELSCOPE_API_KEY")
                or os.getenv("LLM_API_KEY")
                or ""
            )
            resolved_base_url = (
                base_url
                or os.getenv("LLM_BASE_URL")
                or "https://api-inference.modelscope.cn/v1/"
            )
            return resolved_api_key, resolved_base_url

        elif self.provider == "kimi":
            resolved_api_key = (
                api_key
                or os.getenv("KIMI_API_KEY")
                or os.getenv("MOONSHOT_API_KEY")
                or os.getenv("LLM_API_KEY")
                or ""
            )
            resolved_base_url = (
                base_url or os.getenv("LLM_BASE_URL") or "https://api.moonshot.cn/v1"
            )
            return resolved_api_key, resolved_base_url

        elif self.provider == "zhipu":
            resolved_api_key = (
                api_key
                or os.getenv("ZHIPU_API_KEY")
                or os.getenv("GLM_API_KEY")
                or os.getenv("LLM_API_KEY")
                or ""
            )
            resolved_base_url = (
                base_url
                or os.getenv("LLM_BASE_URL")
                or "https://open.bigmodel.cn/api/paas/v4"
            )
            return resolved_api_key, resolved_base_url

        elif self.provider == "ollama":
            resolved_api_key = (
                api_key
                or os.getenv("OLLAMA_API_KEY")
                or os.getenv("LLM_API_KEY")
                or "ollama"
            )
            resolved_base_url = (
                base_url
                or os.getenv("OLLAMA_HOST")
                or os.getenv("LLM_BASE_URL")
                or "http://localhost:11434/v1"
            )
            return resolved_api_key, resolved_base_url

        elif self.provider == "vllm":
            resolved_api_key = (
                api_key
                or os.getenv("VLLM_API_KEY")
                or os.getenv("LLM_API_KEY")
                or "vllm"
            )
            resolved_base_url = (
                base_url
                or os.getenv("VLLM_HOST")
                or os.getenv("LLM_BASE_URL")
                or "http://localhost:8000/v1"
            )
            return resolved_api_key, resolved_base_url

        elif self.provider == "local":
            resolved_api_key = api_key or os.getenv("LLM_API_KEY") or "local"
            resolved_base_url = (
                base_url or os.getenv("LLM_BASE_URL") or "http://localhost:8000/v1"
            )
            return resolved_api_key, resolved_base_url

        else:
            # auto或其他情况：使用通用配置，支持任何OpenAI兼容的服务
            resolved_api_key = api_key or os.getenv("LLM_API_KEY") or ""
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or ""
            return resolved_api_key, resolved_base_url

    def _get_default_model(self) -> str:
        """获取默认模型"""
        if self.provider == "openai":
            return "gpt-3.5-turbo"
        elif self.provider == "deepseek":
            return "deepseek-chat"
        elif self.provider == "qwen":
            return "qwen-plus"
        elif self.provider == "modelscope":
            return "Qwen/Qwen2.5-72B-Instruct"
        elif self.provider == "kimi":
            return "moonshot-v1-8k"
        elif self.provider == "zhipu":
            return "glm-4"
        elif self.provider == "ollama":
            return "llama3.2"  # Ollama常用模型
        elif self.provider == "vllm":
            return "meta-llama/Llama-2-7b-chat-hf"  # vLLM常用模型
        elif self.provider == "local":
            return "local-model"  # 本地模型占位符
        else:
            # auto或其他情况：根据base_url智能推断默认模型
            base_url = os.getenv("LLM_BASE_URL", "")
            base_url_lower = base_url.lower()
            if "modelscope" in base_url_lower:
                return "Qwen/Qwen2.5-72B-Instruct"
            elif "deepseek" in base_url_lower:
                return "deepseek-chat"
            elif "dashscope" in base_url_lower:
                return "qwen-plus"
            elif "moonshot" in base_url_lower:
                return "moonshot-v1-8k"
            elif "bigmodel" in base_url_lower:
                return "glm-4"
            elif "ollama" in base_url_lower or ":11434" in base_url_lower:
                return "llama3.2"
            elif ":8000" in base_url_lower or "vllm" in base_url_lower:
                return "meta-llama/Llama-2-7b-chat-hf"
            elif "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
                return "local-model"
            else:
                return "gpt-3.5-turbo"

    def _create_client(self):
        return OpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )

    def think(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None = None,
    ) -> Iterator[str]:
        print(f"🧠 正在调用 {self.model} 模型...")

        try:
            response = self._client.chat.completions.create(
                model=self.model or "",
                messages=messages,
                temperature=temperature or 0,
                max_tokens=self.max_tokens,
                stream=True,
            )

            print("✅ 大语言模型响应成功：")
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    print(content, end="", flush=True)
                    yield content
            print()
        except Exception as e:
            print(f"❌ 调用 LLM API时发生错误: {e}")
            raise HelloAgentsException(f"LLM调用失败： {str(e)}")

    def invoke(self, messages: list[ChatCompletionMessageParam], **kwargs) -> str:
        """
        非流式调用 LLM，返回完整响应。
        适用于不需要流式处理的场景。
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model or "",
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["temperature", "max_tokens"]
                },
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            raise HelloAgentsException(f"LLM调用失败：{str(e)}")

    def stream_invoke(self, messages: list[ChatCompletionMessageParam], **kwargs) -> Iterator[str]
      """
      流式调用 LLM 的别名方法，与 think 方法功能相同。
      保持向后兼容性。
      """
      temperature = kwargs.get('temperature')
      yield from self.think(messages, temperature)
