import os

from pydantic import BaseModel


class Config(BaseModel):
    """HelloAgents配置类"""

    # LLM配置
    default_model: str = "gpt-4.5-turbo"
    default_provider: str = "openai"
    temperature: float = 0.7
    max_tokens: int | None = None

    # 系统配置
    debug: bool = False
    log_level: str = "INFO"

    # 其他配置
    max_history_length: int | None = 100

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置"""
        env_max_tokens = os.getenv("MAX_TOKENS")
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", 0.7)),
            max_tokens=int(env_max_tokens) if env_max_tokens else None,
        )

    def to_dict(self):
        """转换为字典"""
        return self.to_dict()
