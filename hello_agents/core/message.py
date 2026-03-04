from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """消息类"""

    content: str
    role: MessageRole
    timestamp: datetime | None = None
    metadata: dict[str, Any] | None = None

    def __init__(self, content: str, role: MessageRole, **kwargs) -> None:
        super().__init__(
            content=self.content,
            role=self.role,
            timestamp=kwargs.get("timestamp", datetime.now()),
            metadata=kwargs.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式（OpenAI api格式）"""
        return {"role": self.role, "content": self.content}

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
