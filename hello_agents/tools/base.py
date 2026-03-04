from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolParameter(BaseModel):
    """工具参数定义"""

    name: str
    type: str
    description: str
    required: bool
    default: Any = None


class Tool(ABC):
    """工具基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: dict[str, Any]) -> str:
        """执行工具"""
        pass

    @abstractmethod
    def get_parameter(self) -> list[ToolParameter]:
        """获取工具参数定义"""
        pass

    def validate_parameters(self, parameters: dict[str, Any]) -> bool:
        """验证参数"""
        required_params = [p.name for p in self.get_parameter() if p.required]
        return all(param in parameters for param in required_params)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [param.model_dump() for param in self.get_parameter()],
        }

    def __str__(self) -> str:
        return f"Tool(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()
