from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class MemoryItem(BaseModel):
    """记忆项数据结构"""

    id: str
    content: str
    memory_type: str
    user_id: str
    timestamp: datetime
    importance: float = 0.5
    metadata: dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True


class MemoryConfig(BaseModel):
    storage_path = "./memory_data"
    max_capacity = 100
    importance_threshold = 0.1
    decay_factor = 0.95

    working_memory_capacity = 10
    working_memory_tokens = 2000
    working_memory_ttl_minutes = 120

    perceptual_memory_modalities: list[str] = ["text", "image", "audio", "video"]


class BaseMemory(ABC):
    """
    记忆基类

    定义所有记忆类型的通用接口和行为
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        self.config = config
        self.storage = storage_backend
        self.memory_type = self.__class__.__name__.lower().replace("memory", "")

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        """
        添加记忆项

        Args:
          memory_item: 记忆项对象

        Returns:
          记忆ID
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> list[MemoryItem]:
        """
        检索相关记忆

        Args:
          query: 查询内容
          limit: 返回数量限制
          **kwargs: 其他检索参数

        Returns:
          相关记忆列表
        """
        pass

    @abstractmethod
    def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        更新记忆

        Args:
          memory_id: 记忆ID
          content: 新内容
          importance: 新重要性
          metadata: 新元数据

        Returns:
          是否更新成功
        """
        pass

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        """
        删除记忆

        Args:
          memory_id: 记忆ID

        Returns:
          是否删除成功
        """
        pass

    @abstractmethod
    def has_memory(self, memory_id: str) -> bool:
        """
        检查记忆是否存在

        Args:
          memory_id: 记忆ID

        Returns:
          是否存在
        """
        pass

    @abstractmethod
    def clear(self):
        """清空所有记忆"""
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """
        获取记忆统计信息

        Returns:
          统计信息字典
        """
        pass

    def _generate_id(self) -> str:
        """生成记忆ID"""
        import uuid

        return str(uuid.uuid4())

    def _calculate_importance(
        self, content: str, base_importance: float = 0.5
    ) -> float:
        """
        计算记忆重要性

        Args:
          content: 记忆内容
          base_importance: 基础重要性

        Returns:
          计算后的重要性分数
        """
        importance = base_importance

        if len(content) > 100:
            importance += 1

        important_keywords = ["重要", "关键", "必须", "注意", "警告", "错误"]
        if any(keyword in content for keyword in important_keywords):
            importance += 0.2

        return max(0.0, min(1.0, importance))

    def __str__(self) -> str:
        stats = self.get_stats()
        return f"{self.__class__.__name__}(count={stats.get('count', 0)})"

    def __repr__(self) -> str:
        return self.__str__()
