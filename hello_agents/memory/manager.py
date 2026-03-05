import logging
import uuid
from datetime import datetime
from typing import Any

from .base import MemoryConfig, MemoryItem
from .types.episodic import EpisodicMemory
from .types.perceptual import PerceptualMemory
from .types.semantic import SemanticMemory
from .types.working import WorkingMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(
        self,
        config: MemoryConfig | None = None,
        user_id="default_user",
        enable_working: bool = True,
        enable_episodic: bool = True,
        enable_semantic: bool = True,
        enable_perceptual: bool = False,
    ):
        self.config = config or MemoryConfig()
        self.user_id = user_id

        self.memory_types = {}
        if enable_working:
            self.memory_types["working"] = WorkingMemory(self.config)

        if enable_episodic:
            self.memory_types["episodic"] = EpisodicMemory(self.config)

        if enable_semantic:
            self.memory_types["semantic"] = SemanticMemory(self.config)

        if enable_perceptual:
            self.memory_types["perceptual"] = PerceptualMemory(self.config)

        logger.info(
            f"MemoryManager初始化完成，启用记忆类型：{list(self.memory_types.keys())}"
        )

    def add_memory(
        self,
        content: str,
        memory_type: str = "working",
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
        auto_classify: bool = True,
    ):
        """
        添加记忆

        Args:
          content: 记忆内容
          memory_type: 记忆类型
          importance: 重要性分数（0 - 1）
          metadata: 元数据
          auto_classify: 是否自动分类到合适的记忆类型

        Returns:
          记忆ID
        """
        if auto_classify:
            memory_type = self._classify_memory_type(content, metadata)

        if importance is None:
            importance = self._calculate_importance(content, metadata)

        memory_item = MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            user_id=self.user_id,
            timestamp=datetime.now(),
            importance=importance,
            metadata=metadata or {},
        )

        if memory_type in self.memory_types:
            memory_id = self.memory_types[memory_type].add(memory_item)
            logger.debug(f"添加记忆到 {memory_type}: {memory_id}")
            return memory_id
        else:
            raise ValueError(f"不支持的记忆类型{memory_type}")

    def retrieve_memories(
        self,
        query: str,
        memory_types: list[str] | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
        time_range: tuple | None = None,
    ) -> list[MemoryItem]:
        """
        检索记忆

        Args:
          query: 查询内容
          memory_types: 要检索的记忆类型列表
          limit: 返回数量限制
          min_importance: 最小重要性阈值
          time_range: 时间范围（start_time, end_time）

        Returns:
          检索到的记忆列表
        """
        if memory_types is None:
            memory_types = list(self.memory_types.keys())

        all_results = []
        per_type_limit = max(1, limit // len(memory_types))

        for memory_type in memory_types:
            if memory_type in self.memory_types:
                memory_instance = self.memory_types[memory_type]
                try:
                    # 使用每个记忆类自己的检索方法
                    type_results = memory_instance.retrieve(
                        query=query,
                        limit=per_type_limit,
                        min_importance=min_importance,
                        user_id=self.user_id,
                    )
                    all_results.extend(type_results)
                except Exception as e:
                    logger.warning(f"检索 {memory_type} 记忆时出错：{e}")
                    continue

        all_results.sort(key=lambda x: x.importance, reverse=True)
        return all_results[:limit]

    def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ):
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

        for memory_instance in self.memory_types.values():
            if memory_instance.has_memory(memory_id):
                return memory_instance.update(memory_id, content, importance, metadata)

        logger.warning(f"未找到记忆：{memory_id}")
        return False

    def remove_memory(self, memory_id: str) -> bool:
        """
        删除记忆
        """
        for memory_instance in self.memory_types.values():
            if memory_instance.has_memory(memory_id):
                return memory_instance.remove(memory_id)

        logger.warning(f"未找到记忆：{memory_id}")
        return False

    def forget_memories(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30,
    ) -> int:
        """
        记忆遗忘机制

        Args:
          strategy: 遗忘策略('importance_based', 'time_based', 'capacity_based')
          threshold: 遗忘阈值
          max_age_days: 最大保存天数

        Returns:
          遗忘的记忆数量
        """
        total_forgotten = 0

        for memory_instance in self.memory_types.values():
            if hasattr(memory_instance, "forget"):
                forgotten = memory_instance.forget(strategy, threshold, max_age_days)
                total_forgotten += forgotten

        logger.info(f"记忆遗忘完成：{total_forgotten} 条记忆")
        return total_forgotten

    def consolidate_memories(
        self,
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.7,
    ):
        """
        记忆整合 - 将重要的短期记忆转换为长期记忆

        Args:
          from_type: 源记忆类型
          to_type: 目标记忆类型
          importance_threshold: 重要性阈值

        Returns:
          整合的记忆数量
        """
        if from_type not in self.memory_types or to_type not in self.memory_types:
            logger.warning(f"记忆类型不存在： {from_type} -> {to_type}")
            return 0

        source_memory = self.memory_types[from_type]
        target_memory = self.memory_types[to_type]

        all_memories = source_memory.get_all()
        candidates = [m for m in all_memories if m.importance >= importance_threshold]

        consolidated_count = 0

        for memory in candidates:
            if source_memory.remove(memory.id):
                memory.memory_type = to_type
                memory.importance *= 1.1
                target_memory.add(memory)
                consolidated_count += 1

        logger.info(
            f"记忆整合完成：{consolidated_count} 条记忆从 {from_type} 转移到 {to_type}"
        )
        return consolidated_count

    def get_memory_stats(self) -> dict[str, Any]:
        """获取记忆统计信息"""
        pass

    def _classify_memory_type(self):
        pass

    def _calculate_importance(
        self,
    ):
        pass
