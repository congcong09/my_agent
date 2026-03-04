from typing import Any, Callable

from .base import Tool


class ToolRegistry:
    """
    HelloAgents工具注册表
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, Any]] = {}

    def register_tool(self, tool: Tool):
        """
        注册Tool对象
        """
        if tool.name in self._tools:
            print(f"⚠️ 警告：工具 [{tool.name}] 已存在，将被覆盖")
        self._tools[tool.name] = tool

        print(f"✅ 工具 [{tool.name}] 已注册")

    def register_function(
        self, name: str, description: str, func: Callable[[str], str]
    ):
        """
        直接注册函数作为工具（简便方式）
        """
        if name in self._functions:
            print(f"⚠️ 警告：工具 [{name}] 已存在，将被覆盖")

        self._functions[name] = {"description": description, "func": func}
        print(f"✅ 工具 [{name}] 已注册")

    def unregister(self, name: str):
        """注销工具"""
        if name in self._tools:
            del self._tools[name]
            print(f"🗑 工具 [{name}] 已注销")
        elif name in self._functions:
            del self._functions[name]
            print(f"🗑 工具 [{name}] 已注销")
        else:
            print(f"⚠️ 工具 [{name}] 不存在")

    def get_tool(self, name: str):
        return self._tools.get(name)

    def get_function(self, name: str):
        return self._functions.get(name)

    def execute_tool(self, name: str, input_text: str) -> str:
        """
        执行工具
        """
        if name in self._tools:
            tool = self._tools[name]
            try:
                return tool.run({"input": input_text})
            except Exception as e:
                return f"错误：执行工具 [{name}] 时发生异常：{e}"
        elif name in self._functions:
            func = self._functions[name]["func"]
            try:
                return func(input_text)
            except Exception as e:
                return f"错误：执行工具 [{name}] 时发生异常：{e}"
        else:
            return f"错误：未找到名为 [{name}] 的工具"

    def get_tools_description(self) -> str:
        descriptions = []

        for tool in self._tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")

        for name, info in self._functions.items():
            descriptions.append(f"- {name}: {info['description']}")

        return "\n".join(descriptions) if descriptions else "暂无可用工具"

    def list_tools(self) -> list[str]:
        """列出所有的工具名称"""
        return list(self._tools.keys()) + list(self._functions.keys())

    def get_all_tools(self) -> list[Tool]:
        """获取所有的 Tool 对象"""
        return list(self._tools.values())

    def clear(self):
        """清空所有工具"""
        self._tools.clear()
        self._functions.clear()
        print("🧹 所有工具已清空")


global_registry = ToolRegistry()
