import re

from openai.types.chat import ChatCompletionMessageParam

from hello_agents.tools.registry import ToolRegistry

from ..core import Agent, Config, HelloAgentsLLM, Message

# 默认ReAct提示词模板
DEFAULT_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤：

**Thought:** 分析当前问题，思考需要什么信息或采取什么行动。
**Action:** 选择一个行动，格式必须是以下之一：
- `{{tool_name}}[{{tool_input}}]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循：工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动："""


class ReActAgent(Agent):
    """
    ReAct (Reasoning and Acting) Agent

    结合推理和行动的智能体，能够：
    1. 分析问题并制定行动计划
    2. 调用外部工具获取信息
    3. 基于观察结果进行推理
    4. 迭代执行指导得出最终答案
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        system_prompt: str | None = None,
        config: Config | None = None,
        max_steps: int = 5,
        custom_prompt: str | None = None,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: list[str] = []

        self.prompt_template = custom_prompt or DEFAULT_REACT_PROMPT

    def run(self, input_text: str, **kwargs) -> str:
        """
        运行 ReAct Agent

        Args:
          input_text: 用户问题
          **kwargs: 其他参数

        Returns:
          最终答案
        """
        self.current_history = []
        current_step = 0

        print(f"\n🤖 {self.name} 开始处理问题：{input_text}")

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")
            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history)
            prompt = self.prompt_template.format(
                tools=tools_desc, question=input, history=history_str
            )
            messages: list[ChatCompletionMessageParam] = [
                {"role": "user", "content": prompt}
            ]
            response_text = self.llm.invoke(messages, **kwargs)

            if not response_text:
                print("❌ 错误:LLM未能返回有效响应。")
                break
            thought, action = self._parse_output(response_text)

            if thought:
                print(f"🤔 思考: {thought}")

            if not action:
                print("⚠️ 警告：未能解析出有效的Action，流程终止。")
                break
            # 检查是否完成
            if action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                print(f"🎉 最终答案: {final_answer}")

                # 保存到历史记录
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))

                return final_answer

            # 执行工具调用
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or tool_input is None:
                self.current_history.append("Observation: 无效的Action格式，请检查。")
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")

            # 调用工具
            observation = self.tool_registry.execute_tool(tool_name, tool_input)
            print(f"👀 观察: {observation}")

            # 更新历史
            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")

        print("⏰ 已达到最大步数，流程终止。")
        final_answer = "抱歉，我无法在限定步数内完成这个任务。"

        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))

        return final_answer

    def _parse_output(self, text: str):
        """解析LLM输出，提取思考和行动"""
        thought_match = re.search(r"Thought: (.*)", text)
        action_match = re.search(r"Action: (.*)", text)

        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None

        return thought, action

    def _parse_action(self, action_text: str):
        """解析行动文本，提取工具名称和输入"""
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        if match:
          return match.group(1), match.group(2)
        return None, None

    def _parse_action_input(self, action_text: str)
          """解析行动输入"""
          match = re.match(r"\w+\[(.*)\]", action_text)
          return match.group(1) if match else ""
