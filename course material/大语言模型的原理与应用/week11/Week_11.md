# 智能体经典范式构建：从三种 Agent Loop 到 Deep Research Agent

一个现代智能体最核心的能力，不是“会聊天”，而是能把大语言模型的推理能力接到外部世界上：它能理解问题、拆解任务、调用工具、观察结果，并把这些信息重新组织成答案。

本章我们仍然不急着使用 LangChain、LlamaIndex 这类成熟框架，而是继续“亲手造轮子”。原因很简单：只有亲手写一遍循环、解析、工具调用和失败处理，才会真正理解 Agent 为什么能工作，以及它容易在哪里翻车。

本章会完成四件事：

1. 实现 **ReAct**：边想边做，适合需要搜索、查询、调用 API 的任务。
2. 实现 **Plan-and-Solve**：先规划再执行，适合结构清晰、多步骤推理的任务。
3. 实现 **Reflection**：先生成，再反思，再修改，适合对质量要求高的任务。
4. 基于 **DeepXiv API** 搭建一个迷你 Deep Research Agent，让它能搜索论文、阅读摘要、整理研究结论。

> 本文档是一个 Jupyter Notebook。建议你从上到下依次运行代码，不要一上来就跳到最后一个 Agent。不然很容易出现“代码能跑，但脑子没跟上”的经典事故。

## 1. 环境准备

我们先准备三个最小依赖：

- `openai`：调用任何兼容 OpenAI Chat Completions 接口的模型服务。
- `python-dotenv`：从 `.env` 读取密钥。
- `requests`：调用 DeepXiv API。

如果你已经在课程环境中安装过，可以跳过这一格。

```python
# 如需安装依赖，取消下一行注释后运行
# !pip install openai python-dotenv requests
```

在项目根目录创建 `.env` 文件，并写入：

```bash
LLM_API_KEY="YOUR-API-KEY"
LLM_MODEL_ID="YOUR-MODEL"
LLM_BASE_URL="YOUR-BASE-URL"

# DeepXiv 的部分测试论文和测试 query 不需要 token；
# 但正式使用建议申请并填写。
DEEPXIV_API_TOKEN="YOUR-DEEPXIV-TOKEN"
```

DeepXiv 的 arXiv API 基础地址是：

```text
https://data.rag.ac.cn/arxiv/
```

其中最重要的是检索接口：

```text
GET /arxiv/?type=retrieve&query={QUERY}
```

我们后面会把它封装成 `DeepXivSearch` 工具。

```python
import os
import json
import ast
import re
import requests
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
```

## 2. 封装一个最小 LLM 客户端

不要把模型调用散落在各个类里面。我们先做一个很薄的 `HelloAgentsLLM`，之后所有 Agent 都通过它来“思考”。

这一步看似普通，但很关键：当你以后想换模型、换服务商、加日志、加重试时，只需要改这一层。

```python
class HelloAgentsLLM:
    """一个极简 LLM 客户端，支持任何兼容 OpenAI 接口的服务。"""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
    ):
        self.model = model or os.getenv("LLM_MODEL_ID")
        api_key = api_key or os.getenv("LLM_API_KEY")
        base_url = base_url or os.getenv("LLM_BASE_URL")

        if not all([self.model, api_key, base_url]):
            raise ValueError("请先在 .env 中配置 LLM_MODEL_ID、LLM_API_KEY、LLM_BASE_URL。")

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def think(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        stream: bool = False,
    ) -> str:
        """调用模型并返回文本。教学场景下默认 temperature=0，减少随机性。"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=stream,
        )

        if not stream:
            return response.choices[0].message.content or ""

        chunks = []
        for chunk in response:
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content or ""
            print(text, end="", flush=True)
            chunks.append(text)
        print()
        return "".join(chunks)
```

```python
# 测试 LLM 客户端
# llm = HelloAgentsLLM()
# llm.think([{"role": "user", "content": "用一句话解释什么是智能体。"}])
```

## 3. 工具系统：让 Agent 有“手和脚”

如果说 LLM 是大脑，那么工具就是智能体的手和脚。工具要有三个要素：

1. **名称**：模型调用它时使用的名字，例如 `DeepXivSearch`。
2. **描述**：告诉模型什么时候应该使用这个工具。
3. **执行函数**：真正干活的 Python 函数。

我们先写一个通用工具执行器。

```python
class ToolExecutor:
    """负责注册、描述和执行工具。"""

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, description: str, func: Callable[[str], str]):
        self.tools[name] = {"description": description, "func": func}
        print(f"✅ 工具已注册: {name}")

    def get(self, name: str) -> Optional[Callable[[str], str]]:
        tool = self.tools.get(name)
        return tool["func"] if tool else None

    def describe(self) -> str:
        return "\n".join(
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        )

    def run(self, name: str, tool_input: str) -> str:
        func = self.get(name)
        if not func:
            return f"工具错误：未找到工具 {name}"
        try:
            return func(tool_input)
        except Exception as e:
            return f"工具执行失败：{type(e).__name__}: {e}"
```

## 4. DeepXiv API 封装：把论文库接进 Agent

DeepXiv 提供了面向 Agent 的论文数据接口。我们主要使用四类能力：

- `retrieve`：按语义搜索论文，可返回标题、摘要、TLDR、URL、日期、引用数等。
- `brief`：获取单篇论文的简要信息。
- `preview`：获取论文前若干字符，适合快速扫读。
- `raw`：获取完整 Markdown 内容，适合深度阅读，但会更长。

教学中我们先控制规模：每次只取前 3~5 篇，避免上下文太长。

```python
class DeepXivClient:
    """DeepXiv arXiv API 的极简封装。"""

    BASE_URL = "https://data.rag.ac.cn/arxiv/"

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("DEEPXIV_API_TOKEN")

    def _get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        # 没有 token 时也允许请求：DeepXiv 提供了部分免费测试论文和免费测试 query。
        response = requests.get(self.BASE_URL, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source: str = "arxiv",
        return_contents: bool = False,
        use_fine_rerank: bool = True,
    ) -> Dict[str, Any]:
        return self._get({
            "type": "retrieve",
            "query": query,
            "source": source,
            "top_k": top_k,
            "return_contents": str(return_contents).lower(),
            "use_fine_rerank": str(use_fine_rerank).lower(),
        })

    def brief(self, arxiv_id: str) -> Dict[str, Any]:
        return self._get({"type": "brief", "arxiv_id": arxiv_id})

    def preview(self, arxiv_id: str, characters: int = 6000) -> Dict[str, Any]:
        return self._get({
            "type": "preview",
            "arxiv_id": arxiv_id,
            "characters": characters,
        })

    def raw(self, arxiv_id: str) -> Dict[str, Any]:
        return self._get({"type": "raw", "arxiv_id": arxiv_id})
```

```python
def format_papers(results: Dict[str, Any], max_items: int = 5) -> str:
    """把 DeepXiv 检索结果压缩成适合喂给 LLM 的文本。"""
    papers = results.get("result", [])[:max_items]
    if not papers:
        return "没有检索到相关论文。"

    lines = []
    for i, paper in enumerate(papers, start=1):
        paper_id = paper.get("arxiv_id") or paper.get("biorxiv_id") or paper.get("medrxiv_id")
        title = paper.get("title", "无标题")
        tldr = paper.get("tldr") or paper.get("abstract", "无摘要")
        date = paper.get("date") or paper.get("publish_at", "未知日期")
        url = paper.get("url") or paper.get("src_url", "")
        citations = paper.get("citation_count", paper.get("citations", "未知"))

        lines.append(
            f"[{i}] {title}\n"
            f"ID: {paper_id}\n"
            f"Date: {date}\n"
            f"Citations: {citations}\n"
            f"URL: {url}\n"
            f"Summary: {tldr}"
        )
    return "\n\n".join(lines)


deepxiv = DeepXivClient()


def deepxiv_search_tool(query: str) -> str:
    """供 Agent 调用的论文搜索工具。"""
    print(f"🔍 DeepXivSearch: {query}")
    results = deepxiv.retrieve(query=query, top_k=5, return_contents=False)
    return format_papers(results, max_items=5)
```

```python
# 免费测试 query 示例：DeepXiv 文档中提到 large language model 可免费检索
# print(deepxiv_search_tool("large language model"))
```

## 5. 范式一：ReAct —— 边想边做

ReAct 的核心循环是：

```text
Thought → Action → Observation → Thought → ...
```

它适合那些**不知道下一步该查什么，必须边查边调整**的任务。比如：研究一个新方向、查询最新资料、调用多个 API 拼答案。

为了让代码更稳，这里不用脆弱的 `Thought: ... Action: ...` 文本解析，而是要求模型输出 JSON。真实项目里，结构化输出比正则表达式更抗揍。

```python
REACT_SYSTEM_PROMPT = """
你是一个能够调用外部工具的研究智能体。
你需要在每一步输出一个 JSON 对象，不要输出 Markdown，不要输出额外解释。

可用工具:
{tools}

输出格式:
{{
  "thought": "你对当前问题的分析",
  "action": {{
    "name": "工具名称，或 Finish",
    "input": "工具输入，或最终答案"
  }}
}}

规则:
1. 如果需要查论文或外部知识，调用 DeepXivSearch。
2. 如果已经有足够证据回答问题，使用 Finish。
3. 不要编造检索结果中没有的信息。
"""


def safe_json_loads(text: str) -> Dict[str, Any]:
    """尽量从模型输出中提取 JSON。"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


class ReActAgent:
    def __init__(self, llm: HelloAgentsLLM, tools: ToolExecutor, max_steps: int = 4):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps

    def run(self, question: str) -> str:
        history = []

        for step in range(1, self.max_steps + 1):
            print(f"\n--- ReAct 第 {step} 步 ---")
            messages = [
                {
                    "role": "system",
                    "content": REACT_SYSTEM_PROMPT.format(tools=self.tools.describe()),
                },
                {
                    "role": "user",
                    "content": (
                        f"问题：{question}\n\n"
                        f"历史观察：\n{chr(10).join(history) if history else '暂无'}"
                    ),
                },
            ]

            raw = self.llm.think(messages)
            decision = safe_json_loads(raw)

            thought = decision.get("thought", "")
            action = decision.get("action", {})
            name = action.get("name")
            tool_input = action.get("input", "")

            print(f"🧠 Thought: {thought}")
            print(f"🎬 Action: {name}[{tool_input}]")

            if name == "Finish":
                print("✅ ReAct 完成。")
                return tool_input

            observation = self.tools.run(name, tool_input)
            print(f"👀 Observation:\n{observation[:1200]}")

            history.append(f"Action: {name}[{tool_input}]")
            history.append(f"Observation: {observation}")

        return "已达到最大步数，但还没有得到最终答案。"
```

```python
# ReAct 示例
# llm = HelloAgentsLLM()
# tools = ToolExecutor()
# tools.register(
#     "DeepXivSearch",
#     "搜索 arXiv / bioRxiv / medRxiv 论文。当你需要查找某个研究方向、方法或论文证据时使用。",
#     deepxiv_search_tool,
# )
# react_agent = ReActAgent(llm, tools)
# answer = react_agent.run("请帮我找几篇关于 large language model evaluation 的代表性论文，并总结它们关注的问题。")
# print(answer)
```

### ReAct 小结

ReAct 的优点是灵活：它可以根据观察结果不断调整搜索词。缺点也很明显：每一步都要调用模型，成本更高；而且如果模型输出格式不稳定，循环就会断。

调试时重点看三件事：

1. 模型有没有选对工具。
2. 工具输入是否具体。
3. Observation 是否足够短、足够有信息量。

## 6. 范式二：Plan-and-Solve —— 先画蓝图，再动手

Plan-and-Solve 把任务拆成两个阶段：

```text
Plan：先生成步骤
Solve：再逐步执行
```

它适合结构清晰的任务，例如写报告、做实验计划、解多步题、实现一个模块。它不像 ReAct 那样“走一步看一步”，而是先尽量把路线想清楚。

```python
PLANNER_PROMPT = """
你是一个严谨的任务规划专家。请把用户问题拆成 3 到 6 个可执行步骤。
只输出 JSON 列表，不要输出额外文字。

用户问题:
{question}
"""

EXECUTOR_PROMPT = """
你是一个执行专家。请严格根据当前步骤完成任务。

原始问题:
{question}

完整计划:
{plan}

已完成步骤:
{history}

当前步骤:
{current_step}

请只输出当前步骤的结果，不要重复计划。
"""


class Planner:
    def __init__(self, llm: HelloAgentsLLM):
        self.llm = llm

    def plan(self, question: str) -> List[str]:
        raw = self.llm.think([{"role": "user", "content": PLANNER_PROMPT.format(question=question)}])
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", raw, flags=re.S)
            plan = json.loads(match.group(0)) if match else []
        return plan if isinstance(plan, list) else []


class Executor:
    def __init__(self, llm: HelloAgentsLLM):
        self.llm = llm

    def execute(self, question: str, plan: List[str]) -> str:
        history = []
        last_result = ""

        for i, step in enumerate(plan, start=1):
            print(f"\n-> 执行步骤 {i}/{len(plan)}: {step}")
            prompt = EXECUTOR_PROMPT.format(
                question=question,
                plan=json.dumps(plan, ensure_ascii=False),
                history="\n".join(history) if history else "暂无",
                current_step=step,
            )
            last_result = self.llm.think([{"role": "user", "content": prompt}])
            print(f"✅ 结果: {last_result[:500]}")
            history.append(f"步骤 {i}: {step}\n结果: {last_result}")

        return last_result


class PlanAndSolveAgent:
    def __init__(self, llm: HelloAgentsLLM):
        self.planner = Planner(llm)
        self.executor = Executor(llm)

    def run(self, question: str) -> str:
        print("--- 正在生成计划 ---")
        plan = self.planner.plan(question)
        print(json.dumps(plan, ensure_ascii=False, indent=2))

        if not plan:
            return "计划生成失败。"

        print("\n--- 正在执行计划 ---")
        return self.executor.execute(question, plan)
```

```python
# Plan-and-Solve 示例
# llm = HelloAgentsLLM()
# ps_agent = PlanAndSolveAgent(llm)
# result = ps_agent.run("请设计一个学习 ReAct、Plan-and-Solve、Reflection 的两小时课堂练习流程。")
# print(result)
```

### Plan-and-Solve 小结

Plan-and-Solve 的优势是稳定：先有全局路线，再逐步推进。它适合“目标明确、路径可分解”的任务。

它的问题是：如果第一版计划错了，后面会很认真地沿着错误路线走下去。所以在复杂任务中，经常会把它和 Reflection 结合起来：先规划，再执行，再评审计划是否合理。

## 7. 范式三：Reflection —— 给 Agent 装一个“自我校对器”

Reflection 的流程是：

```text
执行 → 反思 → 优化
```

它适合那些一次生成不一定可靠、但值得多花一点成本打磨的任务，例如代码生成、研究报告、实验设计、重要邮件。

这里我们实现一个最小版本：先生成答案，再让模型扮演评审员指出问题，最后根据反馈重写。

```python
DRAFT_PROMPT = """
你是一名研究助教。请完成下面任务，要求结构清晰，语言简洁。

任务:
{task}
"""

REFLECT_PROMPT = """
你是一名严格的评审员。请审查下面的初稿，只关注三个问题：
1. 是否回答了任务？
2. 是否存在事实跳跃或证据不足？
3. 结构是否清晰？

任务:
{task}

初稿:
{draft}

请输出具体、可执行的修改建议。如果无需修改，请输出“无需改进”。
"""

REFINE_PROMPT = """
你是一名研究助教。请根据评审意见修改初稿。

任务:
{task}

初稿:
{draft}

评审意见:
{feedback}

请输出修改后的最终版本。
"""


class ReflectionAgent:
    def __init__(self, llm: HelloAgentsLLM, max_iterations: int = 2):
        self.llm = llm
        self.max_iterations = max_iterations
        self.memory: List[Dict[str, str]] = []

    def run(self, task: str) -> str:
        print("--- 初始执行 ---")
        draft = self.llm.think([{"role": "user", "content": DRAFT_PROMPT.format(task=task)}])
        self.memory.append({"type": "draft", "content": draft})

        for i in range(1, self.max_iterations + 1):
            print(f"\n--- 第 {i} 轮反思 ---")
            feedback = self.llm.think([{
                "role": "user",
                "content": REFLECT_PROMPT.format(task=task, draft=draft),
            }])
            self.memory.append({"type": "feedback", "content": feedback})
            print(f"🧾 Feedback: {feedback[:600]}")

            if "无需改进" in feedback:
                break

            print("\n--- 根据反馈优化 ---")
            draft = self.llm.think([{
                "role": "user",
                "content": REFINE_PROMPT.format(
                    task=task,
                    draft=draft,
                    feedback=feedback,
                ),
            }])
            self.memory.append({"type": "draft", "content": draft})

        return draft
```

```python
# Reflection 示例
# llm = HelloAgentsLLM()
# reflection_agent = ReflectionAgent(llm)
# final = reflection_agent.run("解释 ReAct 和 Plan-and-Solve 的区别，并给出各自适用场景。")
# print(final)
```

### Reflection 小结

Reflection 本质上是一种“以成本换质量”的策略。它不一定让 Agent 更快，但通常能让结果更稳、更完整。

使用时要注意：评审提示词必须具体。不要只写“请反思一下”，否则模型很可能给出一段漂亮但没什么用的废话。更好的方式是指定评审维度，例如“事实性、逻辑性、证据充分性、可执行性”。

## 8. 实战：搭一个迷你 Deep Research Agent

现在我们把三种范式组合起来，做一个小型 Deep Research Agent。它的目标不是取代真正的深度研究系统，而是让你理解“研究型 Agent”的最小闭环。

它会执行四步：

1. **Plan**：把研究问题拆成若干检索子问题。
2. **Search**：调用 DeepXiv 检索相关论文。
3. **Synthesize**：根据检索结果写一版研究综述。
4. **Reflect**：检查综述是否证据不足、结构混乱，并修改。

这其实就是一个混合范式：

- Plan-and-Solve 负责全局路线。
- ReAct 的工具调用负责拿证据。
- Reflection 负责最后的质量控制。

```python
# Mini Deep Research Agent 示例
# 注意：请自行定义完整的MiniDeepResearchAgent
#
# llm = HelloAgentsLLM()
# research_agent = MiniDeepResearchAgent(llm, DeepXivClient())
# report = research_agent.run("What are the main research directions in large language model evaluation?")
# print(report)
```

## 9. 你应该真正掌握什么

到这里，我们已经完成了三个经典范式和一个迷你研究智能体。请不要只记代码，重点记住它们各自的“任务气质”：

| 范式                  | 核心循环                  | 适合任务            | 常见风险          |
| ------------------- | --------------------- | --------------- | ------------- |
| ReAct               | 思考 → 行动 → 观察          | 需要搜索、查 API、动态调整 | 步数多、成本高、格式易乱  |
| Plan-and-Solve      | 规划 → 执行               | 路径清晰、多步骤推理      | 计划错了会一路错下去    |
| Reflection          | 执行 → 反思 → 优化          | 代码、报告、关键答案      | 成本更高，评审提示词要具体 |
| Deep Research Agent | 规划检索 → 搜索论文 → 综合 → 反思 | 研究综述、论文调研       | 检索不全、证据过度解读   |

一个实用经验是：真实 Agent 很少只用一种范式。更常见的是混合使用：用 Plan-and-Solve 定路线，用 ReAct 拿信息，用 Reflection 做质检。

## 10. 练习

1. 修改 `ReActAgent`，让它除了 `DeepXivSearch` 之外还能调用 `DeepXivPreview`，在找到关键论文后读取前 6000 字符。
2. 修改 `MiniDeepResearchAgent`，让它支持 `source="biorxiv"` 或 `source="medrxiv"`。
3. 给 Reflection 增加一个终止条件：如果连续两轮反馈都只提出风格问题，而不是事实或逻辑问题，就停止迭代。
4. 尝试把最终综述保存成 Markdown 文件，并自动附上论文 URL 列表。

到这一步，你已经不只是“会用 Agent 框架”，而是知道了一个 Agent Loop 是怎么被搭出来的。框架当然好用，但当标准组件不够用的时候，能从零搭一个小循环，才是真正的工程底气。