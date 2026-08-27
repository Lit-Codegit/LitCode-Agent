# LitCode Agent

LitCode Agent 是一个从零实现的小型编程智能体。它通过 OpenAI-compatible 模型检查工作区、修改文件、执行命令，并在透明的命令行流程中报告结果。

本项目刻意不使用任何 Agent 框架。对话状态、工具分发、本地执行、循环终止和错误处理均在仓库内自行实现。

## 当前状态

核心 MVP 已可运行：包含模型适配器、有轮数上限的 Agent 循环、五个本地工具、配置诊断和端到端测试。下一阶段将继续改进真实模型表现和命令行体验。

## 运行原理

```text
任务 -> 模型 -> 工具调用 -> 本地校验与执行 -> 工具结果
         ^                                  |
         +----------- 对话历史 <------------+
```

模型可以列出文件、读取文件、用 `rg` 搜索、原子地执行一次精确文本替换，以及运行带超时和输出限制的命令。文件路径必须位于选定工作区内，危险命令默认要求确认。

## 开发环境

```bash
uv sync
uv run pytest
uv run litcode --help
```

项目配置位于 `.litcode/settings.json`。API Key 不直接写入配置文件；配置只声明读取哪个环境变量：

```bash
export OPENAI_API_KEY="你的密钥"
```

检查配置，输出中不会出现密钥：

```bash
uv run litcode doctor
```

在当前目录执行任务：

```bash
uv run litcode run "检查测试，修复失败行为并完成验证。"
```

指定另一个工作区：

```bash
uv run litcode run --workspace ../small-project "为输入增加校验。"
```

## 配置与 hooks

`.litcode/settings.json` 可以配置模型名、API 地址、API Key 环境变量名、Agent 限制、命令策略和生命周期 hooks。`.litcode/settings.local.json` 可覆盖项目配置，适合本机差异，并已被 Git 忽略。环境变量优先级最高，便于临时切换模型或网关。

Hook 配置借鉴 Claude Code 的事件、matcher 和 command handler 结构。命令从标准输入接收 JSON 事件。具体格式见 [配置说明](.litcode/README.md)。

## 安全模型

文件工具拒绝绝对路径、父目录穿越和解析后的符号链接逃逸。命令从工作区启动，具有超时限制，并在匹配危险模式时要求确认。

命令执行不是操作系统沙箱：得到允许的 shell 命令仍可能访问工作区以外的资源。评估不熟悉的模型时，请使用一次性测试仓库。
