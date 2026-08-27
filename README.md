# LitCode Agent

LitCode Agent 是一个从零实现的小型编程智能体。它通过 OpenAI-compatible 模型检查工作区、修改文件、执行命令，并在透明的命令行流程中报告结果。

本项目刻意不使用任何 Agent 框架。对话状态、工具分发、本地执行、循环终止和错误处理均在仓库内自行实现。

## 当前状态

核心 MVP 已可运行：包含模型适配器、有轮数上限的 Agent 循环、五个本地工具、配置诊断、模型查询与选择、保留上下文的交互会话和端到端测试。下一阶段将继续用真实模型任务校正提示词和工具行为。

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
export DEEPSEEK_API_KEY="你的密钥"
```

检查配置，输出中不会出现密钥：

```bash
uv run litcode doctor
```

查询当前 API 端点公开的模型：

```bash
uv run litcode models
```

启动交互会话：

```bash
uv run litcode chat
```

交互会话会显示用户消息、模型请求轮次、工具参数、工具结果和 hook 状态。它支持以下斜杠命令：

- `/help`：查看命令；
- `/model`：重新查询 API，通过序号切换当前模型；
- `/clear`：清空上下文，开始新会话；
- `/exit`：退出。

`/model` 只影响当前进程，不会改写配置文件；普通消息会保留在当前会话的线性历史中。

在当前目录执行任务：

```bash
uv run litcode run "检查测试，修复失败行为并完成验证。"
```

指定另一个工作区：

```bash
uv run litcode run --workspace ../small-project "为输入增加校验。"
```

也可以为单次运行覆盖配置档或具体模型：

```bash
uv run litcode run --profile fast --model fast-model-id "检查测试。"
```

## 配置与 hooks

`.litcode/settings.json` 通过 `defaultModel` 和 `models` 管理多个模型/API 配置档，也可以配置 Agent 限制、命令策略和生命周期 hooks。`.litcode/settings.local.json` 可覆盖项目配置，适合本机差异，并已被 Git 忽略。环境变量优先级最高，便于临时切换模型或网关。

Hook 配置借鉴 Claude Code 的事件、matcher 和 command handler 结构。命令从标准输入接收 JSON 事件。具体格式见 [配置说明](.litcode/README.md)。

## 安全模型

文件工具拒绝绝对路径、父目录穿越和解析后的符号链接逃逸。命令从工作区启动，具有超时限制，并在匹配危险模式时要求确认。

命令执行不是操作系统沙箱：得到允许的 shell 命令仍可能访问工作区以外的资源。评估不熟悉的模型时，请使用一次性测试仓库。
