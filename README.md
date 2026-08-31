# LitCode Agent

LitCode Agent 是一个从零实现的小型编程智能体。它通过 OpenAI-compatible 模型检查工作区、修改文件、执行命令，并在透明的命令行流程中报告结果。

本项目刻意不使用任何 Agent 框架。对话状态、工具分发、本地执行、循环终止和错误处理均在仓库内自行实现。

## 当前状态

核心 MVP 已可运行：包含模型流式适配器、有轮数上限的 Agent 循环、五个工作区工具、标准 Agent Skills、SQLite 会话持久化、多 pane 并发、渐进式跨会话上下文、持久化 inbox、检查点 rewind/redo/fork、命名只读根和端到端测试。

## 运行原理

```text
任务 -> 模型 -> 工具调用 -> 本地校验与执行 -> 工具结果
         ^                                  |
         +----------- 对话历史 <------------+
```

模型可以列出文件、读取文件、用 `rg` 搜索、原子地执行一次精确文本替换，以及运行带超时和输出限制的命令。文件路径必须位于选定工作区内，危险命令默认要求确认。

这里使用了模型 API 原生的 tool calling 协议，但五个工具的实现、权限边界和执行过程均由 LitCode 自己完成。模型负责选择工具并生成参数，模型厂商不负责读取本地文件或执行命令。

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

启动当前目录的全屏 TUI：

```bash
uv run litcode
```

打开另一个工作区：

```bash
uv run litcode ../small-project
```

`uv run litcode chat [路径]` 作为兼容写法继续可用。

TUI 中央是可滚动的对话和工具时间线，底部是固定的多行输入区，状态栏显示工作区、配置档、模型与当前阶段。模型文本会实时流式显示；工具卡片可以展开或折叠，危险命令会通过弹窗确认。

输入 `/` 会打开命令模糊补全；输入 `@` 会搜索当前工作区和显式配置的命名只读根；输入 `#` 会搜索当前工作区的会话 alias。目录候选以 `/` 结尾，选中后继续导航；文件和会话分别插入 `@{relative/path}`、`#{YYMMDD-HHMM-XXX}`。提交时 LitCode 只附加受大小限制的不可变快照。敏感文件、二进制文件、未授权路径和跨工作区会话都会被拒绝。

每个会话拥有类似 `260830-1432-K7M` 的稳定 alias：前半是本地创建时间，后三位使用易辨认的 Crockford Base32 ASCII 字符。`#` 默认只注入会话 capsule；模型需要细节时可用 `read_session_context` 按 query 读取有界片段。实际 capsule 及来源版本会持久化，不会随来源会话后来变化。

项目 Skill 放在 `.agents/skills/<name>/SKILL.md`。LitCode 启动时只把合法 Skill 的 `name` 和 `description` 放入目录；模型调用 `load_skill` 后才加载正文，`scripts/`、`references/` 和 `assets/` 仍只列文件名、按需读取。Skill 不会自动取得额外文件或命令权限。

会话保存在本地 `.litcode/sessions.db` 中，该文件已被 Git 忽略。每个完成的用户轮次自动建立检查点。`/compact` 只建立摘要视图，不删除原始消息。`/rewind` 会让用户选择是否同时恢复 Agent 通过 `apply_patch` 编辑的文件；如果文件后来被其他人修改，恢复会拒绝覆盖。

工具调用在对话流中默认折叠，标题只显示运行状态、工具名和关键参数。展开后显示有长度上限的运行摘要，不展示完整参数 JSON；模型仍会收到完整工具结果，因此界面摘要不会影响 Agent 的后续判断。

快捷键：

- `Ctrl+Enter`：发送，多行输入中的 `Enter` 保留为换行；
- `F2`：查询并选择 API 模型；
- `Cmd+方向键`：向对应方向创建会话 pane；这是 best-effort 入口，多数 macOS 终端会拦截 Cmd，不应作为唯一入口；
- `Cmd+Shift+方向键`：移动 pane 焦点；
- `Ctrl+W` 后按方向键：可靠的分屏入口；也可直接输入 `/split right`；
- `Ctrl+L`：清空当前会话；
- `Ctrl+C`：任务运行时请求停止，空闲时退出。

斜杠命令：

- `/help`：查看命令；
- `/model`：重新查询 API，通过弹窗切换当前模型；
- `/clear`：清空上下文，开始新会话；
- `/sessions` 或 `/resume`：选择并恢复本工作区的历史会话；
- `/compact [可选要求]`：手动压缩上下文，保留完整原始历史；
- `/rewind`：选择检查点，再选择是否同时恢复 Agent 编辑的文件；
- `/redo`：撤销最近一次 rewind；
- `/fork`：从检查点创建独立会话分支；
- `/split left|right|up|down`：创建 pane，第一版最多 4 个；
- `/focus left|right|up|down`：移动 pane 焦点；
- `/close-pane`：关闭视图但保留持久化会话；
- `/inbox`：显示并确认当前会话收到的跨会话指示；
- `/exit`：退出。

`/model` 只影响当前进程，不会改写配置文件；普通消息会保留在当前会话的线性历史中。停止采用协作式取消：如果模型请求或命令正在阻塞，会在该调用返回后的第一个安全控制边界停止。

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

`agent.maxReferenceFileChars` 和 `agent.maxReferenceChars` 分别限制单个引用文件和单轮全部文件引用；`agent.maxSessionReferenceChars` 限制单轮 `#` capsule 总量，避免跨会话历史无边界占用上下文。

会话工具包括 `list_sessions`、`read_session_context`、`send_session_message` 和 `read_session_inbox`。发送者 session ID 由运行时注入，模型不能伪造；消息只进入目标 inbox，不会自动唤醒目标模型。`permissions.sessionMessages` 可设为 `confirm`、`deny` 或 `allow`，默认 `confirm`。

工作区外路径不能任意漫游，必须在本机配置中显式命名。`list_files`、`read_file`、`search_files` 可读，`apply_patch` 始终只能写工作区：

```json
{
  "permissions": {
    "readRoots": {
      "local": {
        "path": "local",
        "sendToModel": true
      },
      "docs": {
        "path": "../shared-docs",
        "sendToModel": false
      }
    }
  }
}
```

`sendToModel: false` 允许本地工具读取和搜索，但拒绝通过 `@` 把整个文件快照发给模型。仓库的 `.litcode/settings.local.json` 已在本机显式开启 `local` 引用，所以 `@local` 现在可搜索；该配置不会提交。

Hook 配置借鉴 Claude Code 的事件、matcher 和 command handler 结构。命令从标准输入接收 JSON 事件。具体格式见 [配置说明](.litcode/README.md)。

## 安全模型

写工具拒绝绝对路径、父目录穿越和解析后的符号链接逃逸。只读根同样进行 real-path containment 检查。命令从工作区启动，具有超时限制，并在匹配危险模式时要求确认。

命令执行不是操作系统沙箱：得到允许的 shell 命令仍可能访问工作区以外的资源。文件引用内容也会发送到配置的模型 API；评估不熟悉的模型时，请使用一次性测试仓库。
