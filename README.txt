LitCode Agent

仓库地址：https://github.com/Lit-Codegit/LitCode-Agent

LitCode Agent 是一个从零实现的本地编程智能体。它通过 OpenAI-compatible 模型的原生 tool calling 读取和修改文件、搜索代码、执行命令，并循环完成编程任务。项目未使用 Agent 框架或服务端托管的代码执行、文件工具；对话历史、上下文、工具分发、响应解析、终止条件和错误处理均在仓库内实现。支持 Windows 10/11、macOS 和 Linux。

运行方式（两种，功能完全相同，区别只在内容）

方式一：源码运行（推荐，便于复现与阅读实现）

git clone https://github.com/Lit-Codegit/LitCode-Agent.git
cd LitCode-Agent
uv sync --frozen
uv run litcode

需要 Git、Python 3.12 与 uv。该方式附带文档、测试、项目配置样例和 .litcode/skills/ 下的 3 个示例 Skills。

方式二：wheel 全局安装（把工具带到其他机器）

在源码目录执行 uv build，生成 litcode_agent-0.1.0-py3-none-any.whl，发给目标用户；对方只需安装 uv，然后运行：

uv tool install litcode_agent-0.1.0-py3-none-any.whl

以后即可在任意目录直接执行 litcode，无需 uv run。注意 wheel 只包含程序本身，不含文档、测试与示例 Skills（Skills 从不随软件分发，见下）。

凭据与 Skills 说明

API Key 一律从环境变量或用户级凭据文件读取（POSIX 权限 0600），不写入项目配置，绝不进入仓库。Skills 则是在运行时发现的：Agent 从当前工作区的 .litcode/skills/（兼容 .agent/skills、.agents/skills）与用户级 ~/.agents/skills（Windows 为 C:\Users\用户\.agents\skills）读取带 SKILL.md 的目录；可用 /skill 创建、安装、列出或同步到其他 Agent。因此源码仓库里的 3 个示例 Skills 只在仓库内可见，whl 模式不会自带；用户级 Skills 两种方式都共享。

首次使用

启动后输入 /connect，选择供应商并在弹窗中粘贴 API Key、选择模型；密钥只保存到用户级凭据文件。也可用环境变量提供密钥。非交互执行示例：

uv run litcode run "检查测试，修复失败并验证结果。"

特色功能

- Textual 全屏 TUI 实时展示模型输出、工具调用、错误和终止原因，危险命令默认请求确认。
- 文件工具以真实路径限制读写范围；命令设置超时和输出截断；Agent 循环设置最大轮数。
- SQLite 保留原始会话、检查点和分支，支持 @ 文件引用、# 会话引用、rewind/redo、多 pane 和后台子会话。
- 支持流式响应、模型切换、Agent Skills 与本地定时任务；厂商差异集中在模型适配器中。
- 跨平台细节：优先使用 ripgrep 搜索，未安装时自动回退 Python 实现；命令用平台默认 shell（Windows 优先 PowerShell 7）。Skill 同步需要符号链接，Windows 若提示权限不足，请开启 Developer Mode 或使用管理员终端。
