# LitCode Agent

仓库地址：https://github.com/Lit-Codegit/LitCode-Agent

LitCode Agent 是一个从零实现的本地编程智能体。它通过 OpenAI-compatible 模型的原生 tool calling 读取和修改文件、搜索代码、执行命令，并循环完成编程任务。项目未使用 Agent 框架或服务端托管的代码执行、文件工具；对话历史、上下文、工具分发、响应解析、终止条件和错误处理均在仓库内实现。

## 运行方法

支持 Windows 10/11、macOS 和 Linux。源码运行需要 Git、Python 3.12 与 uv；安装 ripgrep（`rg`）可获得更快的搜索速度，未安装时会自动使用 Python 搜索。

```bash
git clone https://github.com/Lit-Codegit/LitCode-Agent.git
cd LitCode-Agent
uv sync --frozen
uv run pytest
uv run litcode
```

首次启动后输入 `/connect`，选择模型供应商，粘贴 API Key 并选择模型；密钥只保存到用户级凭据文件（POSIX 权限为 `0600`），不写入项目配置。也可使用环境变量提供密钥。非交互执行示例：

```bash
uv run litcode run "检查测试，修复失败并验证结果。"
```

## 特色功能

- Textual 全屏 TUI 实时展示模型输出、工具调用、错误和终止原因，危险命令默认请求确认。
- 文件工具以真实路径限制读写范围；命令设置超时和输出截断；Agent 循环设置最大轮数。
- SQLite 保留原始会话、检查点和分支，支持 `@` 文件引用、`#` 会话引用、rewind/redo、多 pane 和后台子会话。
- 支持流式响应、模型切换、Agent Skills 与本地定时任务；厂商差异集中在模型适配器中。

## 他机部署与说明

源码方式最适合复现；也可运行 `uv build` 生成 wheel，再把 wheel 发给其他用户。用户只需安装 uv，然后运行 `uv tool install litcode_agent-0.1.0-py3-none-any.whl`，以后可在任意目录直接执行 `litcode`，无需再写 `uv run`。Windows 优先使用 PowerShell 7，其次使用 Windows PowerShell，再回退到 `cmd.exe`；macOS/Linux 保持系统 `/bin/sh` 行为。Skill 同步需要创建符号链接，Windows 若提示权限不足，应开启 Developer Mode 或使用管理员终端。
