# LitCode 配置

项目共享配置位于 `settings.json`，本机覆盖配置位于 `settings.local.json`。后者已被 Git 忽略。配置加载顺序如下，后加载的值优先：

1. 内置默认值
2. `.litcode/settings.json`
3. `.litcode/settings.local.json`
4. 环境变量

## 模型配置

```json
{
  "defaultModel": "primary",
  "models": {
    "primary": {
      "model": "主模型 ID",
      "baseURL": "https://example.com/v1",
      "apiKeyEnv": "PRIMARY_API_KEY"
    },
    "fast": {
      "model": "快速模型 ID",
      "baseURL": "https://example.com/v1",
      "apiKeyEnv": "FAST_API_KEY"
    }
  }
}
```

`defaultModel` 选择 `models` 中的默认配置档。`apiKeyEnv` 是环境变量名称，不是密钥本身。真实密钥应在 shell 中设置：

```bash
export PRIMARY_API_KEY="你的密钥"
```

`LITCODE_DEFAULT_MODEL=fast` 可以临时切换配置档；`LITCODE_MODEL`、`OPENAI_BASE_URL`、`LITCODE_MAX_ITERATIONS`、`LITCODE_MAX_OUTPUT_CHARS`、`LITCODE_COMMAND_TIMEOUT_SECONDS` 和 `LITCODE_COMMAND_POLICY` 可临时覆盖选中配置档中的具体值。

`agent.maxReferenceFileChars` 限制单个 `@` 引用文件的字符数，`agent.maxReferenceChars` 限制一轮消息全部文件快照的字符数。对应环境变量是 `LITCODE_MAX_REFERENCE_FILE_CHARS` 和 `LITCODE_MAX_REFERENCE_CHARS`，前者不得大于后者。

## 命名只读根

`permissions.readRoots` 用 alias 声明工作区外或被 Git ignore 的参考目录：

```json
{
  "permissions": {
    "readRoots": {
      "docs": {
        "path": "../shared-docs",
        "sendToModel": false
      }
    }
  }
}
```

`path` 的相对路径以工作区为基准。读取工具使用 `docs/相对路径` 访问；`sendToModel` 只控制是否允许 `@{docs/相对路径}` 附加文件快照。建议在 `settings.local.json` 中配置本机绝对路径。

## 会话数据

TUI 会话保存在 `.litcode/sessions.db`，包括原始消息、摘要检查点和 `apply_patch` 的文件前后镜像。该数据库为本机私有文件，已在 `.gitignore` 中排除。

命令行中的 `--profile` 对应 `LITCODE_DEFAULT_MODEL`，用于选择一整套 API 配置；`--model` 只覆盖其中的具体模型 ID。交互会话还可以用 `/model` 查询 API 并临时切换模型，这一操作不会写回 `settings.json`。

## Hook 配置

结构借鉴 Claude Code 的 `.claude/settings.json`，当前只实现 `type: "command"`：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "apply_patch|run_command",
        "hooks": [
          {
            "type": "command",
            "command": "python .litcode/hooks/log_tool.py",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

支持的事件：

- `SessionStart`
- `PreToolUse`
- `PostToolUse`
- `PostToolUseFailure`
- `SessionEnd`

Hook 命令从 stdin 接收 JSON，字段包括 `session_id`、`cwd`、`hook_event_name`，工具事件还包含 `tool_name`、`tool_input`、`tool_response` 和 `tool_use_id`。

`PreToolUse` hook 退出码为 `2` 时阻止该工具调用，stderr 会作为错误反馈交给模型。其他非零退出码会显示诊断信息，但不会改变 Agent 控制流。
