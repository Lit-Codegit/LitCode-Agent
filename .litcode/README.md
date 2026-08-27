# LitCode 配置

项目共享配置位于 `settings.json`，本机覆盖配置位于 `settings.local.json`。后者已被 Git 忽略。配置加载顺序如下，后加载的值优先：

1. 内置默认值
2. `.litcode/settings.json`
3. `.litcode/settings.local.json`
4. 环境变量

## 模型配置

```json
{
  "model": {
    "name": "模型名称",
    "baseURL": "https://example.com/v1",
    "apiKeyEnv": "OPENAI_API_KEY"
  }
}
```

`apiKeyEnv` 是环境变量名称，不是密钥本身。真实密钥应在 shell 中设置：

```bash
export OPENAI_API_KEY="你的密钥"
```

环境变量 `LITCODE_MODEL`、`OPENAI_BASE_URL`、`LITCODE_MAX_ITERATIONS`、`LITCODE_MAX_OUTPUT_CHARS`、`LITCODE_COMMAND_TIMEOUT_SECONDS` 和 `LITCODE_COMMAND_POLICY` 可临时覆盖文件配置。

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
