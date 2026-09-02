# 自动上下文压缩：OpenCode 机制调研

调研日期：2026-09-02。
上游范围：OpenCode 官方仓库 `anomalyco/opencode` 快照，重点阅读
`packages/opencode/src/session/compaction.ts`、`session/overflow.ts` 与
`session/prompt.ts`。

## OpenCode 的做法

- 每次模型步骤结束后记录 token usage，以模型 context/input limit 减去输出保留区，
  达到可用边界便在循环中插入自动压缩任务。
- 压缩摘要作为新消息保存，构造后续模型视图时隐藏更早内容；原始会话记录不删除。
- 摘要完成后插入 synthetic continue 消息，让尚未完成的 Agent 任务继续执行。
- 压缩时按 token 预算保留近期 turn；另有旧工具输出 pruning、插件扩展和专用
  compaction agent。

## LitCode 的选择

**借鉴核心机制**：在正常模型请求前检查上下文边界，命中时先生成摘要，再继续同一
Agent Turn；摘要继续使用现有“会话级摘要 + 边界”的派生视图，SQLite 原始消息不变。

**借鉴接口形状**：提供显式自动压缩开关和边界。LitCode 使用
`agent.autoCompactChars`（环境变量 `LITCODE_AUTO_COMPACT_CHARS`），默认
`200000` 字符，`0` 表示关闭。

**暂不照搬**：不引入 tokenizer、模型目录、专用 compaction agent、工具输出二阶段
pruning、plugin hook 或 synthetic message。当前 OpenAI-compatible 适配器没有稳定的
跨厂商 context limit/stream usage 契约，而项目已有统一的字符预算；字符边界虽然较
保守，但配置简单、行为可预测，也符合学生可完整解释的规模约束。

## 已知限制

- 字符数不等于 token 数；不同语言和模型的安全边界不同，用户应按模型上下文能力
  调整配置。
- 压缩请求本身仍需容纳待压缩历史，所以边界必须留出摘要 prompt 与输出空间。
- 当前摘要会覆盖达到边界前的完整派生视图，不额外保留近期 turn；后续如需改善摘要
  精度，应单独设计“摘要头 + 原文尾”的预算算法。
