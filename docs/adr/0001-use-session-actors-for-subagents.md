---
status: accepted
---

# 使用 Session Actor 表达子 Agent 协作

LitCode 使用带持久化 FIFO Queue 的 Session 作为协作、执行和恢复的核心单位，Subagent 是带 `parent_id` 的 child Session，Pane 只是可拆装视图。我们不再用独立的 Orchestration Run、Task、Scheduler、Ledger 和显式 report/finish 协议表达协作，因为这些状态与 Session 生命周期重复且让 UI 参与调度；前台 child 的最终回答直接成为工具结果，后台 child 的结果进入父 Session Queue。这个选择借鉴 OpenCode 的 Task tool 与 child session 核心机制，但保留 LitCode 的透明队列、用户挂载和安全上限，不引入其后台 job 服务架构。
