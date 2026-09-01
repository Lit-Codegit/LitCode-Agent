# Session Actor 与轻量 Subagent 设计

> 状态：讨论通过，尚未实施。本文描述目标行为，不代表当前代码已经交付。

## 目标

LitCode 用尽可能少的概念支持用户可控与 Agent 自主的多会话协作：用户可以创建、观察、挂载和暂停 Subagent；Agent 可以前台等待或后台运行 child；所有协作输入进入透明队列，不打断目标正在执行的 turn。

设计的核心不是“多个 Pane 互相通信”，而是“单进程内多个 Session 协作”。Pane 是视图，Session 是持久化 mailbox actor。

## 核心模型

```text
Session Tree
└─ Session
   ├─ message history
   ├─ transparent FIFO Session Queue
   ├─ single-consumer runner
   └─ Agent Turn
      └─ spawn_subagent → child Session
```

### Session

Session 保存独立历史、父子 identity、队列和运行状态。同一个 Session 同时最多执行一个 Agent Turn，也最多挂载到一个 Pane。

Root Session 没有父级。Subagent Session 带不可变的 `parent_id`，在历史会话中按文件树形式缩进显示。用户和 Agent 使用同一种 child session 机制；区别只在调用入口和授权来源。

### Pane

Pane 只负责选择和显示 Session：

- LitCode 启动时显示一个 Empty Pane，不预先创建 Session；
- Empty Pane 收到第一条输入时惰性创建 Root Session，命令输入也遵循该规则；
- `/split` 只创建 Empty Pane 或挂载已有 Session，不建立父子关系；
- `/subagent` 创建 child Session，可选择立即挂载到新 Pane；
- 同一 Session 已挂载时，再次选择只聚焦原 Pane；
- `/nohup` 卸载当前 Session，但不暂停它；有 sibling 时由 sibling 子树接管区域，最后一个 Pane 则回到 Empty Pane；
- 分界线可拖动，split ratio 仅影响显示；
- 第一版不跨进程保存 split tree、ratio 和挂载关系。

## Session Queue

每个 Session 只有一个持久化 FIFO Queue。用户、父子 Agent 和其他 Session 都向同一队列追加输入。

- 入队不取消、不插队，也不修改正在运行的 Agent Turn；
- 并发追加由 SQLite 事务原子确定顺序；
- Agent 只决定现在入队还是等待，入队后不能重排、跳过或取消；
- 用户可以重排或取消尚未开始的项目；
- 空闲 Session 在队列非空时自动领取队首；
- Paused Session 继续收消息，但不领取下一条；暂停不打断当前 turn；
- 用户可直接暂停或恢复，Agent 通过默认 `ask` 的权限操作代为控制；
- 恢复后立即继续消费队首。

“透明”不表示把所有队列内容注入每个模型请求。TUI 可以展示完整队列；模型常驻上下文只有有界 Session Catalog：树位置、状态、活动摘要、队列长度和更新时间。Agent 通过只读工具按需分页读取活动、队列与历史。

## Subagent 调用

建议的模型工具形状：

```text
spawn_subagent(prompt, agent?, background=false, session_id?)
```

- `prompt` 完整描述目标、验收条件和协作方式；
- `agent` 选择 Agent Profile，默认 `general`；
- `background=false` 时父 turn 等待 child，child 最终回答直接成为工具结果；
- `background=true` 时立即返回 child session ID，完成后向父 Session Queue 追加一条带来源和 child 引用的结果；
- `session_id` 只用于继续调用方自己直接创建的 child；
- 旁系、跨树和祖孙之间通过目标 Session Queue 协作，不劫持既有 child invocation；
- 不要求 child 调用 `report_task`，也不创建独立 Task 或 Run identity。

用户可以随时把运行中或已结束的 child 挂到 Pane，查看完整历史并直接追加消息。追加消息仍进入同一个 FIFO Queue，不打断当前 turn。

## Agent Profile 与权限

首版只需两个稳定 Profile：

- `general`：继承父级模型和正常工具能力；
- `explore`：只读调查，禁止文件写入和可能写入工作区的命令。

实现、审查和研究不是协议状态，而是任务 prompt。自定义 Profile 以后只承载稳定的 system prompt、模型和工具权限。

child 的有效权限是以下三者的交集：

```text
workspace hard limits ∩ parent delegable permissions ∩ agent profile
```

Profile 和 descendant 只能收紧权限，不能恢复父级 deny。Root Session 从工作区权限开始计算。

后台 Session 的权限请求进入全局确认中心，不依赖 Pane：只暂停请求方并释放执行槽；用户可直接决定或先挂载查看。批准后重新排队，拒绝后向模型返回普通工具错误。应用退出时未决请求按拒绝处理。

## 运行与取消

默认运行限制：

- Session Tree 最大深度 2：Root → Child → Grandchild；
- 每个用户发起的 Root Agent Turn 共享 8 次 descendant invocation 预算；
- 单进程最多 4 个正在请求模型或执行工具的 Agent Turn；
- 等待前台 child 的父 turn 释放执行槽，child 返回后按等待顺序恢复；
- 没有槽的 invocation 保持可见等待状态。

取消采用结构化传播：

- 父 turn 正常完成时，后台 children 可以继续；
- 用户主动停止父 turn，或父 turn 因错误、模型请求失败、运行上限而异常结束时，级联终止本轮仍在运行或等待的 descendants；
- 终止的是 invocation，不删除 child Session、既有历史或来源无关的普通队列消息；
- `/nohup` 只卸载 Pane，不改变取消所有权。

Background Session 只跨 Pane 存活，不跨 LitCode 应用进程。退出时活动 turns 在安全边界终止并标记 `interrupted`，Session Tree、历史和队列保留，重新打开后由用户决定是否继续。第一版同一工作区只允许一个活动 LitCode 进程，不实现 daemon 或跨进程调度。

## 文件与命令并发

文件修改使用短生命周期的 real-path 文件写锁：

1. Agent 读取文件并记录内容版本；
2. 修改工具按规范化 real path 排序取得涉及的文件锁；
3. 确认当前版本仍等于读取版本；
4. 原子落盘并立即释放全部锁；
5. 版本不一致时返回可见冲突，不静默覆盖。

不同文件可以并行修改。任意 shell 命令因无法可靠推导写集合，在子进程运行期间取得工作区级命令锁；该锁与其他命令和文件写入阶段互斥，超时、取消、成功或失败都必须释放。文件读取和模型思考仍可并行。

## 模块边界

目标模块可以保持很小：

```text
LitCodeTUI
├─ PaneLayout             split tree、focus、ratio、mount
├─ SessionTreeView        会话树、状态、队列入口
└─ ConfirmationCenter     全局权限请求

SessionRuntime
├─ SessionQueue           持久化 FIFO 与用户队列操作
├─ SessionRunner          单消费者、暂停、执行槽
└─ InvocationScope        父子等待、预算与级联取消

Agent / AgentSession
├─ 保持单 Session Agent 循环
├─ spawn_subagent tool
└─ 按需 session read/send/control tools

SessionStore
├─ sessions.parent_id / profile / paused
├─ queued_messages / turns
└─ messages / summaries / checkpoints
```

PaneLayout 不调度 Agent；SQLite 不决定下一步任务；模型不维护第二套 orchestration 状态机。SessionRunner 只执行确定性的 mailbox 与资源规则。

## 相对当前实现的删除与替换

删除目标：

- `OrchestrationRun`、`OrchestrationTask`、`OrchestrationEvent`；
- `LocalScheduler`；
- `/orchestrate` 及 pause/resume/cancel orchestration 命令；
- `delegate_session`、`report_task`、`list_orchestration`、`finish_orchestration`；
- orchestration 专用 SQLite 表和 role/write-policy 上下文。

替换为：

- `sessions.parent_id` 与 Agent Profile；
- 一个持久化 Session Queue；
- 每 Session 单消费者 runner；
- `spawn_subagent`、session read/send/control 工具；
- TUI Session Tree、全局确认中心和可拖动 Pane divider；
- 进程级执行槽、文件锁和命令锁。

## 建议迁移切片

每个切片都保持单 Session CLI 和 TUI 可运行：

1. **持久化基础**：加入 `parent_id`、queue item、paused 与 turn 状态，迁移旧数据库；不接入自动执行。
2. **SessionRunner**：用单消费者 FIFO 驱动现有 `AgentSession.ask`，验证入队、暂停、恢复、退出恢复和并发顺序。
3. **Subagent 核心**：实现 child session、前台等待、后台结果、深度/预算/执行槽和级联取消。
4. **删除 orchestration**：迁移测试后移除 run/task/scheduler/report 协议及数据库表的运行依赖；旧表可保留只读迁移期，不立即破坏用户数据库。
5. **TUI 投影**：Session Tree、挂载、`/nohup`、Empty Pane、全局确认和队列管理。
6. **并发写安全**：文件锁、版本校验、命令锁和冲突可见性。
7. **Pane 体验**：拖动 divider 与窄屏行为；布局持久化明确留到后续。

## 验证重点

- FIFO 原子顺序、一次只运行一个 turn、暂停只阻止下一条；
- 前台结果、后台结果入队、继续 direct child；
- 深度 2、8 次调用预算、4 个执行槽和等待父级释放槽；
- 正常完成与异常/主动停止的不同级联行为；
- 未挂载 child 的流式执行、权限确认、完成与重新挂载；
- Empty Pane 不创建幽灵 Session，第一条输入才创建 Root；
- 同 Session 不重复挂载，`/nohup` sibling 收缩和最后 Pane 初始态；
- 同文件版本冲突、多文件锁顺序、命令锁超时释放；
- 应用退出后 turns interrupted，历史和队列可恢复；
- 旧数据库迁移和旧 orchestration 数据不丢失。

## 已知非目标

- 跨 LitCode 进程执行、常驻 daemon 和真正的操作系统级 nohup；
- 多工作区或远程 Session 协作；
- 持久化 Pane 布局；
- 任意深度递归或无上限后台调用；
- 中断目标 turn 来插入跨 Session 消息；
- 自动把所有 Session 历史和队列正文注入模型上下文。
