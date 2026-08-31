# 多会话工作台与渐进式上下文设计（审核稿）

> 实施状态：2026-08-30 已完成第一阶段：alias/`#` capsule、项目 Agent Skills、最多四 pane 并发、方向快捷键 fallback、局部会话读取和持久化 inbox。窄窗折叠、`doctor` 按键探测、单轮 excerpt 总预算、inbox 因果链/身份、读写权限拆分、逐版本迁移器、自动唤醒目标模型、全局/远程 Skill 和 pane 布局跨进程恢复明确留在后续阶段；下文相关段落是目标设计，不代表本阶段已交付。

## 目标

把当前单时间线 TUI 扩展为一个小型“会话工作台”：用户可以像 tmux 一样拆分视图、快速切换独立会话，并用稳定、易输入的会话代号通过 `#` 引用其他会话。跨会话上下文和 Agent Skills 都采用渐进式加载，避免把完整历史或全部技能正文塞进每次模型请求。

这不是操作系统级 tmux，也不启动 shell pane。每个 pane 展示一个 LitCode `AgentSession`；对话、模型调用、工具执行和持久化仍由本项目自行实现。

## 设计原则

1. **会话是运行与持久化单位，pane 只是视图。** 关闭 pane 不结束或删除会话；同一会话不能同时挂载到两个 pane。
2. **跨会话内容是显式快照。** 用户通过 `#` 或 Agent 通过会话工具选中来源，发送时记录来源会话、版本和实际注入文本，重放历史时不会读取到后来变化的内容。
3. **先目录、再摘要、最后局部原文。** 模型先看到很小的会话/Skill 元数据，只在需要时调用工具加载更深一层。
4. **跨会话输出按不受信数据处理。** 引用内容不能覆盖当前会话的 system prompt、权限和用户要求。
5. **并发不能扩大权限。** 所有 pane 共享同一工作区权限；跨会话发送、自动唤醒和 Skill 加载分别受控并可见。
6. **不引入 Agent 框架。** 新能力建立在现有 `Agent`、`AgentSession`、`SessionStore`、工具注册表和 Textual TUI 之上。

## 用户模型

### 会话代号

建议使用本地时间与 Crockford Base32 后缀组成稳定代号：

```text
260830-1432-K7M · 修复模型切换
```

- `260830-1432`：创建时间，方便人按时间定位；
- `K7M`：只使用易辨认的 ASCII 字符，排除 `I/L/O/U`，用于同分钟碰撞消解；
- 后面的中文标题继续取第一条用户任务的 48 字符预览，并允许用户重命名；
- UUID 继续作为数据库主键，代号只是唯一、稳定、可输入的公开 alias。

不建议用 LLM 生成唯一名称：会增加一次请求、可能失败，也不利于离线创建会话。

### pane 操作

- `Cmd+方向键`：向对应方向拆分，并在新 pane 中创建会话或选择已有会话；
- `Cmd+Shift+方向键`：把焦点移动到对应 pane；
- `/split left|right|up|down`、`/focus ...`、`/close-pane`：可发现、可测试的命令入口；
- fallback：`Ctrl+W` 后按方向键，默认启用且可配置。

终端并不保证把 macOS `Cmd` 传给 TUI。LitCode 可以注册 `super+方向键`，但必须保留 fallback，并在 `doctor` 中提供按键探测提示。第一版不修改用户的终端配置。

布局使用一棵二叉 split tree：叶节点是 pane，内部节点记录横向或纵向切分及比例。第一版最多 4 个 pane，防止窄终端不可用；窗口过小时保持会话运行，只把非活动 pane 折叠成标签。

## `#` 跨会话引用

输入 `#` 打开与 `/`、`@` 一致的行内模糊候选，候选显示 alias、标题、运行状态、未读数和模型。选中后插入：

```text
参考 #{260830-1432-K7M} 的测试结论，继续修复这里。
```

提交时分三层加载：

1. **catalog**：候选阶段只有 alias、标题、状态、更新时间，不进入模型上下文；
2. **capsule**：被显式引用时注入有界快照，包含当前任务、最近完成结果、最近文件变更和已有压缩摘要，建议默认总计不超过 4,096 字符；
3. **excerpt**：模型确实需要细节时调用 `read_session_context`，用 query 在原始消息中检索并返回带来源标记的局部片段，单次和单轮都受预算限制。

第一版不为每次引用额外调用模型生成摘要。capsule 优先使用已有 `/compact` 摘要，否则由持久化数据确定性地组合最近用户目标、最终回答和文件变更。这样没有隐藏成本，也不会因摘要模型失败而阻止引用。

建议增加三个只读/通信工具：

- `list_sessions(query?, status?)`：返回有界目录；
- `read_session_context(session, query, max_chars)`：返回相关局部上下文，不返回整库历史；
- `send_session_message(session, instruction)`：向目标会话的持久化 inbox 投递消息。

`send_session_message` 默认只投递并显示未读，不自动发起模型请求。自动唤醒会产生费用、可能形成会话互相指示的无限循环，因此作为后续独立能力，必须具备权限开关、跳数上限、每会话轮数上限和清晰的来源链。

## Agent Skills

实现 [Agent Skills 规范](https://agentskills.io/specification)的核心子集：

- 发现工作区 `.agents/skills/<name>/SKILL.md`；
- 严格校验 `name`、目录名、`description` 和 YAML frontmatter；
- system prompt 只暴露允许使用的 `name + description`；
- 模型通过 `load_skill(name)` 加载完整 `SKILL.md` 正文；
- `scripts/`、`references/`、`assets/` 只列出有界目录，后续通过现有安全读取/命令能力按需使用；
- Skill 内容按不受信指令处理，不能自行扩大工具、目录或命令权限。

第一版只自动发现项目 Skill。全局 Skill 涉及工作区外读取和向模型发送内容，后续通过显式命名只读根启用，不默认扫描用户主目录。暂不实现远程 Skill catalog、自动安装或 `allowed-tools` 自动授权。

Skill 可以教模型何时调用会话工具，但 Skill 本身不是后台进程、事件监听器或调度器。“监视输出”和“发送指示”属于会话运行时：

- TUI 通过进程内事件总线实时更新 pane；
- SQLite inbox 保证进程重启后消息不丢；
- Agent 只有在自己的模型轮次中才能读取/发送；
- 后续若允许自动唤醒，必须由显式 scheduler 执行，不能伪装成 Skill 加载行为。

## 模块边界

```text
LitCodeTUI
  ├─ PaneLayout            只管理 split tree、焦点和尺寸
  ├─ SessionController[*]  每个打开会话的 worker、状态与渲染事件
  └─ CompletionIndex       统一 /、@、# 候选

Agent / AgentSession
  ├─ 保持单会话循环
  ├─ 接收不可变的引用快照
  └─ 通过 ToolRegistry 使用 skill/session 工具

SessionStore
  ├─ sessions / messages / summaries / checkpoints
  ├─ aliases / inbox / reference_snapshots
  └─ 线程安全事务与迁移
```

不把 pane 状态放进 `AgentSession`，不让渲染层决定 Agent 行为。多个会话可以并行等待模型；`apply_patch` 和 `run_command` 先经过工作区级执行锁，避免两个 pane 同时修改或测试同一工作区。所有等待锁、投递、唤醒和冲突都必须显示给用户。

## SQLite 演进

采用显式 `schema_version` 和小步迁移，不删除现有 `.litcode/sessions.db`：

- `sessions.alias`：唯一稳定代号；
- `session_inbox`：来源、目标、正文、状态、创建时间和因果链；
- `session_references`：当前消息实际引用的来源版本和 capsule；
- 可选 `pane_layouts`：第二阶段再持久化 UI 布局。

现有单 SQLite connection 在多 worker 下需要串行访问锁；启用 WAL 前先写并发测试，不把 WAL 当成线程安全的替代品。

## 权限与运行上限

- `permissions.sessionRead`：默认允许同工作区，只读；
- `permissions.sessionSend`：建议默认 `confirm`；
- `permissions.sessionWake`：第一版不实现，未来默认 `deny`；
- 不允许跨工作区 `#` 引用或投递；
- capsule、excerpt、单轮全部 session references 分别设字符上限；
- inbox 消息记录来源 alias 和原始用户/Agent 身份；
- 自动链路未来必须限制 hop、每分钟消息数、并发 worker 数和每会话最大轮次。

## 实施切片与验收

### 切片 A：修复模型切换

先用真实错误文本建立请求级回归测试，再做最小修复；切换失败不得污染当前会话或持久化模型选择。

### 切片 B：会话 identity 与快速切换

增加 alias、迁移和 `#` 候选，但只注入 capsule。验收：旧数据库可迁移；代号唯一稳定；快照重放一致；预算和敏感内容测试通过。

### 切片 C：标准 Skill

增加发现、校验、目录注入和 `load_skill`。验收：只常驻元数据；正文按需加载；重复名、畸形 frontmatter、路径逃逸和权限拒绝均可见。

### 切片 D：pane 布局

先抽出 `SessionController`，再实现 split tree、焦点、关闭和窄屏降级。验收：至少两个会话并行流式显示，事件不会串 pane，关闭 pane 不丢会话。

### 切片 E：会话通信

实现局部检索、inbox 和显式发送。验收：消息不丢、不重复；来源可追踪；目标繁忙时安全排队；不会自动形成模型循环。

每个切片都保持 `litcode run` 和单 pane TUI 可用，并补充一个适合面试复述的小练习。

## 需要审核的决定

1. alias 是否采用 `YYMMDD-HHMM-XXX`，还是更短的 `HHMM-XXX`？推荐前者，跨天仍唯一易懂。
2. pane 第一版是否最多 4 个？推荐 4 个，布局与并发边界最容易解释。
3. `sessionSend` 是否默认每次确认？推荐 `confirm`，项目配置可改为 `allow`。
4. 是否接受第一版“自动发送但不自动唤醒目标模型”？推荐接受，把自动唤醒作为带防循环协议的下一阶段。
5. 全局 Skill 是否暂不默认扫描，只支持项目 `.agents/skills`？推荐接受，避免静默扩大文件和上下文权限。
