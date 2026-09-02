# LitCode 协作模型

LitCode 让用户和 Agent 在同一工作区中创建、观察和组织多个可持续运行的会话，同时保持协作过程可见、可控且足够轻量。

## Language

**Session（会话）**：
保存独立对话历史并承载一次 Agent 执行流的持久化单位。
_Avoid_：Pane、进程

**Agent Turn（执行轮次）**：
一个 Session 从接收一条输入开始，到返回结果、失败、受限停止或被用户终止为止的一次连续执行。
_Avoid_：Session、对话

**Root Session（根会话）**：
不由其他会话派生、代表一棵协作会话树入口的 Session。
_Avoid_：主 Pane

**Subagent Session（子 Agent 会话）**：
由用户或 Agent 为一个委派目标创建、并通过父子关系归入会话树的 Session。
_Avoid_：Worker Pane、Orchestration Task

**Subagent Invocation（子 Agent 调用）**：
父 Agent Turn 在一个 Subagent Session 中发起并等待或后台运行的一次 Agent Turn。
_Avoid_：Orchestration Run、Task Record

**Agent Profile（Agent 配置档）**：
描述一种 Agent 的稳定 system prompt、默认模型与工具权限的命名配置。
_Avoid_：Orchestration Role、一次性任务说明

**Pane（窗格）**：
用户用于选择、挂载并观察或操作至多一个 Session 的可拆分视图。
_Avoid_：Agent、执行容器、进程

**Pristine Session（未使用会话）**：
已挂载到 Pane、但从未收到输入、产生 Turn、修改文件、被引用或参与计划任务的 Root Session。
_Avoid_：Empty Pane、草稿 Pane、空 Session

**Mounted Session（已挂载会话）**：
当前显示在某个 Pane 中的 Session。
_Avoid_：前台 Agent

**Background Session（后台会话）**：
当前未挂载到 Pane、但仍可继续运行并保留完整历史的 Session。
_Avoid_：已关闭会话、孤儿进程

**Session Tree（会话树）**：
以 Root Session 为根、按委派父子关系组织所有 Subagent Session 的层级历史。
_Avoid_：Pane Layout、调用日志

**Session Queue（会话队列）**：
附属于一个 Session、按顺序保存用户和其他 Agent 待处理输入的透明队列。
_Avoid_：Pane Queue、私有 Inbox

**Paused Session（已暂停会话）**：
保留队列和运行状态、但不会领取下一条 Queued Message 的 Session。
_Avoid_：Background Session、已终止会话

**Queued Message（队列消息）**：
用户或一个 Session 发给目标 Session、等待目标在安全边界消费的异步输入。
_Avoid_：中断、实时事件

**File Write Lock（文件写锁）**：
以解析后的工作区真实文件路径为键、阻止两个写操作同时修改同一文件的排他控制。
_Avoid_：Workspace Lock、任务所有权

**Workspace Command Lock（工作区命令锁）**：
任意本地命令运行期间阻止其他命令和文件写工具同时修改工作区的排他控制。
_Avoid_：Agent Turn Lock、只读锁

## Relationships

- 一个 **Pane** 同一时刻恰好挂载一个 **Session**，因此创建后立即可由 Session 标识寻址。
- 一个 **Session** 同一时刻最多挂载到一个 Pane；在选择器中选择已挂载 Session 时聚焦现有 Pane，不创建重复视图。
- 一个 **Session** 同一时刻可以处于 **Mounted Session** 或 **Background Session** 状态。
- `/nohup` 将当前 **Session** 从 **Pane** 卸载到后台，并让相邻 Pane 合并释放出的区域。
- `/nohup` 移除当前 Pane leaf 后，由 split tree 中与它配对的 sibling 子树接管父节点的全部区域，不固定向某个屏幕方向合并。
- 最后一个 Pane 执行 `/nohup` 时保留 Pane 容器，并立即挂载一个新的 Root Session；原 Session 转入后台。
- 用户可以拖动 Pane 之间的分界线调整相邻区域比例；调整视图尺寸不影响任何 Session 的执行状态。
- 第一版不跨 LitCode 进程持久化 Pane split tree、分界比例或挂载关系；重新打开时从单 Pane 开始，由用户从 Session Tree 重新挂载会话。
- Session Tree、对话历史、Session Queue 和父子关系不属于 Pane 布局，必须跨进程重启持久化。
- 用户可以新建 Pane，并从 Session Tree 或历史会话选择器重新挂载一个 Background Session。
- 一个 **Root Session** 可以拥有零个或多个 **Subagent Session**。
- 一个 **Subagent Session** 只有一个父 Session，也可以继续创建自己的 Subagent Session。
- Root Session 的树深度为 0；默认最大子会话深度为 2，即允许 Child 和 Grandchild，深度 2 的 Session 不能继续创建 Subagent Session。
- 用户手动创建独立 Root Session 不计入其他 Session Tree 的深度；子会话深度只通过持久化的父子关系计算。
- 每个用户发起的 Root Agent Turn 默认共享 8 次 Subagent Invocation 预算；所有后代的新建和继续调用都各消耗一次，不能在子 Agent 中重置。
- 达到调用预算后，后续 Subagent Invocation 作为可恢复工具错误拒绝，不暂停或终止已经存在的 Session。
- 用户和 Agent 使用同一种子会话机制创建 **Subagent Session**，区别只在发起者与授权方式。
- Subagent Invocation 的具体目标、验收条件和协作方式写入完整任务 prompt，不固化为 implementer、reviewer 等协议角色。
- 可选 Agent Profile 只表达跨任务稳定的能力；首版内置 general 与只读 explore，未指定时使用 general。
- Subagent Session 的有效权限是工作区硬限制、父 Session 可委派权限与 Agent Profile 权限的交集；Profile 和再次委派只能收紧权限，不能恢复父级已拒绝的能力。
- 用户直接创建 Root Session 时从工作区权限计算，不继承其他 Session 的权限。
- 子会话深度、Root Agent Turn 调用预算和全局执行槽限制同样不能由 descendant 放宽。
- 前台 **Subagent Invocation** 会等待子 Agent Turn，并把子 Agent 的最终回答直接作为调用结果。
- 后台 **Subagent Invocation** 会立即返回 Subagent Session 标识；完成后向父 Session Queue 追加一条引用该子会话的结果消息，不打断父 Agent Turn。
- 子 Agent 的最终回答就是调用完成信号，不要求额外的 report 或 finish 协议。
- 继续既有 Subagent Session 时复用它的 Session 标识和历史，不另建 Task 或 Run 身份。
- Agent 只能通过 Session 标识继续自己直接创建的 Subagent Session；不能把旁系、跨树或孙级 Session 作为自己的既有 child 调用。
- Session 之间的旁系与跨树协作必须通过目标 Session Queue 进行；用户可以从 UI 直接打开并继续任意 Session。
- 单个 LitCode 进程默认最多同时执行 4 个 Agent Turn；Pane 数量不构成执行槽数量或 Subagent 数量上限。
- 没有取得执行槽的 Subagent Invocation 保持可见的等待状态，并按进入等待状态的顺序取得后续空闲槽。
- 执行槽只由正在请求模型或执行工具的 Agent Turn 占用；等待前台 Subagent Invocation 返回的父 Agent Turn 释放执行槽。
- 前台 child 返回后，父 Agent Turn 按进入等待状态的顺序重新取得执行槽并继续。
- **Pane** 的拆分、合并和尺寸不决定 **Session** 是否继续运行。
- LitCode 启动或 `/split` 创建 Pane 时立即创建并挂载一个 Root Session，使其他 Session 可以向它投递第一条 Queued Message。
- `/split` 只创建或调整 Pane，并允许选择挂载已有 Session；它不建立 Session 父子关系。选择已有 Session 或取消创建时，临时 Root Session 仅在仍是 Pristine Session 时删除。
- 关闭 Pane 时，Pristine Session 随 Pane 删除；已经承载任何持久活动的 Session 转为 Background Session。
- `/subagent` 从当前 Session 创建 Subagent Session；可选的 Pane 参数只决定是否立即把子会话挂载到新 Pane。
- 一个 **Session** 只有一个 **Session Queue**，所有用户和 Agent 共享该队列的可见状态。
- TUI 可以展示 Session Queue 的完整来源、内容、顺序与状态；“完全透明”表示没有隐藏项目且可按需读取，不表示每轮自动注入全部队列正文。
- 模型每轮只自动获得有界 Session Catalog，包括树位置、状态、活动摘要、队列长度与更新时间；队列正文和历史通过只读工具按需分页读取。
- **Queued Message** 按发送顺序进入目标 Session Queue，不取消、不插队，也不改变目标正在运行的 Agent Turn。
- 发送方 Agent 可以查看目标的当前 Agent Turn 和完整排队状态，自主选择立即入队或等待后再决定。
- 多个发送方并发入队时，由持久化层原子确定消息顺序；目标 Session 同一时刻只执行一条消息。
- Agent 不能重排、抢占、跳过或取消已经入队的消息；用户可以调整或取消尚未开始执行的消息。
- 空闲 Session 在队列从空变为非空时自动唤起；一个 Agent Turn 结束后继续按 FIFO 领取下一条消息，不需要独立的全局 Scheduler。
- Paused Session 继续接收入队消息，但不领取下一条；暂停操作不会中断已经开始的 Agent Turn。
- 用户可以直接暂停 Session，也可以要求一个 Agent 代为执行暂停操作。
- Agent 可以通过权限受控的 Session 操作暂停或恢复其他 Session，默认需要用户确认；用户直接操作不需要二次确认。
- 恢复一个队列非空的 Paused Session 后，该 Session 立即从队首继续消费。
- 暂停和恢复的发起者、目标与时间对所有 Session 可见。
- 权限确认附属于发起请求的 Session，不附属于 Pane；未挂载的 Background Session 也可以发起全局可见的确认请求。
- 等待权限确认的 Agent Turn 只暂停自身并释放执行槽，其他 Session 继续运行；用户可直接处理请求或先挂载来源 Session 查看上下文。
- 权限请求批准后，原 Agent Turn 重新排队取得执行槽；拒绝后向 Agent 返回可见工具错误。
- LitCode 退出时，未回答的权限请求按拒绝处理，相关 Agent Turn 标记为 interrupted。
- 所有文件修改在落盘前必须取得目标 real path 的 File Write Lock；不同文件不共享同一把写锁。
- File Write Lock 只覆盖一次写工具调用中的版本校验与原子落盘，Agent 思考和其他工具调用期间不持锁。
- 写入前必须确认文件版本仍等于 Agent 读取的版本；不一致时返回可见冲突，不静默覆盖。
- 一次操作修改多个文件时，按规范化 real path 排序取得全部文件锁，并在操作结束或失败时全部释放。
- 任意本地命令在子进程运行期间持有 Workspace Command Lock；该锁与所有 File Write Lock 的写入阶段互斥。
- 命令超时、取消、成功或失败后都必须释放 Workspace Command Lock；文件读取和 Agent 思考不受该锁影响。
- 当用户终止一个 Agent Turn 时，由该轮执行唤起且仍在运行的子 Agent Turn 会被级联终止；对应 Subagent Session 及其历史仍然保留。
- 父 Agent Turn 正常完成时，本轮创建的后台 Subagent Invocation 可以继续运行，并在完成后向父 Session Queue 返回结果。
- 父 Agent Turn 被用户主动停止，或因错误、模型请求失败、运行上限而异常结束时，本轮仍在运行或等待的 descendant invocations 会被级联终止。
- 级联终止不删除 Subagent Session、既有历史或该 Session Queue 中来源无关的普通消息。
- `/nohup` 只改变 Session 的 Pane 挂载状态，不解除子 Agent Turn 对父 Agent Turn 的级联终止关系。
- Background Session 只在当前 LitCode 应用进程存活期间继续执行，不由常驻 daemon 托管。
- 退出 LitCode 时，运行中的 Agent Turn 在安全边界终止并标记为 interrupted；Session Tree、历史和 Session Queue 持久化保留，重新打开后由用户决定是否继续。
- 同一工作区同一时刻只有一个活动 LitCode 应用进程；该进程在内部并发运行多个 Session，不进行跨进程 Agent 调度。
- 第二个 LitCode 进程不能同时接管同一工作区；活动进程退出并释放工作区所有权后，后续进程才能读取并继续持久化状态。

## Example dialogue

> **用户：**“把测试调查交给一个子 Agent，然后让我在右侧 Pane 看它执行。”
> **LitCode：**“我创建了一个 Subagent Session，并把它挂载到右侧 Pane。”
> **用户：**“先 `/nohup`，我想把右侧空间还给主会话。”
> **LitCode：**“该 Session 已转到后台继续运行；它仍保留在 Session Tree 中，可随时重新挂载。”

## Flagged ambiguities

- “多进程 pane 通信”容易把视图、执行和进程混为一谈；已明确目标是单进程内的多 Session 协作，Pane 是视图，Session 是执行与持久化单位。
- “不同 Session 间了解对方在做什么”已明确为自动共享有界活动摘要、按需读取历史，以及通过 Queued Message 显式发送指示；不自动广播完整输出。
- “挂起”曾同时表示从 Pane 卸载和停止执行；已区分为 Background Session（可继续运行）与被终止的 Agent Turn（已停止但 Session 可继续使用）。
- “Pane 的消息队列”会让消息生命周期依赖临时视图；已明确为持久化的 Session Queue，Pane 只展示和操作它。
- 当前实现中的 Orchestration Run、Task、Scheduler、Ledger 和显式 report/finish 协议不属于目标领域模型；目标模型以 Session Tree、Agent Turn 和 Session Queue 表达协作。
