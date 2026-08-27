# LitCode Agent

LitCode Agent is a small coding agent built from first principles. It will use
an OpenAI-compatible language model to inspect a workspace, edit files, run
commands, and report the result through a transparent command-line workflow.

The project intentionally does not use an agent framework. Conversation state,
tool dispatch, local execution, termination, and error handling live in this
repository.

## Current status

The core MVP is runnable: it has an OpenAI-compatible model adapter, a bounded
agent loop, five local tools, environment-based configuration, and a diagnostic
command. UX and real-repository evaluation are still being refined.

## How it works

```text
task -> model -> tool calls -> local validation/execution -> tool results
             ^                                             |
             +---------------- conversation history <------+
```

The model can list files, read files, search with `rg`, apply one exact text
replacement atomically, and run a command with timeout and output limits. File
paths are resolved inside the selected workspace. Dangerous commands are
confirmed by default.

## Development setup

```bash
uv sync
uv run pytest
uv run litcode --help
```

Runtime credentials will be read from environment variables and must never be
committed:

```bash
export OPENAI_API_KEY="..."
export LITCODE_MODEL="..."
# Optional for an OpenAI-compatible gateway:
export OPENAI_BASE_URL="https://example.com/v1"
```

Run the non-secret configuration check with:

```bash
uv run litcode doctor
```

Run a task against the current directory:

```bash
uv run litcode run "Inspect the tests, fix the failing behavior, and verify it."
```

Or select another workspace:

```bash
uv run litcode run --workspace ../small-project "Add input validation."
```

## Safety model

Filesystem tools reject absolute paths, parent traversal, and resolved symlink
escapes. Commands start in the workspace, have a timeout, and require approval
when they match dangerous patterns. Command execution is not an OS sandbox: an
approved shell command can still access resources outside the workspace. Use a
disposable repository when evaluating an unfamiliar model.
