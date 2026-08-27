# LitCode Agent

LitCode Agent is a small coding agent built from first principles. It will use
an OpenAI-compatible language model to inspect a workspace, edit files, run
commands, and report the result through a transparent command-line workflow.

The project intentionally does not use an agent framework. Conversation state,
tool dispatch, local execution, termination, and error handling live in this
repository.

## Current status

The project is under active development. The current milestone provides the
package skeleton, environment-based configuration, and a configuration doctor.

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
