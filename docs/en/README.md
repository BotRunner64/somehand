# somehand Documentation

Use this index by entrypoint. CLI is for running the packaged tool; API is for embedding retargeting in Python.

---

## Start Here

| I want to... | Read |
| --- | --- |
| Install dependencies and verify one run | [Getting Started](getting-started.md) |
| Run from the terminal | [CLI Usage](runtime-modes.md) |
| Embed retargeting in Python | [API Usage](api.md) |
| Select a hand model or change a YAML config | [Configuration](configuration.md) |
| Download runtime assets or check model availability | [Assets & Models](assets-and-models.md) |

---

## Reference

| Doc | What it covers |
| --- | --- |
| [Getting Started](getting-started.md) | Install, assets, first run, optional SDK setup |
| [CLI Usage](runtime-modes.md) | Terminal commands for live, replay, export, and hardware use |
| [API Usage](api.md) | Stable Python imports, one-step engines, and session hooks |
| [Configuration](configuration.md) | Which config file to edit and the fields that usually matter |
| [Assets & Models](assets-and-models.md) | Download commands, local paths, supported config families |
| [Maintainer Guide](maintainer-guide.md) | Docs rules, verification commands, model-update workflow |

---

## Scope

The common CLI path is single-hand retargeting. Bi-hand support is for `viewer` replay/render workflows, and real-hardware control is single-hand only. API users normally provide their own landmark frames and call the engine directly.
