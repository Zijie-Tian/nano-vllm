# Kiro CLI Configuration for COMPASS

This directory contains Kiro CLI agent configurations and skills for the COMPASS project.

## Structure

```
.kiro/
├── agents/           # Agent configurations (JSON)
│   ├── compass-default.json
│   └── compass-deep-thinker.json
└── skills/           # Skill definitions (Markdown with YAML frontmatter)
    ├── compass-commands/
    ├── compass-gpu-testing/
    ├── compass-3rdparty-policy/
    └── ...
```

## Agents

### compass-default
Default COMPASS development agent with full toolset for:
- Code analysis and navigation
- Benchmark execution
- GPU testing
- Documentation management

### compass-deep-thinker
Deep analytical agent for:
- Architecture decisions
- Feasibility verification
- Complex root-cause analysis
- Solution comparison

**Keyboard shortcut**: `Ctrl+Shift+T`

## Skills

Skills are progressively loaded on demand. Each skill references detailed rules in `.codex/rules/`.

| Skill | Purpose |
|-------|---------|
| compass-commands | Runtime commands, environment setup |
| compass-gpu-testing | GPU testing and benchmark execution |
| compass-3rdparty-policy | 3rdparty directory protection |
| compass-code-analysis | Code navigation and call-chain analysis |
| compass-testing | Testing conventions and verification |
| compass-doc-management | Documentation workflow |
| compass-feasibility-report | Feasibility analysis workflow |
| compass-pre-plan-git-check | Pre-implementation git sync check |
| compass-ruler-task-config | RULER task configuration sync |
| compass-kvcache-rope-data | KVCache/RoPE data access |

## Usage

Kiro CLI automatically loads agent configurations from this directory. Switch agents with:

```
/agent compass-deep-thinker
```

Or use the keyboard shortcut `Ctrl+Shift+T` for deep-thinker.

## Relationship to Other Configs

- `.codex/` - Codex-specific configurations (TOML format)
- `.claude/` - Claude-specific configurations
- `.kiro/` - Kiro CLI configurations (JSON format)

All three share the same rule sources in `.codex/rules/` and `.claude/rules/`.
