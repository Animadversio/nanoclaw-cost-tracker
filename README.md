# nanoclaw-cost-tracker

Token usage & API cost breakdown for [NanoClaw](https://github.com/qwibitai/nanoclaw) Claude Code sessions.

## What it does

Parses Claude Code session `.jsonl` files under `nanoclaw/data/sessions/` and reports:

- **Token breakdown**: input, cache write, cache read, output — per project and session
- **Estimated cost**: based on Anthropic public pricing (per-model aware)
- **Disk usage**: cache storage size per project
- **CSV export**: for further analysis

## Usage

```bash
# Full summary across all projects
python3 nanoclaw_cost.py /path/to/nanoclaw/data/sessions

# Drill into a specific project (shows per-session breakdown)
python3 nanoclaw_cost.py /path/to/nanoclaw/data/sessions --project discord_diffusion_objrel

# Sort by total tokens instead of cost
python3 nanoclaw_cost.py /path/to/nanoclaw/data/sessions --sort tokens

# Export to CSV
python3 nanoclaw_cost.py /path/to/nanoclaw/data/sessions --csv report.csv

# Skip disk size scan (faster)
python3 nanoclaw_cost.py /path/to/nanoclaw/data/sessions --no-size
```

## Output columns

| Column | Description |
|--------|-------------|
| API calls | Number of assistant turns (= API requests) |
| Input | Uncached input tokens |
| CacheW | Cache write tokens (1.25× input price) |
| CacheR | Cache read tokens (0.1× input price) |
| Output | Output tokens |
| Total | Sum of all token types |
| Cost | Estimated USD cost |
| Disk | Session directory size on disk |

## Pricing

Uses Anthropic public API pricing as of April 2025 ($/1M tokens):

| Model | Input | Cache Write | Cache Read | Output |
|-------|-------|-------------|------------|--------|
| claude-opus-4.x | $15 | $18.75 | $1.50 | $75 |
| claude-sonnet-4.x | $3 | $3.75 | $0.30 | $15 |
| claude-haiku-4.x | $0.80 | $1.00 | $0.08 | $4 |

Update `PRICING` dict in `nanoclaw_cost.py` to adjust.

## Requirements

Python 3.8+, no external dependencies.
