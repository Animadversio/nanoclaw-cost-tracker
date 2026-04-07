# nanoclaw-cost-tracker

Token usage & API cost breakdown for [NanoClaw](https://github.com/qwibitai/nanoclaw) Claude Code sessions.

Parses Claude Code session `.jsonl` files and reports per-project token counts, estimated API cost (broken down by incoming vs outgoing, and by model family), disk usage, and optional time-series plots.

## Quick setup

Add to your `~/.bashrc`:

```bash
NANOCLAW_SESSIONS="/path/to/nanoclaw/data/sessions"
NANOCLAW_COST_SCRIPT="/path/to/Github/nanoclaw-cost-tracker/nanoclaw_cost.py"
alias nanoclaw-cost="python3 $NANOCLAW_COST_SCRIPT $NANOCLAW_SESSIONS"
alias nanoclaw-cost-plot="python3 $NANOCLAW_COST_SCRIPT $NANOCLAW_SESSIONS --plot --html --out-dir /tmp"
```

Then `source ~/.bashrc`.

## Usage

```bash
# Full summary table across all projects (sorted by cost)
nanoclaw-cost

# Sort by total tokens or API call count instead
nanoclaw-cost --sort tokens
nanoclaw-cost --sort calls
nanoclaw-cost --sort project

# Drill into one project — shows per-session breakdown
nanoclaw-cost --project discord_diffusion_objrel

# Export summary to CSV
nanoclaw-cost --csv ~/cost_report.csv

# Generate matplotlib PNG timeline + interactive HTML dashboard
nanoclaw-cost-plot
# Outputs:
#   /tmp/nanoclaw_cost_timeline.png       — cumulative cost over time per project
#   /tmp/nanoclaw_cost_dashboard.html     — interactive Chart.js (open in browser)

# Plots to a custom directory
nanoclaw-cost --plot --html --out-dir ~/reports

# Skip disk size scan (faster, omits Disk column)
nanoclaw-cost --no-size

# Combine flags freely
nanoclaw-cost --project discord_diffusion_objrel --sort tokens --csv out.csv --no-size
```

Or call the script directly without aliases:

```bash
python3 ~/Github/nanoclaw-cost-tracker/nanoclaw_cost.py ~/nanoclaw/data/sessions [options]
```

## Output columns

| Column | Description |
|--------|-------------|
| calls | Number of assistant turns (= API requests made) |
| Input | Raw uncached input tokens |
| CW-5m | Cache write tokens with 5-min TTL (priced at 1.25× input) |
| CW-1h | Cache write tokens with 1-hour TTL (priced at 2× input) |
| CacheR | Cache read tokens (priced at 0.1× input — very cheap) |
| Output | Output tokens |
| In$ | Cost of all incoming tokens (input + cache writes + cache read) |
| Out$ | Cost of output tokens |
| Total$ | Total estimated cost |
| sonnet | Cost attributed to claude-sonnet-* models |
| opus | Cost attributed to claude-opus-* models |
| haiku | Cost attributed to claude-haiku-* models |
| Tools | Total tool call invocations |
| WebSrch | Number of web_search tool calls |
| WS$ | Estimated web search cost ($10/1000 searches) |
| Disk | Session directory size on disk |

## Plots

*`--plot`* generates `nanoclaw_cost_timeline.png` with one panel per project:
- Left: cumulative cost over time, split by model family (sonnet / opus / haiku) + total
- Right: daily stacked bar — incoming (blue) vs outgoing (red) cost

X-axis adapts to the session time span:
- `< 3 hours` → `HH:MM`
- `< 2 days` → `mm/dd HH:MM`
- `< 2 weeks` → `mm/dd`
- longer → auto

*`--html`* generates `nanoclaw_cost_dashboard.html` — same charts as interactive Chart.js. Open in any browser, no server needed.

## Pricing

Source: https://platform.claude.com/docs/en/about-claude/pricing (updated 2025-04-07)

Cache writes have two tiers: 5-minute TTL (1.25× input) and 1-hour TTL (2× input).
The JSONL stores these separately and both are tracked accurately.

| Model | Input | Cache Write 5m | Cache Write 1h | Cache Read | Output |
|---|---|---|---|---|---|
| Claude Opus 4.6 / 4.5 | $5 | $6.25 | $10 | $0.50 | $25 |
| Claude Opus 4.1 / 4.0 | $15 | $18.75 | $30 | $1.50 | $75 |
| Claude Sonnet 4.x / 3.7 | $3 | $3.75 | $6 | $0.30 | $15 |
| Claude Haiku 4.5 | $1 | $1.25 | $2 | $0.10 | $5 |
| Claude Haiku 3.5 | $0.80 | $1.00 | $1.60 | $0.08 | $4 |
| Claude Opus 3 | $15 | $18.75 | $30 | $1.50 | $75 |
| Claude Haiku 3 | $0.25 | $0.30 | $0.50 | $0.03 | $1.25 |

> Note: Opus 4.6/4.5 are $5/MTok input — significantly cheaper than Opus 4.1/4.0 ($15/MTok).

Update the `PRICING` dict at the top of `nanoclaw_cost.py` to adjust.

## Requirements

Python 3.8+. No external dependencies for the table/CSV output.
`matplotlib` required for `--plot`. `--html` has no extra dependencies (uses CDN-hosted Chart.js).
