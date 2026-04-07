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
| CacheW | Cache write tokens (priced at 1.25× input) |
| CacheR | Cache read tokens (priced at 0.1× input — very cheap) |
| Output | Output tokens |
| In$ | Cost of all incoming tokens (input + cache write + cache read) |
| Out$ | Cost of output tokens |
| Total$ | Total estimated cost |
| sonnet | Cost attributed to claude-sonnet-* models |
| opus | Cost attributed to claude-opus-* models |
| haiku | Cost attributed to claude-haiku-* models |
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

Uses Anthropic public API pricing ($/1M tokens, as of April 2025):

| Model family | Input | Cache Write | Cache Read | Output |
|---|---|---|---|---|
| claude-opus-4.x | $15 | $18.75 | $1.50 | $75 |
| claude-sonnet-4.x | $3 | $3.75 | $0.30 | $15 |
| claude-haiku-4.x | $0.80 | $1.00 | $0.08 | $4 |

Update the `PRICING` dict at the top of `nanoclaw_cost.py` to adjust.

## Requirements

Python 3.8+. No external dependencies for the table/CSV output.
`matplotlib` required for `--plot`. `--html` has no extra dependencies (uses CDN-hosted Chart.js).
