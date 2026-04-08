#!/usr/bin/env python3
"""
nanoclaw_cost.py — Token usage & cost breakdown for NanoClaw Claude Code sessions.

Usage:
    python nanoclaw_cost.py /path/to/nanoclaw/data/sessions
    python nanoclaw_cost.py ... --sort cost
    python nanoclaw_cost.py ... --project discord_diffusion_objrel
    python nanoclaw_cost.py ... --csv out.csv
    python nanoclaw_cost.py ... --plot          # matplotlib PNGs
    python nanoclaw_cost.py ... --html          # interactive HTML (Chart.js, no server needed)
    python nanoclaw_cost.py ... --plot --html   # both
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil

# ── Pricing (per million tokens) ─────────────────────────────────────────────
# Source: https://platform.claude.com/docs/en/about-claude/pricing (2025-04-07)
#
# Tuple: (input, cache_write_5m, cache_write_1h, cache_read, output)
#   cache_write_5m  = 1.25× input  (5-minute TTL cache write)
#   cache_write_1h  = 2.00× input  (1-hour  TTL cache write)
#   cache_read      = 0.10× input  (cache hit / refresh)
#
# Matched by model-id prefix, longest match first.
PRICING: Dict[str, Tuple[float, float, float, float, float]] = {
    # ── Claude 4 series ───────────────────────────────────────────────────────
    # Opus 4.6 / 4.5  — $5 input (NOT $15; only 4.1 and 4.0 are $15)
    "claude-opus-4-6":        ( 5.00,  6.25, 10.00, 0.50, 25.00),
    "claude-opus-4-5":        ( 5.00,  6.25, 10.00, 0.50, 25.00),
    # Opus 4.1 / 4.0  — $15 input
    "claude-opus-4-1":        (15.00, 18.75, 30.00, 1.50, 75.00),
    "claude-opus-4-0":        (15.00, 18.75, 30.00, 1.50, 75.00),
    "claude-opus-4":          (15.00, 18.75, 30.00, 1.50, 75.00),
    # Sonnet 4.x  — $3 input
    "claude-sonnet-4":        ( 3.00,  3.75,  6.00, 0.30, 15.00),
    # Haiku 4.5  — $1 input
    "claude-haiku-4-5":       ( 1.00,  1.25,  2.00, 0.10,  5.00),
    "claude-haiku-4":         ( 1.00,  1.25,  2.00, 0.10,  5.00),
    # ── Claude 3.x series ────────────────────────────────────────────────────
    "claude-3-7-sonnet":      ( 3.00,  3.75,  6.00, 0.30, 15.00),
    "claude-3-5-sonnet":      ( 3.00,  3.75,  6.00, 0.30, 15.00),
    "claude-3-5-haiku":       ( 0.80,  1.00,  1.60, 0.08,  4.00),
    "claude-3-opus":          (15.00, 18.75, 30.00, 1.50, 75.00),
    "claude-3-haiku":         ( 0.25,  0.30,  0.50, 0.03,  1.25),
    "claude-3-sonnet":        ( 3.00,  3.75,  6.00, 0.30, 15.00),
}
# Fallback if model string not matched (assume sonnet-4 pricing)
DEFAULT_PRICING = (3.00, 3.75, 6.00, 0.30, 15.00)

# Human-readable model family labels
MODEL_FAMILIES = [
    ("opus",   ["claude-opus"]),
    ("sonnet", ["claude-sonnet", "claude-3-5-sonnet"]),
    ("haiku",  ["claude-haiku", "claude-3-haiku"]),
]


def get_price(model: str) -> Tuple[float, float, float, float, float]:
    """Return (input, cache_write_5m, cache_write_1h, cache_read, output) per MTok."""
    # Sort by prefix length descending so "claude-opus-4-6" matches before "claude-opus-4"
    for prefix in sorted(PRICING, key=len, reverse=True):
        if model.startswith(prefix):
            return PRICING[prefix]
    return DEFAULT_PRICING


def model_family(model: str) -> str:
    for fam, prefixes in MODEL_FAMILIES:
        for p in prefixes:
            if model.startswith(p):
                return fam
    return model or "unknown"


def tokens_to_cost(inp, cw_5m, cw_1h, cr, out, model) -> float:
    p_in, p_cw5, p_cw1, p_cr, p_out = get_price(model)
    return (inp * p_in + cw_5m * p_cw5 + cw_1h * p_cw1 +
            cr * p_cr + out * p_out) / 1_000_000


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ApiCall:
    """One assistant turn."""
    ts: datetime
    model: str
    input_tokens: int
    cache_write_5m_tokens: int   # ephemeral 5-min cache writes (1.25× input)
    cache_write_1h_tokens: int   # ephemeral 1-hr  cache writes (2.00× input)
    cache_read_tokens: int
    output_tokens: int

    @property
    def cache_write_tokens(self) -> int:
        return self.cache_write_5m_tokens + self.cache_write_1h_tokens

    @property
    def cost(self) -> float:
        return tokens_to_cost(
            self.input_tokens, self.cache_write_5m_tokens, self.cache_write_1h_tokens,
            self.cache_read_tokens, self.output_tokens, self.model
        )

    @property
    def incoming_cost(self) -> float:
        p_in, p_cw5, p_cw1, p_cr, _ = get_price(self.model)
        return (self.input_tokens * p_in +
                self.cache_write_5m_tokens * p_cw5 +
                self.cache_write_1h_tokens * p_cw1 +
                self.cache_read_tokens * p_cr) / 1_000_000

    @property
    def outgoing_cost(self) -> float:
        _, _, _, _, p_out = get_price(self.model)
        return self.output_tokens * p_out / 1_000_000


@dataclass
class SessionStats:
    session_id: str
    project: str
    jsonl_path: str
    is_subagent: bool = False
    calls: List[ApiCall] = field(default_factory=list)
    tool_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # ── aggregates (computed lazily) ──
    def _agg(self, fam=None):
        calls = [c for c in self.calls if (fam is None or model_family(c.model) == fam)]
        return calls

    def input_tokens(self, fam=None):
        return sum(c.input_tokens for c in self._agg(fam))

    def cache_write_tokens(self, fam=None):
        return sum(c.cache_write_tokens for c in self._agg(fam))

    def cache_write_5m_tokens(self, fam=None):
        return sum(c.cache_write_5m_tokens for c in self._agg(fam))

    def cache_write_1h_tokens(self, fam=None):
        return sum(c.cache_write_1h_tokens for c in self._agg(fam))

    def cache_read_tokens(self, fam=None):
        return sum(c.cache_read_tokens for c in self._agg(fam))

    def output_tokens(self, fam=None):
        return sum(c.output_tokens for c in self._agg(fam))

    def total_tokens(self, fam=None):
        c = self._agg(fam)
        return sum(x.input_tokens + x.cache_write_tokens + x.cache_read_tokens + x.output_tokens for x in c)

    def api_calls(self, fam=None):
        return len(self._agg(fam))

    def cost(self, fam=None):
        return sum(c.cost for c in self._agg(fam))

    def incoming_cost(self, fam=None):
        return sum(c.incoming_cost for c in self._agg(fam))

    def outgoing_cost(self, fam=None):
        return sum(c.outgoing_cost for c in self._agg(fam))

    def models_used(self):
        seen = {}
        for c in self.calls:
            seen[c.model] = seen.get(c.model, 0) + 1
        return seen

    def dominant_model(self):
        m = self.models_used()
        return max(m, key=m.get) if m else ""

    def total_tool_calls(self) -> int:
        return sum(self.tool_counts.values())

    def web_search_cost(self) -> float:
        """Estimate cost of web_search tool calls at $10/1000 searches."""
        n = self.tool_counts.get("web_search", 0) + self.tool_counts.get("WebSearch", 0)
        return n * 10.0 / 1000.0


def parse_jsonl(path: str) -> SessionStats:
    p = Path(path)
    parts = p.parts
    try:
        si = parts.index("sessions")
        project = parts[si + 1]
    except (ValueError, IndexError):
        project = p.parent.parent.name

    is_subagent = "subagents" in str(path)
    stats = SessionStats(
        session_id=p.stem,
        project=project,
        jsonl_path=str(path),
        is_subagent=is_subagent,
    )

    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") != "assistant":
                    continue
                msg = obj.get("message", {})
                usage = msg.get("usage")
                if not usage:
                    continue
                ts_str = obj.get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except Exception:
                    ts = datetime.now(timezone.utc)

                # The JSONL stores 5m vs 1h cache writes separately under
                # usage.cache_creation.ephemeral_5m/1h_input_tokens.
                # Fall back to splitting the total evenly if not present.
                cache_creation = usage.get("cache_creation", {})
                cw_5m = cache_creation.get("ephemeral_5m_input_tokens", None)
                cw_1h = cache_creation.get("ephemeral_1h_input_tokens", None)
                cw_total = usage.get("cache_creation_input_tokens", 0)
                if cw_5m is None and cw_1h is None:
                    # older format — assume all writes are 1h (conservative)
                    cw_5m, cw_1h = 0, cw_total
                elif cw_5m is None:
                    cw_5m = cw_total - cw_1h
                elif cw_1h is None:
                    cw_1h = cw_total - cw_5m

                stats.calls.append(ApiCall(
                    ts=ts,
                    model=msg.get("model", ""),
                    input_tokens=usage.get("input_tokens", 0),
                    cache_write_5m_tokens=max(0, cw_5m),
                    cache_write_1h_tokens=max(0, cw_1h),
                    cache_read_tokens=usage.get("cache_read_input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                ))

                # Count tool_use blocks in this assistant turn
                for block in msg.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        name = block.get("name", "unknown")
                        stats.tool_counts[name] += 1
    except (OSError, PermissionError):
        pass

    return stats


def dir_size_mb(path: str) -> float:
    total = 0
    try:
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    except OSError:
        pass
    return total / (1024 * 1024)


# ── Formatting helpers ─────────────────────────────────────────────────────────

def fmt_tok(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def fmt_cost(c: float) -> str:
    if c >= 1.0:
        return f"${c:.3f}"
    return f"${c:.4f}"


# ── Aggregation helper ─────────────────────────────────────────────────────────

def aggregate(sessions: List[SessionStats]) -> SessionStats:
    agg = SessionStats(session_id="(total)", project="", jsonl_path="")
    for s in sessions:
        agg.calls.extend(s.calls)
        for tool, cnt in s.tool_counts.items():
            agg.tool_counts[tool] += cnt
    return agg


# ── Matplotlib plotting ────────────────────────────────────────────────────────

def _bin_calls_by_time(all_calls, bin_minutes=60):
    """Bucket calls into time bins of `bin_minutes` width.
    Returns (bin_datetimes, costs_per_bin) sorted by time."""
    if not all_calls:
        return [], []
    from datetime import timedelta
    bin_td = timedelta(minutes=bin_minutes)
    t0 = min(c.ts for c in all_calls)
    # floor t0 to bin boundary
    epoch = datetime(t0.year, t0.month, t0.day, tzinfo=t0.tzinfo)
    offset = int((t0 - epoch).total_seconds() // (bin_minutes * 60))
    t0_floor = epoch + offset * bin_td

    bins: Dict[int, float] = defaultdict(float)
    for c in all_calls:
        idx = int((c.ts - t0_floor).total_seconds() // (bin_minutes * 60))
        bins[idx] += c.cost

    max_idx = max(bins.keys())
    xs = [t0_floor + i * bin_td for i in range(max_idx + 1)]
    ys = [bins.get(i, 0.0) for i in range(max_idx + 1)]
    return xs, ys


def plot_matplotlib(project_sessions: Dict[str, List[SessionStats]],
                    out_dir: str = ".",
                    mode: str = "cumulative"):
    """
    mode='cumulative'  — original step-line of cumulative cost (default)
    mode='per-call'    — bar chart of cost per time bin (hourly by default)
    mode='both'        — side-by-side: cumulative left, per-call right
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping PNG plots", file=sys.stderr)
        return

    projects = list(project_sessions.keys())
    n = len(projects)

    # layout: each project gets one row; columns depend on mode
    ncols = 2 if mode == "both" else 1
    fig, axes = plt.subplots(n, ncols, figsize=(7 * ncols, 3.5 * n), squeeze=False)
    title = {
        "cumulative": "Cumulative API Cost Over Time — per Project",
        "per-call":   "Cost per Hour — per Project",
        "both":       "API Cost: Cumulative & Per-Hour — per Project",
    }.get(mode, "NanoClaw API Cost")
    fig.suptitle(title, fontsize=13, y=1.01)

    fam_colors = {"opus": "#c0392b", "sonnet": "#2980b9", "haiku": "#27ae60", "unknown": "#888"}

    for idx, proj in enumerate(projects):
        all_calls = []
        for s in project_sessions[proj]:
            all_calls.extend(s.calls)
        all_calls.sort(key=lambda c: c.ts)

        fams = sorted({model_family(c.model) for c in all_calls})

        # ── cumulative panel ──────────────────────────────────────────────────
        if mode in ("cumulative", "both"):
            ax = axes[idx][0]
            if not all_calls:
                ax.set_visible(False)
            else:
                for fam in fams:
                    fc = [c for c in all_calls if model_family(c.model) == fam]
                    ts = [c.ts for c in fc]
                    costs = np.cumsum([c.cost for c in fc])
                    ax.step(ts, costs, where="post", label=fam,
                            color=fam_colors.get(fam, "#888"), linewidth=1.8)
                ts_all = [c.ts for c in all_calls]
                costs_all = np.cumsum([c.cost for c in all_calls])
                ax.step(ts_all, costs_all, where="post", label="total",
                        color="black", linewidth=2.2, linestyle="--", alpha=0.7)
                ax.set_title(proj, fontsize=9)
                ax.set_ylabel("Cumulative cost (USD)")
                loc = mdates.AutoDateLocator()
                ax.xaxis.set_major_locator(loc)
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

        # ── per-call (hourly bar) panel ────────────────────────────────────────
        if mode in ("per-call", "both"):
            col = 1 if mode == "both" else 0
            ax = axes[idx][col]
            if not all_calls:
                ax.set_visible(False)
            else:
                xs, ys = _bin_calls_by_time(all_calls, bin_minutes=60)
                bar_width = (xs[1] - xs[0]).total_seconds() / 86400 * 0.8 if len(xs) > 1 else 0.03
                ax.bar(xs, ys, width=bar_width, color="#2980b9", alpha=0.75, label="cost/hr")
                # overlay incoming vs outgoing as stacked bars
                xs2, ys_in = _bin_calls_by_time(
                    [type('_', (), {'ts': c.ts, 'cost': c.incoming_cost})() for c in all_calls],
                    bin_minutes=60)
                _, ys_out = _bin_calls_by_time(
                    [type('_', (), {'ts': c.ts, 'cost': c.outgoing_cost})() for c in all_calls],
                    bin_minutes=60)
                if xs2:
                    ax.bar(xs2, ys_in,  width=bar_width, color="#2980b9", alpha=0.8, label="incoming")
                    ax.bar(xs2, ys_out, width=bar_width, color="#c0392b", alpha=0.8,
                           bottom=ys_in, label="outgoing")
                ax.set_title(proj + " (hourly)", fontsize=9)
                ax.set_ylabel("Cost per hour (USD)")
                loc = mdates.AutoDateLocator()
                ax.xaxis.set_major_locator(loc)
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "nanoclaw_cost_timeline.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")


# ── HTML / Chart.js plotting ───────────────────────────────────────────────────

def plot_html(project_sessions: Dict[str, List[SessionStats]],
              out_dir: str = "."):
    import json as _json

    projects = list(project_sessions.keys())
    fam_colors = {
        "opus":    "rgba(192,57,43,",
        "sonnet":  "rgba(41,128,185,",
        "haiku":   "rgba(39,174,96,",
        "unknown": "rgba(136,136,136,",
    }

    # Build per-project datasets
    proj_data = {}
    for proj in projects:
        all_calls = []
        for s in project_sessions[proj]:
            all_calls.extend(s.calls)
        all_calls.sort(key=lambda c: c.ts)

        fams = sorted({model_family(c.model) for c in all_calls})
        datasets = []

        # Cumulative total
        total_ts, total_cum = [], []
        cum = 0.0
        for c in all_calls:
            cum += c.cost
            total_ts.append(c.ts.isoformat())
            total_cum.append(round(cum, 5))

        datasets.append({
            "label": "total",
            "data": [{"x": t, "y": v} for t, v in zip(total_ts, total_cum)],
            "borderColor": "rgba(0,0,0,0.7)",
            "borderWidth": 2.5,
            "borderDash": [6, 3],
            "pointRadius": 0,
            "stepped": "after",
            "fill": False,
        })

        # Per-family
        for fam in fams:
            fc = [c for c in all_calls if model_family(c.model) == fam]
            cum_f = 0.0
            pts = []
            for c in fc:
                cum_f += c.cost
                pts.append({"x": c.ts.isoformat(), "y": round(cum_f, 5)})
            color_base = fam_colors.get(fam, "rgba(136,136,136,")
            datasets.append({
                "label": fam,
                "data": pts,
                "borderColor": color_base + "0.9)",
                "backgroundColor": color_base + "0.1)",
                "borderWidth": 1.8,
                "pointRadius": 0,
                "stepped": "after",
                "fill": False,
            })

        # incoming vs outgoing cost (non-cumulative per-day bar — separate chart)
        # Daily cost breakdown
        day_map: Dict[str, Dict[str, float]] = defaultdict(lambda: {"incoming": 0.0, "outgoing": 0.0})
        for c in all_calls:
            day = c.ts.date().isoformat()
            day_map[day]["incoming"] += c.incoming_cost
            day_map[day]["outgoing"] += c.outgoing_cost
        days = sorted(day_map.keys())
        bar_datasets = [
            {
                "label": "incoming (input+cache)",
                "data": [round(day_map[d]["incoming"], 5) for d in days],
                "backgroundColor": "rgba(41,128,185,0.7)",
            },
            {
                "label": "outgoing (output)",
                "data": [round(day_map[d]["outgoing"], 5) for d in days],
                "backgroundColor": "rgba(192,57,43,0.7)",
            },
        ]

        proj_data[proj] = {
            "line_datasets": datasets,
            "bar_labels": days,
            "bar_datasets": bar_datasets,
            "total_cost": round(sum(c.cost for c in all_calls), 4),
            "n_calls": len(all_calls),
        }

    # Build HTML
    panels = []
    for proj in projects:
        d = proj_data[proj]
        line_id = f"line_{proj.replace('-','_')}"
        bar_id  = f"bar_{proj.replace('-','_')}"
        panels.append(f"""
        <div class="panel">
          <h3>{proj}
            <span class="badge">{d['n_calls']} calls · ${d['total_cost']:.3f}</span>
          </h3>
          <div class="charts-row">
            <div class="chart-wrap">
              <canvas id="{line_id}"></canvas>
            </div>
            <div class="chart-wrap">
              <canvas id="{bar_id}"></canvas>
            </div>
          </div>
        </div>
        <script>
        (function() {{
          var lineCtx = document.getElementById('{line_id}').getContext('2d');
          new Chart(lineCtx, {{
            type: 'line',
            data: {{ datasets: {_json.dumps(d['line_datasets'])} }},
            options: {{
              animation: false,
              scales: {{
                x: {{ type: 'time', time: {{ unit: 'day' }},
                       ticks: {{ maxRotation: 30 }} }},
                y: {{ title: {{ display: true, text: 'Cumulative cost (USD)' }} }}
              }},
              plugins: {{
                title: {{ display: true, text: 'Cumulative cost over time' }},
                legend: {{ position: 'top', labels: {{ boxWidth: 12 }} }}
              }}
            }}
          }});

          var barCtx = document.getElementById('{bar_id}').getContext('2d');
          new Chart(barCtx, {{
            type: 'bar',
            data: {{
              labels: {_json.dumps(d['bar_labels'])},
              datasets: {_json.dumps(d['bar_datasets'])}
            }},
            options: {{
              animation: false,
              scales: {{
                x: {{ stacked: true, ticks: {{ maxRotation: 30 }} }},
                y: {{ stacked: true,
                       title: {{ display: true, text: 'Daily cost (USD)' }} }}
              }},
              plugins: {{
                title: {{ display: true, text: 'Daily cost: incoming vs outgoing' }},
                legend: {{ position: 'top', labels: {{ boxWidth: 12 }} }}
              }}
            }}
          }});
        }})();
        </script>
        """)

    html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>NanoClaw API Cost Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
  <style>
    body { font-family: system-ui, sans-serif; background: #f5f5f5; margin: 0; padding: 1rem; }
    h1 { text-align: center; color: #333; }
    .panel { background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.12);
             margin: 1rem auto; max-width: 1100px; padding: 1rem 1.5rem; }
    h3 { margin: 0 0 .5rem; color: #222; font-size: 1rem; }
    .badge { font-size: .75rem; color: #666; font-weight: normal; margin-left: .5rem; }
    .charts-row { display: flex; gap: 1rem; }
    .chart-wrap { flex: 1; min-width: 0; }
    canvas { max-height: 260px; }
  </style>
</head>
<body>
  <h1>NanoClaw API Cost Dashboard</h1>
  """ + "\n".join(panels) + """
</body>
</html>"""

    out_path = os.path.join(out_dir, "nanoclaw_cost_dashboard.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Dashboard saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NanoClaw token usage & cost breakdown")
    parser.add_argument("sessions_dir", help="Path to nanoclaw/data/sessions/")
    parser.add_argument("--project", help="Filter to specific project (shows per-session detail)")
    parser.add_argument("--sort", choices=["cost", "tokens", "calls", "project"],
                        default="cost")
    parser.add_argument("--csv", help="Write results to CSV")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib PNG")
    parser.add_argument("--plot-mode", choices=["cumulative", "per-call", "both"],
                        default="cumulative",
                        help="Plot style: cumulative (default), per-call (hourly bars), both")
    parser.add_argument("--html", action="store_true",
                        help="Generate interactive HTML dashboard (Chart.js)")
    parser.add_argument("--out-dir", default=".", help="Output directory for plots/html")
    parser.add_argument("--no-size", action="store_true", help="Skip disk size scan")
    args = parser.parse_args()

    sessions_root = Path(args.sessions_dir)
    if not sessions_root.exists():
        print(f"ERROR: {sessions_root} does not exist", file=sys.stderr)
        sys.exit(1)

    # Discover & parse JSONL files
    jsonl_files = sorted(sessions_root.rglob("*.jsonl"))
    jsonl_files = [p for p in jsonl_files if "mcp-logs" not in str(p)]
    if args.project:
        jsonl_files = [p for p in jsonl_files if args.project in str(p)]

    all_sessions = [parse_jsonl(str(p)) for p in jsonl_files]

    # Group by project
    project_sessions: Dict[str, List[SessionStats]] = defaultdict(list)
    for s in all_sessions:
        project_sessions[s.project].append(s)

    # Aggregate per project
    project_aggs: Dict[str, SessionStats] = {
        proj: aggregate(sessions)
        for proj, sessions in project_sessions.items()
    }

    # Disk sizes
    disk_mb: Dict[str, float] = {}
    if not args.no_size:
        for proj in project_aggs:
            disk_mb[proj] = dir_size_mb(str(sessions_root / proj))

    # Sort
    projects = list(project_aggs.keys())
    if args.sort == "cost":
        projects.sort(key=lambda p: project_aggs[p].cost(), reverse=True)
    elif args.sort == "tokens":
        projects.sort(key=lambda p: project_aggs[p].total_tokens(), reverse=True)
    elif args.sort == "calls":
        projects.sort(key=lambda p: project_aggs[p].api_calls(), reverse=True)
    elif args.sort == "project":
        projects.sort()

    # ── Print table ────────────────────────────────────────────────────────────
    W = min(shutil.get_terminal_size((140, 40)).columns, 160)
    sep = "─" * W

    print(f"\n{'NanoClaw Token Usage & Cost Report':^{W}}")
    print(f"{'Sessions: ' + str(sessions_root):^{W}}")
    print(sep)

    hdr = (f"{'Project':<32} {'calls':>6}  "
           f"{'Input':>7} {'CW-5m':>7} {'CW-1h':>7} {'CacheR':>7} {'Output':>7}  "
           f"{'In$':>8} {'Out$':>8} {'Total$':>9}  "
           f"{'sonnet':>7} {'opus':>7} {'haiku':>6}  "
           f"{'Tools':>6} {'WebSrch':>7} {'WS$':>6}")
    if not args.no_size:
        hdr += f"  {'Disk':>7}"
    print(hdr)
    print(sep)

    grand = SessionStats(session_id="TOTAL", project="", jsonl_path="")

    for proj in projects:
        agg = project_aggs[proj]
        grand.calls.extend(agg.calls)
        for tool, cnt in agg.tool_counts.items():
            grand.tool_counts[tool] += cnt
        disk = disk_mb.get(proj, 0.0)

        ws_n = agg.tool_counts.get("web_search", 0) + agg.tool_counts.get("WebSearch", 0)
        row = (
            f"{proj:<32} {agg.api_calls():>6,}  "
            f"{fmt_tok(agg.input_tokens()):>7} "
            f"{fmt_tok(agg.cache_write_5m_tokens()):>7} "
            f"{fmt_tok(agg.cache_write_1h_tokens()):>7} "
            f"{fmt_tok(agg.cache_read_tokens()):>7} "
            f"{fmt_tok(agg.output_tokens()):>7}  "
            f"{fmt_cost(agg.incoming_cost()):>8} "
            f"{fmt_cost(agg.outgoing_cost()):>8} "
            f"{fmt_cost(agg.cost()):>9}  "
            f"{fmt_cost(agg.cost('sonnet')):>7} "
            f"{fmt_cost(agg.cost('opus')):>7} "
            f"{fmt_cost(agg.cost('haiku')):>6}  "
            f"{agg.total_tool_calls():>6,} "
            f"{ws_n:>7,} "
            f"{fmt_cost(agg.web_search_cost()):>6}"
        )
        if not args.no_size:
            row += f"  {disk:>6.1f}MB"
        print(row)

    print(sep)
    ws_g = grand.tool_counts.get("web_search", 0) + grand.tool_counts.get("WebSearch", 0)
    row_g = (
        f"{'TOTAL':<32} {grand.api_calls():>6,}  "
        f"{fmt_tok(grand.input_tokens()):>7} "
        f"{fmt_tok(grand.cache_write_5m_tokens()):>7} "
        f"{fmt_tok(grand.cache_write_1h_tokens()):>7} "
        f"{fmt_tok(grand.cache_read_tokens()):>7} "
        f"{fmt_tok(grand.output_tokens()):>7}  "
        f"{fmt_cost(grand.incoming_cost()):>8} "
        f"{fmt_cost(grand.outgoing_cost()):>8} "
        f"{fmt_cost(grand.cost()):>9}  "
        f"{fmt_cost(grand.cost('sonnet')):>7} "
        f"{fmt_cost(grand.cost('opus')):>7} "
        f"{fmt_cost(grand.cost('haiku')):>6}  "
        f"{grand.total_tool_calls():>6,} "
        f"{ws_g:>7,} "
        f"{fmt_cost(grand.web_search_cost()):>6}"
    )
    if not args.no_size:
        row_g += f"  {sum(disk_mb.values()):>6.1f}MB"
    print(row_g)
    print()

    # ── Top tool calls breakdown ────────────────────────────────────────────────
    if grand.tool_counts:
        top_tools = sorted(grand.tool_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        print("Top tool calls (all projects):")
        for tool, cnt in top_tools:
            bar = "█" * min(40, int(cnt / max(1, top_tools[0][1]) * 40))
            print(f"  {tool:<40} {cnt:>6,}  {bar}")
        print()

    # ── Per-session detail (when --project) ────────────────────────────────────
    if args.project:
        for proj, sessions in project_sessions.items():
            agg_p = project_aggs[proj]
            print(f"\n── Per-session: {proj} ──")
            print(f"{'Session':<40} {'sub':>3} {'calls':>6}  {'In$':>8} {'Out$':>8} {'Total$':>9}  {'Model'}")
            print("─" * 90)
            sessions_sorted = sorted(sessions, key=lambda s: s.cost(), reverse=True)
            for s in sessions_sorted:
                dm = s.dominant_model()
                print(
                    f"{s.session_id:<40} {'Y' if s.is_subagent else 'N':>3} "
                    f"{s.api_calls():>6,}  "
                    f"{fmt_cost(s.incoming_cost()):>8} "
                    f"{fmt_cost(s.outgoing_cost()):>8} "
                    f"{fmt_cost(s.cost()):>9}  "
                    f"{dm}"
                )
            if agg_p.tool_counts:
                print(f"\n  Tool call breakdown ({agg_p.total_tool_calls()} total):")
                top = sorted(agg_p.tool_counts.items(), key=lambda x: x[1], reverse=True)
                for tool, cnt in top:
                    print(f"    {tool:<40} {cnt:>5,}")

    # ── CSV ────────────────────────────────────────────────────────────────────
    if args.csv:
        import csv
        rows = []
        for proj in projects:
            agg = project_aggs[proj]
            ws_n = agg.tool_counts.get("web_search", 0) + agg.tool_counts.get("WebSearch", 0)
            rows.append({
                "project": proj,
                "api_calls": agg.api_calls(),
                "input_tokens": agg.input_tokens(),
                "cache_write_5m_tokens": agg.cache_write_5m_tokens(),
                "cache_write_1h_tokens": agg.cache_write_1h_tokens(),
                "cache_read_tokens": agg.cache_read_tokens(),
                "output_tokens": agg.output_tokens(),
                "total_tokens": agg.total_tokens(),
                "incoming_cost_usd": round(agg.incoming_cost(), 5),
                "outgoing_cost_usd": round(agg.outgoing_cost(), 5),
                "total_cost_usd": round(agg.cost(), 5),
                "sonnet_cost_usd": round(agg.cost("sonnet"), 5),
                "opus_cost_usd": round(agg.cost("opus"), 5),
                "haiku_cost_usd": round(agg.cost("haiku"), 5),
                "dominant_model": agg.dominant_model(),
                "total_tool_calls": agg.total_tool_calls(),
                "web_search_calls": ws_n,
                "web_search_cost_usd": round(agg.web_search_cost(), 5),
                "disk_mb": round(disk_mb.get(proj, 0.0), 2),
            })
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV: {args.csv}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    filt = {p: project_sessions[p] for p in projects}  # keep sort order

    if args.plot:
        plot_matplotlib(filt, args.out_dir, mode=args.plot_mode)

    if args.html:
        plot_html(filt, args.out_dir)


if __name__ == "__main__":
    main()
