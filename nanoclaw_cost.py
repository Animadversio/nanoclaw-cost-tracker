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

# ── Pricing (per million tokens, Anthropic public pricing 2025-04) ────────────
# (input, cache_write, cache_read, output) per 1M tokens
PRICING: Dict[str, Tuple[float, float, float, float]] = {
    "claude-opus-4":      (15.00, 18.75, 1.50, 75.00),
    "claude-opus-4-5":    (15.00, 18.75, 1.50, 75.00),
    "claude-sonnet-4":    ( 3.00,  3.75, 0.30, 15.00),
    "claude-sonnet-4-6":  ( 3.00,  3.75, 0.30, 15.00),
    "claude-haiku-4":     ( 0.80,  1.00, 0.08,  4.00),
    "claude-haiku-4-5":   ( 0.80,  1.00, 0.08,  4.00),
    # legacy
    "claude-3-5-sonnet":  ( 3.00,  3.75, 0.30, 15.00),
    "claude-3-opus":      (15.00, 18.75, 1.50, 75.00),
    "claude-3-haiku":     ( 0.25,  0.30, 0.03,  1.25),
}
DEFAULT_PRICING = (3.00, 3.75, 0.30, 15.00)

# Human-readable model family labels
MODEL_FAMILIES = [
    ("opus",   ["claude-opus"]),
    ("sonnet", ["claude-sonnet", "claude-3-5-sonnet"]),
    ("haiku",  ["claude-haiku", "claude-3-haiku"]),
]


def get_price(model: str) -> Tuple[float, float, float, float]:
    for prefix, prices in PRICING.items():
        if model.startswith(prefix):
            return prices
    return DEFAULT_PRICING


def model_family(model: str) -> str:
    for fam, prefixes in MODEL_FAMILIES:
        for p in prefixes:
            if model.startswith(p):
                return fam
    return model or "unknown"


def tokens_to_cost(inp, cw, cr, out, model) -> float:
    p_in, p_cw, p_cr, p_out = get_price(model)
    return (inp * p_in + cw * p_cw + cr * p_cr + out * p_out) / 1_000_000


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ApiCall:
    """One assistant turn."""
    ts: datetime
    model: str
    input_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    output_tokens: int

    @property
    def cost(self) -> float:
        return tokens_to_cost(
            self.input_tokens, self.cache_write_tokens,
            self.cache_read_tokens, self.output_tokens, self.model
        )

    @property
    def incoming_cost(self) -> float:
        p_in, p_cw, p_cr, _ = get_price(self.model)
        return (self.input_tokens * p_in + self.cache_write_tokens * p_cw +
                self.cache_read_tokens * p_cr) / 1_000_000

    @property
    def outgoing_cost(self) -> float:
        _, _, _, p_out = get_price(self.model)
        return self.output_tokens * p_out / 1_000_000


@dataclass
class SessionStats:
    session_id: str
    project: str
    jsonl_path: str
    is_subagent: bool = False
    calls: List[ApiCall] = field(default_factory=list)

    # ── aggregates (computed lazily) ──
    def _agg(self, fam=None):
        calls = [c for c in self.calls if (fam is None or model_family(c.model) == fam)]
        return calls

    def input_tokens(self, fam=None):
        return sum(c.input_tokens for c in self._agg(fam))

    def cache_write_tokens(self, fam=None):
        return sum(c.cache_write_tokens for c in self._agg(fam))

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

                stats.calls.append(ApiCall(
                    ts=ts,
                    model=msg.get("model", ""),
                    input_tokens=usage.get("input_tokens", 0),
                    cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
                    cache_read_tokens=usage.get("cache_read_input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                ))
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
    return agg


# ── Matplotlib plotting ────────────────────────────────────────────────────────

def plot_matplotlib(project_sessions: Dict[str, List[SessionStats]],
                    out_dir: str = "."):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping PNG plots", file=sys.stderr)
        return

    projects = list(project_sessions.keys())
    n = len(projects)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), squeeze=False)
    fig.suptitle("Cumulative API Cost Over Time — per Project", fontsize=13, y=1.01)

    fam_colors = {"opus": "#c0392b", "sonnet": "#2980b9", "haiku": "#27ae60", "unknown": "#888"}

    for idx, proj in enumerate(projects):
        ax = axes[idx // cols][idx % cols]
        all_calls = []
        for s in project_sessions[proj]:
            all_calls.extend(s.calls)
        if not all_calls:
            ax.set_visible(False)
            continue

        all_calls.sort(key=lambda c: c.ts)

        # Split by model family and cumulative cost
        fams = list({model_family(c.model) for c in all_calls})
        fams.sort()

        for fam in fams:
            fc = [c for c in all_calls if model_family(c.model) == fam]
            ts = [c.ts for c in fc]
            costs = np.cumsum([c.cost for c in fc])
            ax.step(ts, costs, where="post", label=fam,
                    color=fam_colors.get(fam, "#888"), linewidth=1.8)

        # Total cumulative
        ts_all = [c.ts for c in all_calls]
        costs_all = np.cumsum([c.cost for c in all_calls])
        ax.step(ts_all, costs_all, where="post", label="total",
                color="black", linewidth=2.2, linestyle="--", alpha=0.7)

        ax.set_title(proj, fontsize=9)
        ax.set_ylabel("Cumulative cost (USD)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

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
           f"{'Input':>7} {'CacheW':>7} {'CacheR':>7} {'Output':>7}  "
           f"{'In$':>8} {'Out$':>8} {'Total$':>9}  "
           f"{'sonnet':>7} {'opus':>7} {'haiku':>6}")
    if not args.no_size:
        hdr += f"  {'Disk':>7}"
    print(hdr)
    print(sep)

    grand = SessionStats(session_id="TOTAL", project="", jsonl_path="")

    for proj in projects:
        agg = project_aggs[proj]
        grand.calls.extend(agg.calls)
        disk = disk_mb.get(proj, 0.0)

        row = (
            f"{proj:<32} {agg.api_calls():>6,}  "
            f"{fmt_tok(agg.input_tokens()):>7} "
            f"{fmt_tok(agg.cache_write_tokens()):>7} "
            f"{fmt_tok(agg.cache_read_tokens()):>7} "
            f"{fmt_tok(agg.output_tokens()):>7}  "
            f"{fmt_cost(agg.incoming_cost()):>8} "
            f"{fmt_cost(agg.outgoing_cost()):>8} "
            f"{fmt_cost(agg.cost()):>9}  "
            f"{fmt_cost(agg.cost('sonnet')):>7} "
            f"{fmt_cost(agg.cost('opus')):>7} "
            f"{fmt_cost(agg.cost('haiku')):>6}"
        )
        if not args.no_size:
            row += f"  {disk:>6.1f}MB"
        print(row)

    print(sep)
    row_g = (
        f"{'TOTAL':<32} {grand.api_calls():>6,}  "
        f"{fmt_tok(grand.input_tokens()):>7} "
        f"{fmt_tok(grand.cache_write_tokens()):>7} "
        f"{fmt_tok(grand.cache_read_tokens()):>7} "
        f"{fmt_tok(grand.output_tokens()):>7}  "
        f"{fmt_cost(grand.incoming_cost()):>8} "
        f"{fmt_cost(grand.outgoing_cost()):>8} "
        f"{fmt_cost(grand.cost()):>9}  "
        f"{fmt_cost(grand.cost('sonnet')):>7} "
        f"{fmt_cost(grand.cost('opus')):>7} "
        f"{fmt_cost(grand.cost('haiku')):>6}"
    )
    if not args.no_size:
        row_g += f"  {sum(disk_mb.values()):>6.1f}MB"
    print(row_g)
    print()

    # ── Per-session detail (when --project) ────────────────────────────────────
    if args.project:
        for proj, sessions in project_sessions.items():
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

    # ── CSV ────────────────────────────────────────────────────────────────────
    if args.csv:
        import csv
        rows = []
        for proj in projects:
            agg = project_aggs[proj]
            rows.append({
                "project": proj,
                "api_calls": agg.api_calls(),
                "input_tokens": agg.input_tokens(),
                "cache_write_tokens": agg.cache_write_tokens(),
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
        plot_matplotlib(filt, args.out_dir)

    if args.html:
        plot_html(filt, args.out_dir)


if __name__ == "__main__":
    main()
