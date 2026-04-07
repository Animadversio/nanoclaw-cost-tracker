#!/usr/bin/env python3
"""
nanoclaw_cost.py — Token usage & cost breakdown for NanoClaw Claude Code sessions.

Usage:
    python nanoclaw_cost.py /path/to/nanoclaw/data/sessions
    python nanoclaw_cost.py /path/to/nanoclaw/data/sessions --sort cost
    python nanoclaw_cost.py /path/to/nanoclaw/data/sessions --project discord_diffusion_objrel
    python nanoclaw_cost.py /path/to/nanoclaw/data/sessions --csv out.csv
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional
import shutil

# ── Pricing (per million tokens, as of 2025-04) ──────────────────────────────
# https://www.anthropic.com/pricing
PRICING = {
    # model-id-prefix -> (input, cache_write, cache_read, output) per 1M tokens
    "claude-opus-4":      (15.00,  18.75,  1.50, 75.00),
    "claude-opus-4-5":    (15.00,  18.75,  1.50, 75.00),
    "claude-sonnet-4":    ( 3.00,   3.75,  0.30, 15.00),
    "claude-sonnet-4-6":  ( 3.00,   3.75,  0.30, 15.00),
    "claude-haiku-4":     ( 0.80,   1.00,  0.08,  4.00),
    "claude-haiku-4-5":   ( 0.80,   1.00,  0.08,  4.00),
    # legacy fallback
    "claude-3-5-sonnet":  ( 3.00,   3.75,  0.30, 15.00),
    "claude-3-opus":      (15.00,  18.75,  1.50, 75.00),
    "claude-3-haiku":     ( 0.25,   0.30,  0.03,  1.25),
}
DEFAULT_PRICING = (3.00, 3.75, 0.30, 15.00)  # sonnet fallback


def get_price(model: str):
    if not model:
        return DEFAULT_PRICING
    for prefix, prices in PRICING.items():
        if model.startswith(prefix):
            return prices
    return DEFAULT_PRICING


def tokens_to_cost(input_tok, cache_write_tok, cache_read_tok, output_tok, model):
    p_in, p_cw, p_cr, p_out = get_price(model)
    return (
        input_tok      * p_in  / 1_000_000 +
        cache_write_tok * p_cw  / 1_000_000 +
        cache_read_tok  * p_cr  / 1_000_000 +
        output_tok      * p_out / 1_000_000
    )


@dataclass
class SessionStats:
    session_id: str
    project: str
    jsonl_path: str
    is_subagent: bool = False

    input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    models: Dict[str, int] = field(default_factory=dict)  # model -> call count

    @property
    def total_tokens(self):
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens + self.output_tokens

    @property
    def cost_usd(self):
        # use the dominant model for pricing
        dominant = max(self.models, key=self.models.get) if self.models else ""
        return tokens_to_cost(
            self.input_tokens, self.cache_write_tokens,
            self.cache_read_tokens, self.output_tokens, dominant
        )


def parse_jsonl(path: str) -> SessionStats:
    p = Path(path)
    # project = the named session dir (e.g. discord_diffusion_objrel)
    parts = p.parts
    # find "sessions" in path
    try:
        si = parts.index("sessions")
        project = parts[si + 1]
    except (ValueError, IndexError):
        project = p.parent.parent.name

    is_subagent = "subagents" in str(path)
    session_id = p.stem

    stats = SessionStats(
        session_id=session_id,
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
                model = msg.get("model", "")
                stats.input_tokens       += usage.get("input_tokens", 0)
                stats.cache_write_tokens += usage.get("cache_creation_input_tokens", 0)
                stats.cache_read_tokens  += usage.get("cache_read_input_tokens", 0)
                stats.output_tokens      += usage.get("output_tokens", 0)
                stats.api_calls          += 1
                stats.models[model]       = stats.models.get(model, 0) + 1
    except (OSError, PermissionError) as e:
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


def fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def fmt_cost(c: float) -> str:
    if c >= 1.0:
        return f"${c:.3f}"
    return f"${c:.4f}"


def main():
    parser = argparse.ArgumentParser(description="NanoClaw token usage & cost breakdown")
    parser.add_argument("sessions_dir", help="Path to nanoclaw/data/sessions/")
    parser.add_argument("--project", help="Filter to a specific project name")
    parser.add_argument("--sort", choices=["cost", "tokens", "calls", "project"],
                        default="cost", help="Sort by (default: cost)")
    parser.add_argument("--include-subagents", action="store_true",
                        help="Include subagent sessions separately (default: roll up into parent)")
    parser.add_argument("--csv", help="Also write results to a CSV file")
    parser.add_argument("--no-size", action="store_true",
                        help="Skip disk size calculation (faster)")
    args = parser.parse_args()

    sessions_root = Path(args.sessions_dir)
    if not sessions_root.exists():
        print(f"ERROR: {sessions_root} does not exist", file=sys.stderr)
        sys.exit(1)

    # Find all .jsonl files under .claude/projects/
    jsonl_files = sorted(sessions_root.rglob("*.jsonl"))
    # Exclude mcp-logs
    jsonl_files = [p for p in jsonl_files if "mcp-logs" not in str(p)]

    if args.project:
        jsonl_files = [p for p in jsonl_files if args.project in str(p)]

    # Parse all sessions
    all_sessions = [parse_jsonl(str(p)) for p in jsonl_files]

    # Roll up subagents into their parent project totals
    # Group by project
    project_totals: Dict[str, SessionStats] = {}
    project_sessions: Dict[str, list] = defaultdict(list)

    for s in all_sessions:
        if s.project not in project_totals:
            project_totals[s.project] = SessionStats(
                session_id="(total)", project=s.project, jsonl_path=""
            )
        pt = project_totals[s.project]
        pt.input_tokens       += s.input_tokens
        pt.cache_write_tokens += s.cache_write_tokens
        pt.cache_read_tokens  += s.cache_read_tokens
        pt.output_tokens      += s.output_tokens
        pt.api_calls          += s.api_calls
        for m, c in s.models.items():
            pt.models[m] = pt.models.get(m, 0) + c
        project_sessions[s.project].append(s)

    # Disk sizes
    disk_sizes: Dict[str, float] = {}
    if not args.no_size:
        for proj_name in project_totals:
            proj_path = sessions_root / proj_name
            disk_sizes[proj_name] = dir_size_mb(str(proj_path))

    # Sort projects
    projects = list(project_totals.values())
    if args.sort == "cost":
        projects.sort(key=lambda x: x.cost_usd, reverse=True)
    elif args.sort == "tokens":
        projects.sort(key=lambda x: x.total_tokens, reverse=True)
    elif args.sort == "calls":
        projects.sort(key=lambda x: x.api_calls, reverse=True)
    elif args.sort == "project":
        projects.sort(key=lambda x: x.project)

    # ── Print summary ──────────────────────────────────────────────────────────
    W = shutil.get_terminal_size((120, 40)).columns
    sep = "─" * min(W, 130)

    print(f"\n{'NanoClaw Token Usage & Cost Report':^{min(W,130)}}")
    print(f"{'Sessions root: ' + str(sessions_root):^{min(W,130)}}")
    print(sep)

    hdr = f"{'Project':<30} {'API calls':>9} {'Input':>8} {'CacheW':>8} {'CacheR':>8} {'Output':>8} {'Total':>9} {'Cost':>9}"
    if not args.no_size:
        hdr += f"  {'Disk':>7}"
    print(hdr)
    print(sep)

    grand = SessionStats(session_id="GRAND TOTAL", project="", jsonl_path="")
    total_disk = 0.0

    for pt in projects:
        dominant = max(pt.models, key=pt.models.get) if pt.models else "?"
        disk = disk_sizes.get(pt.project, 0.0)
        total_disk += disk

        row = (
            f"{pt.project:<30} "
            f"{pt.api_calls:>9,} "
            f"{fmt_tokens(pt.input_tokens):>8} "
            f"{fmt_tokens(pt.cache_write_tokens):>8} "
            f"{fmt_tokens(pt.cache_read_tokens):>8} "
            f"{fmt_tokens(pt.output_tokens):>8} "
            f"{fmt_tokens(pt.total_tokens):>9} "
            f"{fmt_cost(pt.cost_usd):>9}"
        )
        if not args.no_size:
            row += f"  {disk:>6.1f}MB"
        print(row)

        # accumulate grand total
        grand.input_tokens       += pt.input_tokens
        grand.cache_write_tokens += pt.cache_write_tokens
        grand.cache_read_tokens  += pt.cache_read_tokens
        grand.output_tokens      += pt.output_tokens
        grand.api_calls          += pt.api_calls
        for m, c in pt.models.items():
            grand.models[m] = grand.models.get(m, 0) + c

    print(sep)
    grand_row = (
        f"{'TOTAL':<30} "
        f"{grand.api_calls:>9,} "
        f"{fmt_tokens(grand.input_tokens):>8} "
        f"{fmt_tokens(grand.cache_write_tokens):>8} "
        f"{fmt_tokens(grand.cache_read_tokens):>8} "
        f"{fmt_tokens(grand.output_tokens):>8} "
        f"{fmt_tokens(grand.total_tokens):>9} "
        f"{fmt_cost(grand.cost_usd):>9}"
    )
    if not args.no_size:
        grand_row += f"  {total_disk:>6.1f}MB"
    print(grand_row)

    # ── Per-session breakdown (if --project specified) ─────────────────────────
    if args.project:
        print(f"\n{'Per-session breakdown for: ' + args.project}")
        print(sep)
        sessions = project_sessions.get(args.project, [])
        if args.sort == "cost":
            sessions.sort(key=lambda x: x.cost_usd, reverse=True)
        elif args.sort == "tokens":
            sessions.sort(key=lambda x: x.total_tokens, reverse=True)

        hdr2 = f"{'Session ID':<40} {'sub':>4} {'calls':>6} {'Input':>8} {'CacheW':>8} {'CacheR':>8} {'Output':>8} {'Cost':>9} {'Model'}"
        print(hdr2)
        print(sep)
        for s in sessions:
            dominant = max(s.models, key=s.models.get) if s.models else "?"
            print(
                f"{s.session_id:<40} "
                f"{'Y' if s.is_subagent else 'N':>4} "
                f"{s.api_calls:>6,} "
                f"{fmt_tokens(s.input_tokens):>8} "
                f"{fmt_tokens(s.cache_write_tokens):>8} "
                f"{fmt_tokens(s.cache_read_tokens):>8} "
                f"{fmt_tokens(s.output_tokens):>8} "
                f"{fmt_cost(s.cost_usd):>9} "
                f"{dominant}"
            )

    # ── CSV export ─────────────────────────────────────────────────────────────
    if args.csv:
        import csv
        rows = []
        for pt in projects:
            dominant = max(pt.models, key=pt.models.get) if pt.models else ""
            disk = disk_sizes.get(pt.project, 0.0)
            rows.append({
                "project": pt.project,
                "api_calls": pt.api_calls,
                "input_tokens": pt.input_tokens,
                "cache_write_tokens": pt.cache_write_tokens,
                "cache_read_tokens": pt.cache_read_tokens,
                "output_tokens": pt.output_tokens,
                "total_tokens": pt.total_tokens,
                "cost_usd": round(pt.cost_usd, 5),
                "dominant_model": dominant,
                "disk_mb": round(disk, 2),
            })
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written to: {args.csv}")

    print()


if __name__ == "__main__":
    main()
