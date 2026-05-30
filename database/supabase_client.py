# =============================================================
# database/supabase_client.py
# Phase 6 — Database Client with Full Offline Fallback
#
# FIX (CRITICAL): Previously, a missing SUPABASE_URL caused the
# batch pipeline to silently crash on every write.  Now:
#   • SUPABASE_URL set   → live Supabase client (unchanged behaviour)
#   • SUPABASE_URL unset → OfflineClient writes all rows to
#     local JSONL + CSV under OFFLINE_DIR (default: offline_data/)
#
# All callers use the same chained API:
#   supabase.table("t").upsert(rows).execute()
# Zero changes required in any caller file.
# =============================================================

import os
import csv
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Where offline writes land
OFFLINE_DIR = Path(os.getenv("OFFLINE_FALLBACK_DIR", "offline_data"))


# =============================================================
# OFFLINE TABLE WRITER
# =============================================================
class _OfflineTableWriter:
    """
    Mimics the Supabase table fluent API but persists to local files.
    Supports: .insert(), .upsert(), .update().eq(), .select(),
              .order(), .limit(), .execute()
    """

    def __init__(self, table_name: str, offline_dir: Path):
        self._table   = table_name
        self._dir     = offline_dir
        self._rows    = []
        self._op      = "noop"
        self._update_data    = {}
        self._update_filters = {}

    # ── write ops ──────────────────────────────────────────────

    def insert(self, rows):
        self._rows = rows if isinstance(rows, list) else [rows]
        self._op   = "insert"
        return self

    def upsert(self, rows, on_conflict: str = ""):
        self._rows = rows if isinstance(rows, list) else [rows]
        self._op   = "upsert"
        return self

    def update(self, data: dict):
        self._update_data = data
        self._op = "update"
        return self

    # ── filter / read stubs ────────────────────────────────────

    def eq(self, col: str, val):
        self._update_filters[col] = val
        return self

    def select(self, *args):
        self._op = "select"
        return self

    def order(self, *args, **kwargs):
        return self

    def limit(self, n: int):
        return self

    # ── flush ──────────────────────────────────────────────────

    def execute(self):
        try:
            self._dir.mkdir(parents=True, exist_ok=True)

            if self._op in ("insert", "upsert") and self._rows:
                self._write_jsonl(self._rows)
                self._write_csv(self._rows)
                print(
                    f"[OFFLINE CLIENT] {self._op.upper()} "
                    f"{len(self._rows)} rows → {self._table}"
                )

            elif self._op == "update":
                entry = {
                    "_op":      "UPDATE",
                    "_table":   self._table,
                    "_filters": self._update_filters,
                    "_data":    self._update_data,
                    "_ts":      datetime.now(timezone.utc).isoformat(),
                }
                self._write_jsonl([entry])
                print(
                    f"[OFFLINE CLIENT] UPDATE logged → {self._table} "
                    f"filters={self._update_filters}"
                )

        except Exception as exc:
            print(f"[OFFLINE CLIENT] Write error ({self._table}): {exc}")
            traceback.print_exc()

        class _MockResponse:
            data = []
        return _MockResponse()

    # ── helpers ────────────────────────────────────────────────

    def _write_jsonl(self, rows: list):
        path = self._dir / f"{self._table}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")

    def _write_csv(self, rows: list):
        if not rows:
            return
        path       = self._dir / f"{self._table}.csv"
        fieldnames = list(rows[0].keys())
        write_hdr  = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore")
            if write_hdr:
                writer.writeheader()
            writer.writerows(rows)


# =============================================================
# OFFLINE CLIENT
# =============================================================
class OfflineClient:
    """
    Drop-in replacement for the Supabase client.
    All writes go to OFFLINE_DIR/<table>.{jsonl,csv}.
    """

    def __init__(self, offline_dir: Path = OFFLINE_DIR):
        self._dir = offline_dir
        print(
            f"[OFFLINE CLIENT] Running in offline mode. "
            f"All writes → {self._dir.resolve()}"
        )

    def table(self, name: str) -> _OfflineTableWriter:
        return _OfflineTableWriter(name, self._dir)

    def __getattr__(self, item):
        return lambda *a, **kw: None


# =============================================================
# FACTORY
# =============================================================
def _build_client():
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            from supabase import create_client
            client = create_client(SUPABASE_URL, SUPABASE_KEY)
            print("✅ Supabase connected")
            return client
        except ImportError:
            print("⚠️  supabase-py not installed — using OfflineClient.")
        except Exception as exc:
            print(f"❌ Supabase error: {exc} — using OfflineClient.")
    else:
        print("⚠️  SUPABASE_URL/KEY not set — using OfflineClient.")

    return OfflineClient()


# Module-level singleton — import this everywhere
supabase = _build_client()


def is_online() -> bool:
    return not isinstance(supabase, OfflineClient)


def client_mode() -> str:
    return "supabase_live" if is_online() else "offline_local"
