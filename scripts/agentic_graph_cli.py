#!/usr/bin/env python3
"""CLI for the local HENRI agentic event graph.

The default vault is external to the production repository. Set
OBSIDIAN_VAULT_PATH to the real Obsidian vault before use.
"""
from __future__ import annotations

import argparse
import json
import sys

from agentic_event_store import (
    append_edge,
    append_event,
    default_vault_path,
    query_events,
    verify_local_events,
    write_projection,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", default=None, help="external Obsidian vault path")
    sub = parser.add_subparsers(dest="command", required=True)

    event = sub.add_parser("event")
    event.add_argument("event_type")
    event.add_argument("--stream", required=True)
    event.add_argument("--actor", required=True)
    event.add_argument("--status", default="observed")
    event.add_argument("--payload", required=True, help="JSON object")
    event.add_argument("--run-id")
    event.add_argument("--parent-event-id")
    event.add_argument("--source-uri")

    edge = sub.add_parser("edge")
    edge.add_argument("source_event_id")
    edge.add_argument("target_event_id")
    edge.add_argument("relation")
    edge.add_argument("--actor", required=True)
    edge.add_argument("--status", default="derived")

    query = sub.add_parser("query")
    query.add_argument("--stream")
    query.add_argument("--event-type")
    query.add_argument("--after")
    query.add_argument("--before")
    query.add_argument("--limit", type=int, default=100)

    sub.add_parser("project")
    sub.add_parser("verify")
    args = parser.parse_args()
    vault = args.vault or str(default_vault_path())

    try:
        if args.command == "event":
            payload = json.loads(args.payload)
            if not isinstance(payload, dict):
                raise ValueError("--payload must decode to a JSON object")
            result = append_event(
                args.event_type,
                payload,
                stream=args.stream,
                actor=args.actor,
                causal_status=args.status,
                run_id=args.run_id,
                parent_event_id=args.parent_event_id,
                source_uri=args.source_uri,
                vault_path=vault,
            )
        elif args.command == "edge":
            result = append_edge(
                args.source_event_id,
                args.target_event_id,
                args.relation,
                actor=args.actor,
                causal_status=args.status,
                vault_path=vault,
            )
        elif args.command == "query":
            result = query_events(
                vault_path=vault,
                stream=args.stream,
                event_type=args.event_type,
                after=args.after,
                before=args.before,
                limit=args.limit,
            )
        elif args.command == "project":
            result = {"projection": str(write_projection(vault))}
        else:
            result = verify_local_events(vault)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": type(exc).__name__, "message": str(exc)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
