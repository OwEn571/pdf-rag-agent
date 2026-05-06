from __future__ import annotations

import argparse
from pathlib import Path
import sys

if str(PROJECT_ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.services.retrieval.indexing import V4IngestionService  # noqa: E402
from app.services.infra.model_clients import ModelClients  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild V4 paper/block indices")
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    setup_logging(settings.log_level)
    service = V4IngestionService(settings, clients=ModelClients(settings))
    force_rebuild = True
    if args.resume and not args.force_rebuild:
        force_rebuild = False
    stats = service.rebuild(max_papers=args.max_papers, force_rebuild=force_rebuild)
    print(stats.to_dict())


if __name__ == "__main__":
    main()
