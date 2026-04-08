import sys
import os
import time
import logging
import argparse
import subprocess
from datetime import datetime, timezone

                                                    
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("weather_agent")


def _print_status() -> None:
    from forecast_manager import get_manager
    mgr = get_manager()
    info = mgr.status()
    print("\n=== Forecast RAG Status ===")
    print(f"  Total chunks  : {info['total_forecast_chunks']}")
    print(f"  Fresh chunks  : {info['fresh_chunks']}")
    print(f"  Expired chunks: {info['expired_chunks']}")
    print(f"  Next expiry   : {info['next_expiry']}")
    if info["cities_indexed"]:
        print(f"  Cities ({len(info['cities_indexed'])}): {', '.join(info['cities_indexed'])}")
    else:
        print("  Cities        : (none indexed yet)")
    print()


def _run_once(force: bool = False) -> None:
    from forecast_manager import check_and_refresh
    refreshed = check_and_refresh(force=force)
    if refreshed:
        print("[ingest_forecast] Weekly forecast data ingested successfully.")
    else:
        print("[ingest_forecast] Forecast data is still fresh — nothing to do.")


def _run_verification() -> bool:
    project_root = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(project_root, "tests", "test_post_ingest_verification.py")
    print("\n[ingest_forecast] Running post-ingest verification …")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short",
         "--no-header", "-q"],
        cwd=project_root,
    )
    if result.returncode == 0:
        print("[ingest_forecast] Verification PASSED — store is healthy.")
    else:
        print("[ingest_forecast] Verification FAILED — check output above.")
    return result.returncode == 0


def _run_daemon(interval_seconds: int) -> None:
    print(
        f"[ingest_forecast] Daemon started. "
        f"Checking every {interval_seconds // 3600}h "
        f"{(interval_seconds % 3600) // 60}m."
    )
    print("  Press Ctrl+C to stop.\n")
    while True:
        now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[{now}] Running TTL check …")
        try:
            _run_once(force=False)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logger.error(f"[daemon] Unhandled error: {exc}", exc_info=True)
        print(f"[{now}] Next check in {interval_seconds} s.")
        time.sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Saudi city weather forecast ingestor with 7-day TTL."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a full re-ingest even if current data is still fresh.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current forecast status and exit.",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Stay running and re-check the TTL periodically.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=86400,
        metavar="SECONDS",
        help="Daemon check interval in seconds (default: 86400 = 24 h).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After ingestion, run post-ingest verification tests against the live store.",
    )
    args = parser.parse_args()

    if args.status:
        _print_status()
        return

    if args.daemon:
        _run_daemon(args.interval)
    else:
        _run_once(force=args.force)
        if args.verify:
            ok = _run_verification()
            sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
