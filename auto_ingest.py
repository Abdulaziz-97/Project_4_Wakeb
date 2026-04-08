import os
import sys
import time
import argparse
import logging
import subprocess
import platform
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

                                                                             
                
                                                                             
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

LOGS_DIR  = PROJECT_ROOT / "logs"
LOG_FILE  = LOGS_DIR / "auto_ingest.log"
LOCK_FILE = PROJECT_ROOT / ".auto_ingest.lock"

LOGS_DIR.mkdir(exist_ok=True)

                                                                             
                                                      
                                                                             

def _setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger("auto_ingest")
    logger.setLevel(level)
    logger.handlers.clear()

             
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(ch)

                   
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3,
                             encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(fh)

    return logger


log = _setup_logging()

                                                                             
                                      
                                                                             

class _RunLock:
    def __enter__(self):
        if LOCK_FILE.exists():
            pid = LOCK_FILE.read_text().strip()
            log.warning(f"Lock file exists (PID {pid}). Another cycle may be running.")
        LOCK_FILE.write_text(str(os.getpid()))
        return self

    def __exit__(self, *_):
        LOCK_FILE.unlink(missing_ok=True)


                                                                             
            
                                                                             

def run_cycle(force: bool = False, verify: bool = True) -> int:
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    log.info("=" * 60)
    log.info(f"Ingestion cycle started  [{now}]")
    log.info(f"  force={force}  verify={verify}")

                                 
    try:
        from forecast_manager import check_and_refresh
        refreshed = check_and_refresh(force=force)
    except Exception as exc:
        log.error(f"check_and_refresh() raised: {exc}", exc_info=True)
        log.info("Cycle finished  [FAILED — ingestion error]")
        return 2

    if refreshed:
        log.info("Fresh data ingested for all 28 Saudi cities (7-day TTL).")
    else:
        log.info("Forecast data is still fresh -- skipping API calls.")

                                       
    if verify:
        log.info("Running post-ingest verification ...")
        test_file = str(PROJECT_ROOT / "tests" / "test_post_ingest_verification.py")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file,
             "-q", "--tb=short", "--no-header"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
                                                
        for line in result.stdout.splitlines():
            log.info(f"  [pytest] {line}")
        for line in result.stderr.splitlines():
            if line.strip():
                log.warning(f"  [pytest] {line}")

        if result.returncode == 0:
            log.info("Verification PASSED — store is healthy.")
        else:
            log.error("Verification FAILED — see lines above.")
            log.info("Cycle finished  [FAILED — verification]")
            return 1

    log.info(f"Cycle finished  [OK  refreshed={refreshed}]")
    return 0


                                                                             
               
                                                                             

def print_status() -> None:
    try:
        from forecast_manager import get_manager
        info = get_manager().status()
    except Exception as exc:
        log.error(f"Could not read status: {exc}")
        return

    sep = "+" + "-" * 42 + "+"
    print(f"\n{sep}")
    print(f"|   Forecast RAG -- Store Status         |")
    print(sep)
    print(f"|  Total chunks   : {info['total_forecast_chunks']:<23}|")
    print(f"|  Fresh           : {info['fresh_chunks']:<22}|")
    print(f"|  Expired         : {info['expired_chunks']:<22}|")
    print(f"|  Next expiry     : {info['next_expiry']:<22}|")
    print(f"|  Cities indexed  : {len(info['cities_indexed']):<22}|")
    print(sep)
    if info["cities_indexed"]:
        for i in range(0, len(info["cities_indexed"]), 3):
            row = ", ".join(info["cities_indexed"][i:i+3])
            print(f"|  {row:<40}|")
    else:
        print("|  (none -- run auto_ingest.py first)    |")
    print(f"{sep}\n")


                                                                             
                                
                                                                             

def run_daemon(interval_hours: float = 24.0, force_first: bool = False) -> None:
    import schedule

    log.info(f"Daemon started — checking every {interval_hours:.1f} h.")
    log.info("  Press Ctrl+C to stop.\n")

    def _job():
        with _RunLock():
            run_cycle(force=False, verify=True)

                     
    with _RunLock():
        run_cycle(force=force_first, verify=True)

                                  
    schedule.every(interval_hours).hours.do(_job)

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)                                               
    except KeyboardInterrupt:
        log.info("Daemon stopped by user.")


                                                                             
                                    
                                                                             

TASK_NAME = "WeatherForecastAutoIngest"


def install_task(run_time: str = "00:00", day: str = "MON") -> None:
    if platform.system() != "Windows":
        log.error("--install is only supported on Windows.")
        sys.exit(1)

    python_exe = sys.executable
    script     = str(PROJECT_ROOT / "auto_ingest.py")

    cmd = [
        "schtasks", "/create",
        "/tn",  TASK_NAME,
        "/tr",  f'"{python_exe}" "{script}"',
        "/sc",  "WEEKLY",
        "/d",   day,
        "/st",  run_time,
        "/f",                                             
    ]
    log.info(f"Registering Task Scheduler task '{TASK_NAME}' ...")
    log.info(f"  Schedule : every {day} at {run_time}")
    log.info(f"  Command  : {python_exe} {script}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        log.info("Task registered successfully.")
        log.info("  Trigger  : every %s at %s", day, run_time)
        log.info("  Run now  : schtasks /run /tn %s", TASK_NAME)
        log.info("  View UI  : taskschd.msc")
        log.info("  Disable  : python auto_ingest.py --uninstall")
    else:
        err = (result.stdout + result.stderr).strip()
        log.error("schtasks failed: %s", err)
        if "Access is denied" in err:
            log.error(
                "Tip: open this terminal as Administrator and re-run, "
                "OR use daemon mode instead:  python auto_ingest.py --daemon"
            )
        sys.exit(1)


def uninstall_task() -> None:
    if platform.system() != "Windows":
        log.error("--uninstall is only supported on Windows.")
        sys.exit(1)

    cmd = ["schtasks", "/delete", "/tn", TASK_NAME, "/f"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        log.info(f"Task '{TASK_NAME}' removed from Task Scheduler.")
    else:
        log.warning(f"Could not remove task: {result.stdout} {result.stderr}")


                                                                             
     
                                                                             

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-scheduler for Saudi city weather forecast ingestion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force a full re-ingest even if the current data is still fresh.",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip post-ingest verification tests.",
    )
    parser.add_argument(
        "--daemon", action="store_true",
        help="Stay running and re-check the TTL every --interval hours.",
    )
    parser.add_argument(
        "--interval", type=float, default=24.0, metavar="HOURS",
        help="Daemon re-check interval in hours (default: 24).",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print store status and exit.",
    )
    parser.add_argument(
        "--install", action="store_true",
        help="Register a weekly Windows Task Scheduler entry and exit.",
    )
    parser.add_argument(
        "--uninstall", action="store_true",
        help="Remove the Windows Task Scheduler entry and exit.",
    )
    parser.add_argument(
        "--run-time", default="00:00", metavar="HH:MM",
        help="Time-of-day for the scheduled task (default: 00:00).",
    )
    parser.add_argument(
        "--day", default="MON",
        choices=["MON","TUE","WED","THU","FRI","SAT","SUN"],
        help="Day of week for the scheduled task (default: MON).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

                                         
    if args.verbose:
        for h in log.handlers:
            h.setLevel(logging.DEBUG)
        log.setLevel(logging.DEBUG)

    if args.status:
        print_status()
        return

    if args.install:
        install_task(run_time=args.run_time, day=args.day)
        return

    if args.uninstall:
        uninstall_task()
        return

    verify = not args.no_verify

    if args.daemon:
        run_daemon(interval_hours=args.interval, force_first=args.force)
    else:
        with _RunLock():
            rc = run_cycle(force=args.force, verify=verify)
        sys.exit(rc)


if __name__ == "__main__":
    main()
