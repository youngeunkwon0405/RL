import subprocess
import time
import getpass
import datetime

# ─── Configuration ──────────────────────────────────────────────────────────────
SBATCH_SCRIPT = "launch_scripts/qwen-1b_math.sh"           # your sbatch script filename
CHECK_INTERVAL_RUNNING = 10 * 60     # seconds to wait if jobs are running (10 min)
CHECK_INTERVAL_FREE    = 60 * 60 * 4.5    # seconds to wait after submitting (1 hour)

USER = getpass.getuser()             # detect your username
# ────────────────────────────────────────────────────────────────────────────────

def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def submit_job():
    print(f"[{timestamp()}] Submitting job: bash {SBATCH_SCRIPT}")
    try:
        subprocess.run(["bash", SBATCH_SCRIPT], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[{timestamp()}] ERROR submitting job: {e}")

def has_running_jobs():
    # -h: no header, -u USER: only your jobs
    res = subprocess.run(
        ["squeue", "-h", "-u", USER],
        capture_output=True, text=True
    )
    # any non-empty line means at least one job is pending/running
    return bool(res.stdout.strip())

def main():
    # initial submission
    submit_job()
    print(f"[{timestamp()}] Sleeping for {CHECK_INTERVAL_FREE//60} minutes...")
    time.sleep(CHECK_INTERVAL_FREE)

    # now loop forever
    while True:
        if not has_running_jobs():
            print(f"[{timestamp()}] No running jobs detected. Re-submitting.")
            submit_job()
            print(f"[{timestamp()}] Sleeping for {CHECK_INTERVAL_FREE//60} minutes...")
            time.sleep(CHECK_INTERVAL_FREE)
        else:
            print(f"[{timestamp()}] Jobs still running; checking again in {CHECK_INTERVAL_RUNNING//60} minutes.")
            time.sleep(CHECK_INTERVAL_RUNNING)

if __name__ == "__main__":
    main()