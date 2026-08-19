"""
Script to force-stop stuck or unfinished LangSmith runs.
"""

from datetime import datetime, timezone
from dotenv import load_dotenv
from langsmith import Client
from loguru import logger

load_dotenv()

def stop_stuck_runs():
    client = Client()
    project_name = "github-codebase-rag"
    logger.info(f"Checking for unfinished runs in LangSmith project '{project_name}'...")

    runs = list(client.list_runs(project_name=project_name))
    stuck_count = 0
    now = datetime.now(timezone.utc)

    for r in runs:
        if r.end_time is None:
            logger.info(f"Stopping stuck run: {r.id} ({r.name}) started at {r.start_time}")
            try:
                client.update_run(
                    r.id,
                    end_time=now,
                    error="Force stopped by user",
                )
                stuck_count += 1
            except Exception as e:
                logger.warning(f"Could not update run {r.id}: {e}")

    if stuck_count > 0:
        logger.success(f"Force stopped {stuck_count} stuck LangSmith run(s).")
    else:
        logger.info("No stuck runs found in LangSmith.")

if __name__ == "__main__":
    stop_stuck_runs()
