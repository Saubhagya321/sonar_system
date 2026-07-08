"""
Port Scheduler — OOP Refactored
================================
Schedules per-port API calls at 20:00 local time and runs a stored procedure
1 hour before the earliest port run. Sequential execution is intentional
(ThreadPoolExecutor(1)) to prevent parallel port processing.
"""

import logging
import requests
import pytz
import pandas as pd

from datetime import datetime, time, timedelta, timezone
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_SCHEDULER_STARTED

import config

# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

log = logging.getLogger("PORT-SCHEDULER")


# ==========================================================
# DATA CLASSES
# ==========================================================

@dataclass
class Port:
    """Represents a single port record loaded from the database."""
    id: int
    code: str
    timezone: str


@dataclass
class SchedulerConfig:
    """Central configuration for the scheduler."""
    api_url: str
    test_mode: bool
    test_spread_seconds: int
    batch_size: int
    misfire_grace_time: int


# ==========================================================
# DATABASE MANAGER
# ==========================================================

class DatabaseManager:
    """Handles all database interactions."""

    def __init__(self, connection_string: str, port_db_name: str) -> None:
        self._engine: Engine = create_engine(connection_string)
        self._port_db_name = port_db_name

    # ----------------------------------------------------------
    # Port Loading
    # ----------------------------------------------------------

    def load_ports(self) -> list[Port]:
        """
        Fetches all ETG-flagged ports with their timezone descriptions.

        Returns:
            List of Port dataclass instances. Returns empty list on failure.
        """
        query = f"""
            SELECT p.Id, p.Code, t.Description AS Timezone
            FROM [{self._port_db_name}].dbo.Port p
            JOIN [{self._port_db_name}].dbo.Timezone t ON p.RefTimeZone = t.ID
            WHERE p.EtgFlag = 1
        """
        try:
            log.info("Loading ports from database")
            df = pd.read_sql(query, self._engine)
            ports = [Port(id=row["Id"], code=row["Code"], timezone=row["Timezone"]) for _, row in df.iterrows()]
            log.info(f"{len(ports)} port(s) loaded successfully")
            return ports
        except SQLAlchemyError:
            log.exception("SQLAlchemyError: Failed to load ports from database")
            return []
        except Exception:
            log.exception("Unexpected error while loading ports")
            return []

    # ----------------------------------------------------------
    # Execution Tracking
    # ----------------------------------------------------------

    def already_ran_today(self, port_id: int) -> bool:
        """
        Checks whether a port job has already been executed today (UTC date).

        Args:
            port_id: The port's primary key.

        Returns:
            True if already ran today, False otherwise.
            Returns False on DB error to allow the job to attempt execution.
        """
        today = datetime.now(timezone.utc).date()
        query = text("""
            SELECT COUNT(*)
            FROM PortDailyRun
            WHERE PortId = :port_id
              AND CAST(RunDate AS DATE) = :today
        """)
        try:
            with self._engine.connect() as conn:
                result = conn.execute(query, {"port_id": port_id, "today": today}).scalar()
            return (result or 0) > 0
        except SQLAlchemyError:
            log.exception(f"SQLAlchemyError: Could not check execution status for port_id={port_id}")
            return False
        except Exception:
            log.exception(f"Unexpected error checking execution status for port_id={port_id}")
            return False

    def mark_ran(self, port_id: int) -> bool:
        """
        Records that a port job executed successfully today.

        Args:
            port_id: The port's primary key.

        Returns:
            True on success, False on failure.
        """
        query = text("""
            INSERT INTO PortDailyRun (PortId, RunDate)
            VALUES (:port_id, GETUTCDATE())
        """)
        try:
            with self._engine.begin() as conn:
                conn.execute(query, {"port_id": port_id})
            log.info(f"Execution recorded for port_id={port_id}")
            return True
        except SQLAlchemyError:
            log.exception(f"SQLAlchemyError: Failed to record execution for port_id={port_id}")
            return False
        except Exception:
            log.exception(f"Unexpected error recording execution for port_id={port_id}")
            return False

    # ----------------------------------------------------------
    # Stored Procedure
    # ----------------------------------------------------------

    def run_update_booking_sp(self) -> bool:
        """
        Executes the UpdateBookingFeedVehicleType stored procedure.

        Returns:
            True on success, False on failure.
        """
        log.info("Running SP: UpdateBookingFeedVehicleType")
        try:
            with self._engine.begin() as conn:
                conn.execute(text("EXEC [dbo].[UpdateBookingFeedVehicleType]"))
            log.info("Stored procedure executed successfully")
            return True
        except SQLAlchemyError:
            log.exception("SQLAlchemyError: Stored procedure execution failed")
            return False
        except Exception:
            log.exception("Unexpected error during stored procedure execution")
            return False


# ==========================================================
# API CLIENT
# ==========================================================

class PortApiClient:
    """Handles HTTP communication with the port batch-process API."""

    def __init__(self, config: SchedulerConfig) -> None:
        self._config = config

    def call_batch_process(self, port: Port) -> bool:
        """
        POSTs a batch-process request for the given port.

        Args:
            port: Port instance to process.

        Returns:
            True if the API returned HTTP 200, False otherwise.
        """
        url = self._config.api_url
        payload = {"port_id": port.id, "port_code": port.code, "batch_size": self._config.batch_size}

        log.info(f"Calling API for port {port.code}: {url}")

        try:
            response = requests.post(url, json=payload, timeout=(10, 14400))
            log.info(f"Port {port.code} — API response: {response.status_code}")

            if response.status_code == 200:
                return True

            log.error(f"Port {port.code} — API error body: {response.text}")
            return False

        except requests.exceptions.ConnectionError:
            log.exception(f"ConnectionError: Could not reach API for port {port.code}")
            return False
        # except requests.exceptions.Timeout:
        #     log.exception(f"Timeout: API call exceeded {self._config.api_timeout_seconds}s for port {port.code}")
        #     return False
        except requests.exceptions.RequestException:
            log.exception(f"RequestException: Unexpected HTTP error for port {port.code}")
            return False


# ==========================================================
# JOB RUNNER
# ==========================================================

class PortJobRunner:
    """
    Encapsulates the logic for a single port's scheduled job.
    Injected with a DatabaseManager and PortApiClient for testability.
    """

    def __init__(self, db: DatabaseManager, api_client: PortApiClient) -> None:
        self._db = db
        self._api = api_client

    def run(self, port: Port) -> None:
        """
        Entry point for a scheduled port job.
        Checks idempotency, calls the API, and records execution.

        Args:
            port: Port to process.
        """
        log.info(f"Starting job for port {port.code} (id={port.id})")

        try:
            if self._db.already_ran_today(port.id):
                log.info(f"Port {port.code} already executed today — skipping")
                return

            success = self._api.call_batch_process(port)

            if success:
                self._db.mark_ran(port.id)
                log.info(f"Port {port.code} job completed successfully")
            else:
                log.error(f"Port {port.code} job failed — execution NOT recorded")

        except Exception:
            log.exception(f"Unhandled error in job runner for port {port.code}")


# ==========================================================
# SCHEDULER
# ==========================================================

class PortScheduler:
    """
    Builds and manages the APScheduler instance.

    Design notes:
    - ThreadPoolExecutor(1) is intentional: enforces sequential execution
      so ports never run in parallel, preventing DB/API contention.
    - SP job is scheduled 1 hour before the earliest 20:00 UTC across ports.
    - In test mode, SP runs a few seconds before the first port job fires,
      mirroring the production "SP runs before ports" ordering.
    """

    _EXECUTORS = {
        # max_workers=1 is intentional — DO NOT increase.
        # Parallel port execution would cause DB/API contention.
        "default": ThreadPoolExecutor(max_workers=1)
    }

    def __init__(self, config: SchedulerConfig, db: DatabaseManager) -> None:
        self._config = config
        self._db = db
        self._api_client = PortApiClient(config)
        self._job_runner = PortJobRunner(db, self._api_client)
        self._scheduler = BlockingScheduler(executors=self._EXECUTORS)
        self._scheduler.add_listener(self._on_scheduler_started, EVENT_SCHEDULER_STARTED)
        self._scheduler.add_listener(self._on_job_error, EVENT_JOB_ERROR)

    # ----------------------------------------------------------
    # Listeners
    # ----------------------------------------------------------

    def _on_scheduler_started(self, event) -> None:
        """Logs the next run time for every registered job on startup."""
        try:
            log.info("Scheduler started — listing next run times:")
            for job in self._scheduler.get_jobs():
                log.info(f"  {job.id} → next run at {job.next_run_time}")
        except Exception:
            log.exception("Error while logging next run times after scheduler start")

    def _on_job_error(self, event) -> None:
        """Logs any job that raises an unhandled exception."""
        log.error(f"Scheduler job crashed: job_id={event.job_id} | exception={event.exception}")

    # ----------------------------------------------------------
    # SP Scheduling
    # ----------------------------------------------------------

    def _get_earliest_port_run_utc(self, ports: list[Port]) -> Optional[datetime]:
        """
        Calculates the earliest 20:00 local time across all ports, in UTC.

        Args:
            ports: List of Port instances.

        Returns:
            The earliest UTC datetime for tonight's 20:00 run, or None if
            all ports have already passed their run time today.
        """
        now_utc = datetime.now(timezone.utc)
        earliest: Optional[datetime] = None

        for port in ports:
            try:
                tz = pytz.timezone(port.timezone)
                local_now = now_utc.astimezone(tz)
                local_run = tz.localize(datetime.combine(local_now.date(), time(20, 0)))
                run_utc = local_run.astimezone(timezone.utc)

                if run_utc < now_utc:
                    log.warning(f"Port {port.code}: 20:00 run time has already passed today — skipping for SP calc")
                    continue

                if earliest is None or run_utc < earliest:
                    earliest = run_utc

            except pytz.UnknownTimeZoneError:
                log.error(f"Unknown timezone '{port.timezone}' for port {port.code} — skipping")
            except Exception:
                log.exception(f"Error computing run UTC for port {port.code}")

        return earliest

    def schedule_sp_job(self, ports: list[Port]) -> None:
        """
        Schedules the UpdateBookingFeedVehicleType SP to run 1 hour before
        the earliest port job tonight.

        In TEST MODE, the SP is scheduled a few seconds before the first port
        job fires (now + 5s vs the first port job at now + 0s * spread),
        preserving the production ordering without waiting for real 20:00 times.

        Args:
            ports: List of Port instances used to determine the earliest run time.
        """
        if not ports:
            log.warning("No ports provided — SP job will not be scheduled")
            return

        try:
            if self._config.test_mode:
                # Port jobs are scheduled at now+0s, now+spread, now+2*spread …
                # Run the SP 5 seconds from now so it completes before ports start.
                sp_run_time = datetime.now(timezone.utc)
                log.info(f"TEST MODE → SP scheduled at {sp_run_time} UTC (5s from now, before first port job)")
            else:
                earliest = self._get_earliest_port_run_utc(ports)

                if earliest is None:
                    log.warning("All port run times have passed today — SP job will not be scheduled")
                    return

                sp_run_time = earliest - timedelta(hours=1)
                log.info(
                    f"SP scheduled to run at {sp_run_time} UTC "
                    f"(1 hour before earliest port run at {earliest} UTC)"
                )

            self._scheduler.add_job(
                self._db.run_update_booking_sp,
                DateTrigger(run_date=sp_run_time),
                id="daily_sp_job",
                replace_existing=True,
                misfire_grace_time=self._config.misfire_grace_time,
            )

        except Exception:
            log.exception("Failed to schedule SP job")

    # ----------------------------------------------------------
    # Port Job Scheduling
    # ----------------------------------------------------------

    def schedule_port_jobs(self, ports: list[Port]) -> None:
        """
        Registers a scheduled job for each port.
        In TEST_MODE, jobs are spread TEST_SPREAD_SECONDS apart from now.
        In production, jobs fire daily at 20:00 in the port's local timezone.

        Args:
            ports: List of Port instances to schedule.
        """
        if not ports:
            log.warning("No ports to schedule")
            return

        now_utc = datetime.now(timezone.utc)

        for i, port in enumerate(ports):
            try:
                if self._config.test_mode:
                    run_time = now_utc + timedelta(seconds=(i+20) * self._config.test_spread_seconds)
                    trigger = DateTrigger(run_date=run_time)
                    log.info(f"TEST MODE → Port {port.code} scheduled at {run_time} UTC")
                else:
                    tz = pytz.timezone(port.timezone)
                    trigger = CronTrigger(hour=20, minute=0, timezone=tz)
                    log.info(f"Port {port.code} scheduled daily at 20:00 ({port.timezone})")

                self._scheduler.add_job(
                    self._job_runner.run,
                    trigger,
                    args=[port],
                    id=f"port_{port.id}",
                    replace_existing=True,
                    misfire_grace_time=self._config.misfire_grace_time,
                    max_instances=1,
                )

            except pytz.UnknownTimeZoneError:
                log.error(f"Unknown timezone '{port.timezone}' for port {port.code} — job not scheduled")
            except Exception:
                log.exception(f"Failed to schedule job for port {port.code} (id={port.id})")

    # ----------------------------------------------------------
    # Start
    # ----------------------------------------------------------

    def start(self) -> None:
        """Loads ports, registers all jobs, and starts the blocking scheduler."""
        try:
            if self._config.test_mode:
                log.warning("===== TEST MODE ENABLED =====")

            log.info("Starting scheduler")
            ports = self._db.load_ports()

            if not ports:
                log.error("No ports loaded — scheduler will not start")
                return

            self.schedule_sp_job(ports)
            self.schedule_port_jobs(ports)

            log.info("All jobs registered — starting blocking scheduler")
            self._scheduler.start()

        except KeyboardInterrupt:
            log.info("Scheduler stopped by user (KeyboardInterrupt)")
        except Exception:
            log.exception("Scheduler crashed during startup")


# ==========================================================
# MAIN ENTRY
# ==========================================================

if __name__ == "__main__":
    _config = SchedulerConfig(
        config.API_URL,
        config.TEST_MODE,
        config.TEST_SPREAD_SECONDS,
        config.BATCH_SIZE,
        config.MISFIRE_GRACE_TIME,
    )

    _db = DatabaseManager(connection_string=config.DB_CONNECTION_STRING, port_db_name=config.PORT_DB_NAME)
    _scheduler = PortScheduler(config=_config, db=_db)
    _scheduler.start()
