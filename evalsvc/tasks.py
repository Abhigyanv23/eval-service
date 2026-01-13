"""Async execution. With EVALSVC_REDIS_URL set, work goes to Celery workers; otherwise it runs on an
in-process thread pool so the service also works as a single process for local development."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from celery import Celery
from celery.signals import worker_init, worker_process_init
from sqlalchemy.exc import OperationalError

from evalsvc.config import settings

celery_app = Celery("evalsvc", broker=settings.redis_url or "memory://", backend=None)
celery_app.conf.update(
    task_acks_late=True,               # a worker crash re-queues the item instead of losing it
    worker_prefetch_multiplier=1,      # judge calls are slow; don't let one worker hoard items
    task_default_queue="evals",
    broker_connection_retry_on_startup=True,
)

@worker_init.connect
def _init_worker(**_):
    from evalsvc.db import SessionLocal, init_db
    from evalsvc.rubric_store import seed_rubrics

    init_db()
    with SessionLocal() as s:
        seed_rubrics(s)


@worker_process_init.connect
def _reset_db_pool(**_):
    # Prefork children inherit the parent's open Postgres connections; sharing one socket across processes
    # corrupts the protocol stream ("server closed the connection unexpectedly"). Give each child its own pool.
    from evalsvc.db import engine

    engine.dispose(close=False)


_local_pool = ThreadPoolExecutor(max_workers=max(settings.judge_concurrency, 1), thread_name_prefix="batch")


@celery_app.task(name="evalsvc.score_item", autoretry_for=(ConnectionError, OperationalError), retry_backoff=True,
                 max_retries=5)
def score_item_task(result_id: str) -> None:
    from evalsvc.service import process_result

    process_result(result_id)


@celery_app.task(name="evalsvc.calibrate")
def calibrate_task(calibration_run_id: str) -> None:
    from evalsvc.calibration import execute_calibration

    execute_calibration(calibration_run_id)


@celery_app.task(name="evalsvc.consistency")
def consistency_task(consistency_run_id: str) -> None:
    from evalsvc.consistency import execute_consistency

    execute_consistency(consistency_run_id)


def dispatch_consistency(consistency_run_id: str) -> None:
    from evalsvc.consistency import execute_consistency

    if settings.sync_tasks:
        execute_consistency(consistency_run_id)
    elif settings.redis_url:
        consistency_task.delay(consistency_run_id)
    else:
        _local_pool.submit(execute_consistency, consistency_run_id)


def dispatch_items(result_ids: list[str]) -> None:
    from evalsvc.service import process_result

    if settings.sync_tasks:
        for rid in result_ids:
            process_result(rid)
    elif settings.redis_url:
        for rid in result_ids:
            score_item_task.delay(rid)
    else:
        for rid in result_ids:
            _local_pool.submit(process_result, rid)


def dispatch_calibration(calibration_run_id: str) -> None:
    from evalsvc.calibration import execute_calibration

    if settings.sync_tasks:
        execute_calibration(calibration_run_id)
    elif settings.redis_url:
        calibrate_task.delay(calibration_run_id)
    else:
        _local_pool.submit(execute_calibration, calibration_run_id)
