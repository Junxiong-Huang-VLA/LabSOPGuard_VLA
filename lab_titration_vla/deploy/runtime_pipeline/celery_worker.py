from __future__ import annotations

import os
from celery import Celery

redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
app = Celery("labsopguard_worker", broker=redis_url, backend=redis_url)


@app.task(name="labsopguard.echo")
def echo_task(payload: dict) -> dict:
    return {"ok": True, "payload": payload}


if __name__ == "__main__":
    app.worker_main(
        [
            "worker",
            "--loglevel=INFO",
            "--pool=solo",
        ]
    )

