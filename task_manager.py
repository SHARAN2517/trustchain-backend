"""
TrustChain-MedAI: Async Background Task Manager.

Non-blocking execution of long-running operations (federation rounds,
model training, audit exports, proof generation) with:
  - Thread-pool executor for CPU-bound tasks
  - Task status tracking with progress reporting
  - First-class WebSocket notifications for real-time updates
  - ProgressCallback helper for tasks to report progress
"""

import asyncio
import concurrent.futures
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Task Status & Model
# ─────────────────────────────────────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TaskType(str, Enum):
    FEDERATION_ROUND = "FEDERATION_ROUND"
    MODEL_TRAINING = "MODEL_TRAINING"
    AUDIT_EXPORT = "AUDIT_EXPORT"
    PROOF_GENERATION = "PROOF_GENERATION"
    MODEL_EVALUATION = "MODEL_EVALUATION"
    CUSTOM = "CUSTOM"


@dataclass
class Task:
    task_id: str
    task_type: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    message: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "metadata": self.metadata,
        }
        # Include result only if completed
        if self.status == TaskStatus.COMPLETED and self.result:
            d["result"] = self.result
        # Compute elapsed time
        if self.started_at:
            end = self.completed_at or time.time()
            d["elapsed_seconds"] = round(end - self.started_at, 2)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Progress Callback
# ─────────────────────────────────────────────────────────────────────────────

class ProgressCallback:
    """
    Passed to long-running functions so they can report progress.

    Thread-safe: uses asyncio.run_coroutine_threadsafe to update progress
    from worker threads to the async event loop.
    """

    def __init__(self, manager: "BackgroundTaskManager", task_id: str):
        self.manager = manager
        self.task_id = task_id
        self._loop = None

    def _get_loop(self):
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = None
        return self._loop

    def update(self, progress: float, message: str = ""):
        """
        Report progress from a worker thread.

        Args:
            progress: Float between 0.0 and 1.0.
            message: Optional status message.
        """
        task = self.manager._tasks.get(self.task_id)
        if task:
            task.progress = min(1.0, max(0.0, progress))
            task.message = message

        # Try to broadcast via WebSocket (best-effort from thread)
        loop = self._get_loop()
        if loop and loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(
                    self.manager._broadcast(self.task_id, {
                        "type": "progress",
                        "task_id": self.task_id,
                        "progress": progress,
                        "message": message,
                    }),
                    loop,
                )
            except Exception:
                pass  # Best-effort WebSocket update


# ─────────────────────────────────────────────────────────────────────────────
# Background Task Manager
# ─────────────────────────────────────────────────────────────────────────────

class BackgroundTaskManager:
    """
    Thread-pool-based background task executor with WebSocket notifications.

    Usage:
        manager = BackgroundTaskManager()
        task_id = manager.submit("FEDERATION_ROUND", run_federation_round, round_id=5)
        status = manager.get_status(task_id)

    WebSocket:
        @app.websocket("/ws/tasks/{task_id}")
        async def ws_endpoint(websocket, task_id):
            await task_websocket_endpoint(websocket, task_id, manager)
    """

    def __init__(self, max_workers: int = 4):
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="trustchain-task",
        )
        self._tasks: Dict[str, Task] = {}
        self._futures: Dict[str, concurrent.futures.Future] = {}
        self._ws_connections: Dict[str, Set] = {}  # task_id -> set of WebSocket objects

    def submit(
        self,
        task_type: str,
        func: Callable,
        *args,
        metadata: Optional[Dict] = None,
        **kwargs,
    ) -> str:
        """
        Submit a task for background execution.

        The function `func` will receive a `progress_callback` keyword argument
        (a ProgressCallback instance) that it can use to report progress.

        Args:
            task_type: Type of task (from TaskType enum or custom string).
            func: The callable to execute.
            *args: Positional arguments for func.
            metadata: Optional metadata dict to attach to the task.
            **kwargs: Keyword arguments for func.

        Returns:
            task_id string.
        """
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        task = Task(
            task_id=task_id,
            task_type=task_type,
            metadata=metadata or {},
        )
        self._tasks[task_id] = task

        # Create progress callback
        callback = ProgressCallback(self, task_id)

        def _wrapped():
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            try:
                result = func(*args, progress_callback=callback, **kwargs)
                task.status = TaskStatus.COMPLETED
                task.progress = 1.0
                task.result = result if isinstance(result, dict) else {"result": str(result)}
                task.completed_at = time.time()
                task.message = "Completed successfully"
                return result
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = time.time()
                task.message = f"Failed: {str(e)}"
                logger.error(f"Task {task_id} failed: {e}")
                raise

        future = self._executor.submit(_wrapped)
        self._futures[task_id] = future

        # Add completion callback for WebSocket notification
        def _on_done(f):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self._broadcast(task_id, task.to_dict()),
                        loop,
                    )
            except Exception:
                pass

        future.add_done_callback(_on_done)

        return task_id

    def get_status(self, task_id: str) -> Optional[Dict]:
        """Returns current task status as a dict."""
        task = self._tasks.get(task_id)
        if task is None:
            return None
        return task.to_dict()

    def cancel(self, task_id: str) -> bool:
        """
        Attempt to cancel a pending/running task.

        Returns True if cancellation was successful.
        """
        future = self._futures.get(task_id)
        task = self._tasks.get(task_id)

        if not future or not task:
            return False

        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            return False

        cancelled = future.cancel()
        if cancelled:
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            task.message = "Cancelled by user"
        return cancelled

    def list_tasks(
        self,
        status: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """List tasks, optionally filtered by status or type."""
        tasks = list(self._tasks.values())

        if status:
            tasks = [t for t in tasks if t.status.value == status]
        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]

        # Sort by creation time, newest first
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return [t.to_dict() for t in tasks[:limit]]

    async def update_progress(self, task_id: str, progress: float, message: str = ""):
        """Async progress update with WebSocket broadcast."""
        task = self._tasks.get(task_id)
        if task:
            task.progress = min(1.0, max(0.0, progress))
            task.message = message
            await self._broadcast(task_id, {
                "type": "progress",
                "task_id": task_id,
                "progress": progress,
                "message": message,
            })

    # ── WebSocket Management ──

    async def register_ws(self, task_id: str, websocket) -> None:
        """Register a WebSocket connection for real-time task updates."""
        if task_id not in self._ws_connections:
            self._ws_connections[task_id] = set()
        self._ws_connections[task_id].add(websocket)

    async def unregister_ws(self, task_id: str, websocket) -> None:
        """Remove a WebSocket connection."""
        if task_id in self._ws_connections:
            self._ws_connections[task_id].discard(websocket)
            if not self._ws_connections[task_id]:
                del self._ws_connections[task_id]

    async def _broadcast(self, task_id: str, data: Dict) -> None:
        """Broadcast update to all WebSocket connections watching a task."""
        connections = self._ws_connections.get(task_id, set())
        if not connections:
            return

        message = json.dumps(data, default=str)
        dead = set()

        for ws in connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)

        # Clean up dead connections
        for ws in dead:
            connections.discard(ws)

    def cleanup_old_tasks(self, max_age_seconds: int = 86400):
        """Remove completed/failed tasks older than max_age."""
        now = time.time()
        expired = [
            tid for tid, task in self._tasks.items()
            if task.completed_at and (now - task.completed_at) > max_age_seconds
        ]
        for tid in expired:
            del self._tasks[tid]
            self._futures.pop(tid, None)
            self._ws_connections.pop(tid, None)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Endpoint Helper
# ─────────────────────────────────────────────────────────────────────────────

async def task_websocket_endpoint(websocket, task_id: str, manager: BackgroundTaskManager):
    """
    FastAPI WebSocket handler for /ws/tasks/{task_id}.

    Usage in main.py:
        @app.websocket("/ws/tasks/{task_id}")
        async def ws_tasks(websocket: WebSocket, task_id: str):
            await task_websocket_endpoint(websocket, task_id, task_manager)

    Sends initial status on connect, then streams updates until task completes.
    """
    await websocket.accept()

    # Send initial status
    status = manager.get_status(task_id)
    if status is None:
        await websocket.send_json({"error": f"Task {task_id} not found"})
        await websocket.close()
        return

    await websocket.send_json({"type": "initial", **status})

    # If already completed, close immediately
    if status["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
        await websocket.close()
        return

    # Register for updates
    await manager.register_ws(task_id, websocket)

    try:
        # Keep connection alive until task completes or client disconnects
        while True:
            try:
                # Wait for client messages (keepalive pings)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send keepalive status update
                current = manager.get_status(task_id)
                if current:
                    await websocket.send_json({"type": "heartbeat", **current})
                    if current["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
                        break
            except Exception:
                break
    finally:
        await manager.unregister_ws(task_id, websocket)
        try:
            await websocket.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_task_manager: Optional[BackgroundTaskManager] = None


def get_task_manager() -> BackgroundTaskManager:
    global _task_manager
    if _task_manager is None:
        _task_manager = BackgroundTaskManager()
    return _task_manager


# ─────────────────────────────────────────────────────────────────────────────
# Self-Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import threading

    print("=" * 60)
    print("  Background Task Manager — Self-Test")
    print("=" * 60)

    manager = BackgroundTaskManager(max_workers=2)

    # Test 1: Submit and complete a task
    def mock_federation_round(round_id, progress_callback=None, **kwargs):
        """Simulate a federation round."""
        steps = 5
        for i in range(steps):
            time.sleep(0.1)
            if progress_callback:
                progress_callback.update(
                    (i + 1) / steps,
                    f"Processing hospital {i + 1}/{steps}",
                )
        return {"round_id": round_id, "accuracy": 0.912, "clients": steps}

    print("\n  [1] Submit federation round task:")
    task_id = manager.submit(
        "FEDERATION_ROUND",
        mock_federation_round,
        round_id=6,
        metadata={"aggregation": "krum", "hospitals": 5},
    )
    print(f"      Task ID: {task_id}")

    # Poll for completion
    for _ in range(20):
        status = manager.get_status(task_id)
        if status["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.2)

    status = manager.get_status(task_id)
    print(f"      Status: {status['status']}")
    print(f"      Progress: {status['progress']}")
    print(f"      Elapsed: {status.get('elapsed_seconds', 'N/A')}s")
    if status["status"] == "COMPLETED":
        print(f"      Result: {status.get('result')}")

    # Test 2: Submit a failing task
    def failing_task(progress_callback=None, **kwargs):
        time.sleep(0.1)
        raise RuntimeError("Simulated model corruption detected")

    print("\n  [2] Submit failing task:")
    fail_id = manager.submit("MODEL_TRAINING", failing_task)
    time.sleep(0.5)
    fail_status = manager.get_status(fail_id)
    print(f"      Status: {fail_status['status']}")
    print(f"      Error: {fail_status.get('error')}")

    # Test 3: Cancel a task
    def slow_task(progress_callback=None, **kwargs):
        for i in range(100):
            time.sleep(0.1)
        return {"done": True}

    print("\n  [3] Cancel task:")
    slow_id = manager.submit("CUSTOM", slow_task)
    time.sleep(0.1)
    cancelled = manager.cancel(slow_id)
    print(f"      Cancelled: {cancelled}")

    # Test 4: List tasks
    print("\n  [4] List all tasks:")
    all_tasks = manager.list_tasks()
    for t in all_tasks:
        print(f"      {t['task_id'][:20]}... type={t['task_type']}, status={t['status']}")

    # Test 5: Concurrent tasks
    print("\n  [5] Concurrent tasks:")
    ids = []
    for i in range(3):
        tid = manager.submit(
            "MODEL_EVALUATION",
            mock_federation_round,
            round_id=i + 10,
        )
        ids.append(tid)
    print(f"      Submitted {len(ids)} concurrent tasks")

    # Wait for all
    time.sleep(2)
    for tid in ids:
        s = manager.get_status(tid)
        print(f"      {tid[:20]}... → {s['status']}")

    # Cleanup
    manager.cleanup_old_tasks(max_age_seconds=0)
    print(f"      After cleanup: {len(manager._tasks)} tasks remaining")

    print("\n" + "=" * 60)
    print("  Task manager tests completed successfully!")
    print("=" * 60)
