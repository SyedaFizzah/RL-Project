from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta, timezone
import numpy as np
import torch
import os
import sys
 # Ensure correct PyTorch version is installed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.database.models import Task, TimeLog, EnergyLog, get_db
from backend.rl.task_env import TaskSchedulerEnv
from backend.rl.dqn_agent import DQNAgent

app = FastAPI(title="Task Scheduler API")

# Allow the React frontend to call this API without being blocked
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained DQN agent once when the server starts
_env = TaskSchedulerEnv(seed=42)
_agent = DQNAgent(
    _env.observation_space.shape[0],
    _env.action_space.n
)
MODEL_PATH = "saved_model/dqn_model.pth"
if os.path.exists(MODEL_PATH):
    _agent.load(MODEL_PATH)
    _agent.epsilon = 0.0
    print("Trained DQN model loaded successfully.")
else:
    print("WARNING: No trained model found at saved_model/dqn_model.pth")
    print("Run train.py first before starting the server.")


# ── Request body schemas ───────────────────────────────────────────────────────

class TaskCreate(BaseModel):
    title: str
    category: str = "admin"
    priority: int = 2
    effort: int = 2
    deadline_days: int = 7

class EnergyInput(BaseModel):
    energy_level: int

class TimeLogBatch(BaseModel):
    intervals: List[dict]


# ── Task endpoints ─────────────────────────────────────────────────────────────

@app.get("/tasks")
def get_tasks(db: Session = Depends(get_db)):
    """Return all tasks that are not yet done."""
    tasks = db.query(Task).filter(Task.status != "done").all()
    return [
        {
            "id": t.id,
            "title": t.title,
            "category": t.category,
            "priority": t.priority,
            "effort": t.effort,
            "deadline_days": t.deadline_days,
            "status": t.status,
        }
        for t in tasks
    ]


@app.post("/tasks")
def create_task(task: TaskCreate, db: Session = Depends(get_db)):
    """Add a new task."""
    
    new_task = Task(**task.model_dump())
    db.add(new_task)
    db.commit()
    db.refresh(new_task)
    return {"id": new_task.id, "message": "Task created"}


@app.post("/tasks/{task_id}/complete")
def complete_task(task_id: int, db: Session = Depends(get_db)):
    """Mark a task as done."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task.status = "done"
    db.commit()
    return {"message": "Task marked complete"}


# ── Recommendation endpoint ────────────────────────────────────────────────────

@app.get("/recommend")
def recommend_task(energy: int = 3, db: Session = Depends(get_db)):
    """
    Ask the trained DQN agent which task to do next.
    Builds the 36-dim state vector from real DB tasks and returns
    the top pick plus the full Q-value ranking.
    """
    pending = db.query(Task).filter(Task.status == "pending").all()

    if not pending:
        return {"message": "No pending tasks.", "recommended": None, "all_ranked": []}

    # Build task feature block — pad to 8 tasks with zeros if fewer exist
    task_features = []
    task_map = {}
    for i in range(8):
        if i < len(pending):
            t = pending[i]
            task_map[i] = t
            features = [
                t.priority / 3.0,
                t.effort / 4.0,
                min(t.deadline_days, 7) / 7.0,
                0.0,
            ]
        else:
            features = [0.0, 0.0, 0.0, 0.0]
        task_features.extend(features)

    # Compute productivity score from recent Chrome extension data
    two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
    recent = db.query(TimeLog).filter(TimeLog.logged_at >= two_hours_ago).all()
    productive   = sum(l.duration_seconds for l in recent if l.category == "productive")
    distracting  = sum(l.duration_seconds for l in recent if l.category == "distracting")
    total        = sum(l.duration_seconds for l in recent) or 1
    productivity = float(np.clip((productive - distracting) / total, -1.0, 1.0))

    current_hour = datetime.now().hour
    slot = max(0, min(18, int((current_hour - 9) * 60 / 25)))

    global_features = [
        slot / 19.0,
        energy / 5.0,
        productivity,
        len(pending) / 8.0,
    ]

    state = np.array(task_features + global_features, dtype=np.float32)

    # Get Q-values from the trained network
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = _agent.main_net(state_tensor).squeeze(0).numpy()

    # Build ranked list of real tasks only
    ranked = []
    for i, q in enumerate(q_values[:8]):
        if i in task_map:
            ranked.append({
                "task_id":  task_map[i].id,
                "title":    task_map[i].title,
                "q_value":  round(float(q), 3),
                "priority": task_map[i].priority,
                "deadline": task_map[i].deadline_days,
            })
    ranked.sort(key=lambda x: x["q_value"], reverse=True)

    return {
        "recommended":        ranked[0] if ranked else None,
        "all_ranked":         ranked,
        "energy_used":        energy,
        "productivity_score": round(productivity, 3),
    }


# ── Supporting endpoints ───────────────────────────────────────────────────────

@app.post("/energy")
def log_energy(data: EnergyInput, db: Session = Depends(get_db)):
    """Log the user's current energy level."""
    db.add(EnergyLog(energy_level=data.energy_level))
    db.commit()
    return {"message": "Energy logged"}


@app.post("/timelog")
def receive_timelog(data: TimeLogBatch, db: Session = Depends(get_db)):
    """Receive a batch of time intervals from the Chrome extension."""
    for interval in data.intervals:
        db.add(TimeLog(
            hostname=interval.get("hostname", ""),
            duration_seconds=interval.get("durationSeconds", 0),
            category=interval.get("category", "neutral"),
            start_timestamp=interval.get("startTimestamp", 0),
        ))
    db.commit()
    return {"message": f"{len(data.intervals)} intervals logged"}


@app.get("/report/weekly")
def weekly_report(db: Session = Depends(get_db)):
    """Return weekly productivity summary."""
    one_week_ago = datetime.now(timezone.utc) - timedelta(days=7)

    completed = db.query(Task).filter(Task.status == "done").count()
    pending   = db.query(Task).filter(Task.status == "pending").count()

    energy_logs = db.query(EnergyLog)\
                    .filter(EnergyLog.logged_at >= one_week_ago).all()
    avg_energy = (
        round(sum(l.energy_level for l in energy_logs) / len(energy_logs), 2)
        if energy_logs else 0
    )

    return {
        "completed_tasks":  completed,
        "pending_tasks":    pending,
        "average_energy":   avg_energy,
        "total_sessions":   len(energy_logs),
    }
