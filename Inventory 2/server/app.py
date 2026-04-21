# server/app.py
# FastAPI server — this is what runs inside Docker and on Hugging Face Spaces.
# It exposes three HTTP endpoints:
#   POST /reset  → start a new episode
#   POST /step   → take one action
#   GET  /state  → read the current environment state

from fastapi import FastAPI
from pydantic import BaseModel
from env.environment import InventoryRestockEnvironment
from agent.baseline_agent import simple_agent
from env.grader import run_grader


app = FastAPI(
    title="Inventory Restock OpenEnv",
    description="A real-world RL environment for shop inventory restocking decisions.",
    version="1.0.0",
)

# One environment instance per server process.
# (For multi-session support, you would use per-WebSocket sessions.)
env = InventoryRestockEnvironment()


# ── Request models ──────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: int = 1  # 1=Easy, 2=Medium, 3=Hard


class StepRequest(BaseModel):
    restock_quantity: int  # 0-50


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Inventory Restock Environment",
        "tasks": {
            "1": "Easy - Stable demand, 1-day lead time",
            "2": "Medium - Variable demand, 3-day lead time",
            "3": "Hard - Demand spikes, 5-day lead time",
        },
        "endpoints": ["/reset", "/step", "/state"],
    }


@app.post("/reset")
def reset(request: ResetRequest | None = None):
    task_id = request.task_id if request else 1
    obs = env.reset(task_id=task_id)
    return obs.model_dump()


@app.post("/step")
def step(request: StepRequest):
    """Take one step: agent places (or skips) a restock order."""
    from env.models import RestockAction
    action = RestockAction(restock_quantity=request.restock_quantity)
    obs = env.step(action)
    return obs.model_dump()


@app.get("/state")
def state():
    """Return the full internal environment state."""
    return env.state.model_dump()
    
#new add
@app.post("/auto_step")
def auto_step():
    """Agent automatically decides restock action."""
    from env.models import RestockAction

    obs = env.state
    action = simple_agent(obs)
    obs = env.step(action)

    return obs.model_dump()


# ✅ CORRECT GRADER ENDPOINT
@app.get("/evaluate")
def evaluate():
    """
    Runs all 3 tasks and returns scores (HF REQUIRED)
    """
    return run_grader()
@app.get("/grade")
def grade():
    """
    Runs all 3 tasks and returns scores (REQUIRED for HF validation)
    """
    results = run_grader(env, simple_agent)
    return results
    # Get current observation from state
    obs = env.state

    # Let agent decide
    action = simple_agent(obs)

    # Apply action
    obs = env.step(action)

    return obs.model_dump()

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()