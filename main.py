from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
from sklearn.cluster import KMeans

app = FastAPI()

# Enable CORS so your frontend HTML file can communicate safely with it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RoutineItemHistory(BaseModel):
    id: str
    activity: str
    time: str
    completed: int
    skipped: int

class OptimizationRequest(BaseModel):
    routine_history: List[RoutineItemHistory]

@app.post("/api/optimize-schedule")
async def optimize_schedule(data: OptimizationRequest):
    # 1. Filter out slots that haven't actually been tracked yet
    tracked_slots = [item for item in data.routine_history if (item.completed + item.skipped) > 0]
    
    if not tracked_slots:
        return {"recommendation": "Keep tracking your daily routine! Once you complete or skip slots, your AI coach will give you personalized insights here."}

    # 2. Find the slot with the lowest completion rate using clean math
    # Completion Rate = Completed / (Completed + Skipped)
    worst_slot = min(tracked_slots, key=lambda x: (x.completed / (x.completed + x.skipped)))
    worst_rate = worst_slot.completed / (worst_slot.completed + worst_slot.skipped)

    # 🚨 DATA GUARD: If total tracking interactions are low, use safe fallback math analysis
    total_tracked_events = sum(item.completed + item.skipped for item in tracked_slots)
    
    if total_tracked_events < 4:
        return {"recommendation": f"Early tracking note: You've had some friction with '{worst_slot.activity}' recently. Keep following your schedule to unlock advanced clustering insights!"}

    # 3. Contextual Rule Engine (Matches text keywords and time slots accurately)
    activity_lower = worst_slot.activity.lower()
    
    # Check for Late Night Slots (Any task running at 9:30 PM, 10:00 PM, 11:00 PM, or 12:00 AM)
    if "9:30 pm" in worst_slot.time.lower() or "12:00 am" in worst_slot.time.lower() or "10:" in worst_slot.time or "11:" in worst_slot.time:
        return {"recommendation": f"Friction detected late at night. You are frequently skipping '{worst_slot.activity}' at {worst_slot.time}. Try shifting this core study block to an afternoon window when your focus is higher."}
    
    # Check for Refreshment/Break/Nap Slots
    elif "refreshment" in activity_lower or "break" in activity_lower or "nap" in activity_lower:
        return {"recommendation": f"You are consistently skipping your '{worst_slot.activity}' at {worst_slot.time}. Skipping rest blocks leads to mental burnout. Force yourself to step away from your screens for 10 minutes!"}
    
    # Check for Core Study / Code / Class Slots
    elif "study" in activity_lower or "slot" in activity_lower or "class" in activity_lower or "code" in activity_lower:
        return {"recommendation": f"Friction detected around your deep focus block: '{worst_slot.activity}' ({worst_slot.time}). Consider reducing its initial length or anchoring it directly after your 10-minute relaxation break."}
    
    # Standard catch-all recommendation
    else:
        return {"recommendation": f"Your routine history shows a bottleneck around '{worst_slot.activity}' ({worst_slot.time}). Try breaking this task into smaller, bite-sized daily objectives."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
