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
    tracked_slots = [item for item in data.routine_history if (item.completed + item.skipped) > 0]
    
    if not tracked_slots:
        return {"recommendation": "Keep tracking your daily routine! Once you complete or skip slots, your AI coach will give you personalized insights here."}

    # --- 🤖 1. K-MEANS CLUSTERING ENGINE ---
    # Prepare data points: [completion_rate, total_tracked_count]
    features = []
    for item in tracked_slots:
        total = item.completed + item.skipped
        comp_rate = item.completed / total
        features.append([comp_rate, total])

    # K-Means requires at least as many samples as clusters (2 clusters = min 2 items tracked)
    ml_detected_problems = []
    if len(features) >= 2:
        X = np.array(features)
        
        # Split routine into 2 groups: High Performance vs Friction Vectors
        kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
        clusters = kmeans.fit_predict(X)
        
        # Identify which cluster center represents the lower completion rate
        cluster_centers = kmeans.cluster_centers_
        low_perf_cluster_idx = np.argmin(cluster_centers[:, 0])
        
        # Filter out the specific activities grouped into the low-efficiency cluster
        ml_detected_problems = [tracked_slots[i] for i, c in enumerate(clusters) if c == low_perf_cluster_idx]

    # --- 🎯 2. DECISION MATRIX & RECOMMENDATION ---
    # Prioritize clustering insights if ML found distinct friction points, otherwise fall back to math min
    if ml_detected_problems:
        worst_slot = min(ml_detected_problems, key=lambda x: (x.completed / (x.completed + x.skipped)))
    else:
        worst_slot = min(tracked_slots, key=lambda x: (x.completed / (x.completed + x.skipped)))

    # --- 💬 3. CONTEXTUAL OVERRIDES ---
    activity_lower = worst_slot.activity.lower()
    
    # Late Night Filter
    if any(time_key in worst_slot.time.lower() for time_key in ["9:30 pm", "12:00 am", "10:", "11:"]):
        return {"recommendation": f"Friction detected late at night. You are frequently skipping '{worst_slot.activity}' at {worst_slot.time}. Try shifting this core study block to an afternoon window when your focus is higher."}
    
    # Rest / Refreshment Filter
    elif any(rest_key in activity_lower for rest_key in ["refreshment", "break", "nap"]):
        return {"recommendation": f"You are consistently skipping your '{worst_slot.activity}' at {worst_slot.time}. Skipping rest blocks leads to mental burnout. Force yourself to step away from your screens for 10 minutes!"}
    
    # Core Technical/Study Focus Filter
    elif any(study_key in activity_lower for study_key in ["study", "slot", "class", "code"]):
        return {"recommendation": f"Friction detected around your deep focus block: '{worst_slot.activity}' ({worst_slot.time}). Consider reducing its initial length or anchoring it directly after your 10-minute relaxation break."}
    
    # Standard Catch-all
    else:
        return {"recommendation": f"Your routine history shows a bottleneck around '{worst_slot.activity}' ({worst_slot.time}). Try breaking this task into smaller, bite-sized daily objectives."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
