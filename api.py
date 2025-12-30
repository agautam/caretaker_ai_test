"""
FastAPI application for HVAC sensor data analysis.
Implements the "Digital Twin" schema using Pydantic for explicit AI contracts.
"""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field
import os

# --- 1. THE DIGITAL TWIN SCHEMA (Pydantic) ---
class AnomalyEvent(BaseModel):
    issue_type: str = Field(..., description="The classification of the issue (e.g., SHORT_CYCLING)")
    severity: Literal["LOW", "MEDIUM", "CRITICAL"]
    timestamps: List[str]
    description: str

class SystemHealth(BaseModel):
    unit_id: str
    status: Literal["HEALTHY", "WARNING", "CRITICAL"] = Field(..., description="High-level health flag")
    last_contact: str
    anomalies: List[AnomalyEvent] = []
    raw_metrics_summary: Dict[str, float] = {}

# Global variable to store the DataFrame
hvac_df = None

# --- 2. CORE LOGIC ---
def detect_short_cycling_for_unit(unit_df, energy_threshold=1.0) -> List[str]:
    """
    Detects short-cycling specifically for a single unit's dataframe.
    Returns a list of timestamps where the pattern was found.
    """
    if unit_df.empty:
        return []

    # Sort
    df = unit_df.sort_values('timestamp').reset_index(drop=True)
    
    # State: 1 = ON, 0 = OFF
    df['state'] = (df['energy_consumption'] > energy_threshold).astype(int)
    
    # Detect transitions (ON -> OFF)
    # diff() gives 1 if 0->1, -1 if 1->0. We want 1->0 (falling edge)
    # But simpler logic:
    df['prev_state'] = df['state'].shift(1)
    # Transition is True where Prev was 1 (ON) and Current is 0 (OFF)
    transitions = df[(df['prev_state'] == 1) & (df['state'] == 0)].copy()
    
    if len(transitions) == 0:
        return []

    short_cycle_timestamps = []
    
    # Sliding window logic
    for i in range(len(transitions)):
        current_time = transitions.iloc[i]['timestamp']
        window_end = current_time + timedelta(hours=1)
        
        transitions_in_window = transitions[
            (transitions['timestamp'] >= current_time) & 
            (transitions['timestamp'] < window_end)
        ]
        
        # Rule: > 3 transitions in 1 hour
        if len(transitions_in_window) >= 3:
            for _, row in transitions_in_window.iterrows():
                if row['timestamp'] not in short_cycle_timestamps:
                    short_cycle_timestamps.append(row['timestamp'])
                    
    return sorted(list(set(short_cycle_timestamps)))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data when the app starts."""
    global hvac_df
    csv_file = 'hvac_sensor_data.csv'
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"HVAC data file '{csv_file}' not found. Run 'generate_hvac_data.py' first.")
    
    print(f"Loading HVAC sensor data from {csv_file}...")
    hvac_df = pd.read_csv(csv_file)
    hvac_df['timestamp'] = pd.to_datetime(hvac_df['timestamp'])
    
    print(f"Loaded {len(hvac_df)} records")
    yield
    hvac_df = None

app = FastAPI(
    title="Caretaker Hubbub API",
    description="Digital Twin interface for HVAC fleet monitoring",
    version="2.0.0",
    lifespan=lifespan
)

# --- 3. ENDPOINTS ---

@app.get("/")
async def root():
    return {
        "message": "Caretaker Hubbub API v2.0",
        "endpoints": {
            "/health/{unit_id}": "GET - Returns Digital Twin object (SystemHealth)"
        }
    }

@app.get("/health/{unit_id}", response_model=SystemHealth)
async def get_unit_health(unit_id: str):
    """
    Returns the Digital Twin status for a specific unit.
    The API (not the AI) determines if the status is HEALTHY or CRITICAL.
    """
    global hvac_df
    if hvac_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    # Filter for unit
    unit_df = hvac_df[hvac_df['unit_id'] == unit_id].copy()
    if unit_df.empty:
        raise HTTPException(status_code=404, detail=f"Unit {unit_id} not found")

    # --- Run Analysis ---
    short_cycle_ts = detect_short_cycling_for_unit(unit_df)
    
    # --- Build the Twin State ---
    anomalies = []
    status = "HEALTHY"
    
    # If logic detects issues, escalate status
    if short_cycle_ts:
        status = "CRITICAL"
        anomalies.append(AnomalyEvent(
            issue_type="SHORT_CYCLING",
            severity="CRITICAL",
            timestamps=[ts.isoformat() for ts in short_cycle_ts],
            description=f"Unit cycled ON/OFF {len(short_cycle_ts)} times rapidly, exceeding safety threshold."
        ))

    # Calculate latest metrics for context
    latest_row = unit_df.loc[unit_df['timestamp'].idxmax()]
    
    return SystemHealth(
        unit_id=unit_id,
        status=status,
        last_contact=latest_row['timestamp'].isoformat(),
        anomalies=anomalies,
        raw_metrics_summary={
            "current_temp": float(latest_row['temperature']),
            "current_energy": float(latest_row['energy_consumption']),
            "error_code": float(latest_row['error_code']) # 0.0 usually means fine
        }
    )

@app.get("/latest")
async def get_latest():
    """Legacy endpoint for raw data checks."""
    global hvac_df
    if hvac_df is None: return {}
    latest = hvac_df.loc[hvac_df['timestamp'].idxmax()]
    res = latest.to_dict()
    res['timestamp'] = res['timestamp'].isoformat()
    return res