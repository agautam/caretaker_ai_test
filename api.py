"""
FastAPI application for HVAC sensor data analysis.

Endpoints:
- GET /latest: Returns the most recent sensor reading
- GET /diagnose: Detects and returns short-cycling events
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import os

# Global variable to store the DataFrame
hvac_df = None

def detect_short_cycling(df, energy_threshold=1.0):
    """
    Detect short-cycling events in HVAC data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: timestamp, unit_id, energy_consumption
    energy_threshold : float
        Energy consumption threshold to determine ON/OFF state (kW)
        Values above threshold = ON, below = OFF
    
    Returns:
    --------
    dict
        Dictionary mapping unit_id to list of short-cycling event timestamps
    """
    
    # Convert timestamp to datetime if it's a string
    if df['timestamp'].dtype == 'object':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by unit_id and timestamp
    df = df.sort_values(['unit_id', 'timestamp']).reset_index(drop=True)
    
    # Determine ON/OFF state based on energy consumption
    df['state'] = df['energy_consumption'] > energy_threshold
    df['state'] = df['state'].astype(int)  # True=1 (ON), False=0 (OFF)
    
    short_cycling_events = {}
    
    # Process each unit separately
    for unit_id in df['unit_id'].unique():
        unit_data = df[df['unit_id'] == unit_id].copy()
        unit_data = unit_data.sort_values('timestamp').reset_index(drop=True)
        
        # Detect state transitions (ON->OFF)
        unit_data['prev_state'] = unit_data['state'].shift(1)
        unit_data['state_change'] = (unit_data['prev_state'] == 1) & (unit_data['state'] == 0)
        
        # Get timestamps of ON->OFF transitions
        transitions = unit_data[unit_data['state_change'] == True].copy()
        
        if len(transitions) == 0:
            continue
        
        # For each transition, check if it's part of a short-cycling pattern
        # We need to find periods where there are >3 ON->OFF transitions within 1 hour
        
        short_cycle_timestamps = []
        
        # Use a sliding 1-hour window
        for i in range(len(transitions)):
            current_time = transitions.iloc[i]['timestamp']
            window_end = current_time + timedelta(hours=1)
            
            # Count transitions in this 1-hour window
            transitions_in_window = transitions[
                (transitions['timestamp'] >= current_time) & 
                (transitions['timestamp'] < window_end)
            ]
            
            if len(transitions_in_window) > 3:
                # This is a short-cycling event
                # Add all timestamps in this window to the list
                for _, row in transitions_in_window.iterrows():
                    if row['timestamp'] not in short_cycle_timestamps:
                        short_cycle_timestamps.append(row['timestamp'])
        
        # Remove duplicates and sort
        short_cycle_timestamps = sorted(list(set(short_cycle_timestamps)))
        
        if short_cycle_timestamps:
            short_cycling_events[unit_id] = short_cycle_timestamps
    
    return short_cycling_events

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data when the app starts."""
    global hvac_df
    
    csv_file = 'hvac_sensor_data.csv'
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"HVAC data file '{csv_file}' not found. "
            "Please run 'generate_hvac_data.py' first to generate the data."
        )
    
    print(f"Loading HVAC sensor data from {csv_file}...")
    hvac_df = pd.read_csv(csv_file)
    
    # Convert timestamp to datetime
    hvac_df['timestamp'] = pd.to_datetime(hvac_df['timestamp'])
    
    print(f"Loaded {len(hvac_df)} records")
    print(f"Time range: {hvac_df['timestamp'].min()} to {hvac_df['timestamp'].max()}")
    print(f"Units: {', '.join(hvac_df['unit_id'].unique())}")
    
    yield
    
    # Cleanup (if needed)
    hvac_df = None

app = FastAPI(
    title="HVAC Sensor Data API",
    description="API for querying HVAC sensor data and detecting short-cycling events",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "HVAC Sensor Data API",
        "endpoints": {
            "/latest": "GET - Returns the most recent sensor reading",
            "/diagnose": "GET - Detects and returns short-cycling events"
        }
    }

@app.get("/latest")
async def get_latest():
    """
    Returns the very last row of data (current status).
    
    Returns the most recent sensor reading across all units.
    """
    global hvac_df
    
    if hvac_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    # Get the row with the latest timestamp
    latest_row = hvac_df.loc[hvac_df['timestamp'].idxmax()]
    
    # Convert to dictionary and format timestamp as string
    result = latest_row.to_dict()
    result['timestamp'] = result['timestamp'].isoformat()
    
    return result

@app.get("/diagnose")
async def diagnose():
    """
    Runs the short-cycling detection logic and returns a JSON list of timestamps 
    where issues were found.
    
    Returns a dictionary mapping unit_id to a list of timestamps where 
    short-cycling events were detected.
    """
    global hvac_df
    
    if hvac_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    # Run short-cycling detection
    short_cycling_events = detect_short_cycling(hvac_df, energy_threshold=1.0)
    
    # Convert timestamps to ISO format strings for JSON serialization
    result = {}
    for unit_id, timestamps in short_cycling_events.items():
        result[unit_id] = [ts.isoformat() for ts in timestamps]
    
    return result

