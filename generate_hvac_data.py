"""
Generate synthetic HVAC sensor data with realistic Hysteresis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_hvac_data(start_time=None, units=None, interval_minutes=5):
    if start_time is None:
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    if units is None:
        units = ['UNIT_001', 'UNIT_002', 'UNIT_003']
    
    # Calculate number of data points (24 hours)
    num_points = (24 * 60) // interval_minutes
    timestamps = [start_time + timedelta(minutes=i * interval_minutes) for i in range(num_points)]
    all_data = []
    
    # Short-cycling event parameters
    short_cycle_unit = "UNIT_001" # Fixed for testing
    short_cycle_start_idx = int((14 * 60) // interval_minutes)
    short_cycle_duration_rows = int(30 // interval_minutes) # 30 mins / 5 = 6 rows
    short_cycle_end_idx = short_cycle_start_idx + short_cycle_duration_rows
    
    print(f"Short-cycling event for {short_cycle_unit} at indices {short_cycle_start_idx}-{short_cycle_end_idx}")
    
    for unit_id in units:
        # --- 1. Temperature Generation ---
        hour_of_day = np.array([ts.hour + ts.minute/60 for ts in timestamps])
        base_temp = 68 + 6 * np.sin((hour_of_day - 6) * np.pi / 12)
        base_temp = np.clip(base_temp, 65, 78)
        # Reduce noise slightly to 0.5 to prevent extreme jumps, relying on hysteresis for cycle control
        temperature = base_temp + np.random.normal(0, 0.5, num_points)
        
        humidity = np.clip(50 - 15 * np.sin((hour_of_day - 6) * np.pi / 12) + np.random.normal(0, 2, num_points), 30, 70)
        error_code = np.zeros(num_points, dtype=int)
        energy_consumption = np.full(num_points, 0.5) # Idle baseline

        # --- 2. Stateful Thermostat Logic (Hysteresis) ---
        target_temp = 70
        is_running = False
        
        # We must loop to determine state, as current state depends on previous state
        for i in range(num_points):
            current_temp = temperature[i]
            
            # Hysteresis Logic:
            # COOLING MODE assumed (Active when hot)
            # Turn ON if > 74 (Target + 4)
            # Turn OFF if < 71 (Target + 1)
            # Else keep state
            
            if is_running:
                if current_temp < 71:
                    is_running = False
            else:
                if current_temp > 74:
                    is_running = True
            
            # Override for Short Cycling Event
            if unit_id == short_cycle_unit and short_cycle_start_idx <= i < short_cycle_end_idx:
                # Force toggle every 5 mins (High freq)
                cycle_state = (i - short_cycle_start_idx) % 2 
                is_running = (cycle_state == 0) # ON, OFF, ON, OFF...
                error_code[i] = 4

            # Apply Energy
            if is_running:
                # Add slight noise to running energy
                energy_consumption[i] = 2.5 + np.random.normal(0, 0.1)
                
                # Physics Feedback: If AC runs, temp should drop in next step
                # (Simple simulation: modify the FUTURE temp in the array)
                if i < num_points - 1:
                    temperature[i+1] -= 0.2 # Cool down effect
            else:
                # If OFF, temp naturally rises (recuperation)
                if i < num_points - 1:
                    temperature[i+1] += 0.1

        # Create DataFrame
        unit_data = pd.DataFrame({
            'timestamp': timestamps,
            'unit_id': unit_id,
            'temperature': np.round(temperature, 2),
            'humidity': np.round(humidity, 1),
            'energy_consumption': np.round(energy_consumption, 2),
            'error_code': error_code
        })
        all_data.append(unit_data)
    
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values(['timestamp', 'unit_id']).reset_index(drop=True)
    return df

def main():
    df = generate_hvac_data()
    df.to_csv('hvac_sensor_data.csv', index=False)
    print("New Hysteresis Data Generated.")

if __name__ == '__main__':
    main()