"""
Analyze HVAC sensor data to detect short-cycling events.

Short Cycling Definition:
A unit turns ON and then OFF within a 5-minute window more than 3 times in an hour.
"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np

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

def print_short_cycling_results(short_cycling_events):
    """
    Print short-cycling detection results.
    
    Parameters:
    -----------
    short_cycling_events : dict
        Dictionary mapping unit_id to list of timestamps
    """
    
    if not short_cycling_events:
        print("No short-cycling events detected.")
        return
    
    print("=" * 80)
    print("SHORT-CYCLING DETECTION RESULTS")
    print("=" * 80)
    print(f"\nDefinition: Unit turns ON and then OFF within a 5-minute window")
    print(f"more than 3 times in an hour.\n")
    
    for unit_id, timestamps in short_cycling_events.items():
        print(f"\n{'=' * 80}")
        print(f"Unit: {unit_id}")
        print(f"Number of short-cycling transitions detected: {len(timestamps)}")
        print(f"{'=' * 80}")
        
        # Group consecutive timestamps into events
        if timestamps:
            events = []
            current_event = [timestamps[0]]
            
            for i in range(1, len(timestamps)):
                # If timestamps are within 1 hour of each other, they're part of the same event
                if (timestamps[i] - current_event[-1]) <= timedelta(hours=1):
                    current_event.append(timestamps[i])
                else:
                    events.append(current_event)
                    current_event = [timestamps[i]]
            
            events.append(current_event)
            
            print(f"\nShort-cycling event periods:")
            for idx, event in enumerate(events, 1):
                start_time = event[0]
                end_time = event[-1]
                duration = end_time - start_time
                print(f"\n  Event {idx}:")
                print(f"    Start: {start_time}")
                print(f"    End:   {end_time}")
                print(f"    Duration: {duration}")
                print(f"    Transitions: {len(event)}")
                print(f"    Timestamps:")
                for ts in event:
                    print(f"      - {ts}")

def main():
    """Main function to load data and detect short-cycling."""
    
    csv_file = 'hvac_sensor_data.csv'
    
    print(f"Loading HVAC sensor data from {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records")
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Units: {', '.join(df['unit_id'].unique())}\n")
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        print("Please run 'generate_hvac_data.py' first to generate the data.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Detect short-cycling events
    print("Analyzing data for short-cycling events...")
    short_cycling_events = detect_short_cycling(df, energy_threshold=1.0)
    
    # Print results
    print_short_cycling_results(short_cycling_events)
    
    # Summary statistics
    if short_cycling_events:
        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print(f"{'=' * 80}")
        total_events = sum(len(ts) for ts in short_cycling_events.values())
        print(f"Total units with short-cycling: {len(short_cycling_events)}")
        print(f"Total short-cycling transitions: {total_events}")
        for unit_id, timestamps in short_cycling_events.items():
            print(f"  {unit_id}: {len(timestamps)} transitions")

if __name__ == '__main__':
    main()

