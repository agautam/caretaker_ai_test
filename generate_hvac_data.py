"""
Generate synthetic HVAC sensor data for a 24-hour period.
Includes realistic temperature, humidity, energy consumption, and a short-cycling event.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_hvac_data(start_time=None, units=None, interval_minutes=5):
    """
    Generate synthetic HVAC sensor data.
    
    Parameters:
    -----------
    start_time : datetime, optional
        Start time for data generation. Defaults to current time.
    units : list, optional
        List of unit IDs. Defaults to ['UNIT_001', 'UNIT_002', 'UNIT_003'].
    interval_minutes : int
        Time interval between data points in minutes. Default is 5 minutes.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: timestamp, unit_id, temperature, humidity, 
        energy_consumption, error_code
    """
    
    if start_time is None:
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    if units is None:
        units = ['UNIT_001', 'UNIT_002', 'UNIT_003']
    
    # Calculate number of data points (24 hours)
    num_points = (24 * 60) // interval_minutes
    
    # Generate timestamps
    timestamps = [start_time + timedelta(minutes=i * interval_minutes) 
                  for i in range(num_points)]
    
    all_data = []
    
    # Short-cycling event parameters (randomly assign to one unit)
    short_cycle_unit = random.choice(units)
    # Short-cycling happens around 2-4 PM (hours 14-16)
    short_cycle_start_idx = int((14 * 60) // interval_minutes)
    short_cycle_duration = int((30 * 60) // interval_minutes)  # 30 minutes of rapid cycling
    
    print(f"Short-cycling event will occur for {short_cycle_unit} starting at index {short_cycle_start_idx}")
    
    for unit_id in units:
        # Base temperature varies throughout the day (cooler at night, warmer during day)
        hour_of_day = np.array([ts.hour + ts.minute/60 for ts in timestamps])
        
        # Temperature pattern: cooler at night (65-68°F), warmer during day (72-78°F)
        base_temp = 68 + 6 * np.sin((hour_of_day - 6) * np.pi / 12)
        base_temp = np.clip(base_temp, 65, 78)
        
        # Add some random variation
        temperature = base_temp + np.random.normal(0, 1.5, num_points)
        
        # Humidity inversely correlates with temperature (higher when cooler)
        base_humidity = 50 - 15 * np.sin((hour_of_day - 6) * np.pi / 12)
        base_humidity = np.clip(base_humidity, 30, 65)
        humidity = base_humidity + np.random.normal(0, 5, num_points)
        humidity = np.clip(humidity, 25, 70)
        
        # Energy consumption depends on HVAC activity
        # Higher consumption when temperature is outside comfort zone (68-72°F)
        target_temp = 70
        temp_deviation = np.abs(temperature - target_temp)
        
        # Base energy consumption (idle state)
        energy_consumption = np.full(num_points, 0.5)  # 0.5 kW idle
        
        # Add energy spikes when HVAC is active
        hvac_active = temp_deviation > 3  # HVAC kicks in when temp deviates >3°F
        energy_consumption[hvac_active] = 2.5 + np.random.normal(0, 0.3, np.sum(hvac_active))
        
        # Short-cycling event for the selected unit
        if unit_id == short_cycle_unit:
            short_cycle_end_idx = min(short_cycle_start_idx + short_cycle_duration, num_points)
            
            # During short-cycling, rapid on/off pattern (every 2-3 data points)
            for idx in range(short_cycle_start_idx, short_cycle_end_idx):
                cycle_state = (idx - short_cycle_start_idx) % 4  # 4-point cycle
                if cycle_state < 2:  # On for 2 points
                    energy_consumption[idx] = 3.0 + np.random.normal(0, 0.5, 1)[0]
                    temperature[idx] = target_temp - 1 + np.random.normal(0, 0.5, 1)[0]
                else:  # Off for 2 points
                    energy_consumption[idx] = 0.3 + np.random.normal(0, 0.1, 1)[0]
                    temperature[idx] = target_temp + 2 + np.random.normal(0, 0.5, 1)[0]
            
            print(f"Applied short-cycling pattern to {unit_id}")
        
        # Error codes: mostly 0 (normal), occasional errors
        error_code = np.zeros(num_points, dtype=int)
        
        # Random error events (1-2% of time)
        error_indices = np.random.choice(num_points, size=int(num_points * 0.015), replace=False)
        error_code[error_indices] = np.random.choice([1, 2, 3], size=len(error_indices))
        # Error codes: 1=High pressure, 2=Low refrigerant, 3=Sensor fault
        
        # Ensure short-cycling event has an error code
        if unit_id == short_cycle_unit:
            # Mark short-cycling period with error code 4 (short-cycling detected)
            error_code[short_cycle_start_idx:short_cycle_end_idx] = 4
        
        # Create DataFrame for this unit
        unit_data = pd.DataFrame({
            'timestamp': timestamps,
            'unit_id': unit_id,
            'temperature': np.round(temperature, 2),
            'humidity': np.round(humidity, 1),
            'energy_consumption': np.round(energy_consumption, 2),
            'error_code': error_code
        })
        
        all_data.append(unit_data)
    
    # Combine all units
    df = pd.concat(all_data, ignore_index=True)
    
    # Sort by timestamp and unit_id
    df = df.sort_values(['timestamp', 'unit_id']).reset_index(drop=True)
    
    return df

def main():
    """Main function to generate and save HVAC data."""
    
    print("Generating synthetic HVAC sensor data...")
    
    # Generate data
    df = generate_hvac_data(
        start_time=datetime(2024, 1, 15, 0, 0, 0),  # Start at midnight
        units=['UNIT_001', 'UNIT_002', 'UNIT_003'],
        interval_minutes=5  # 5-minute intervals (288 data points per unit)
    )
    
    # Save to CSV
    output_file = 'hvac_sensor_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\nData generated successfully!")
    print(f"Total records: {len(df)}")
    print(f"Time period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Units: {df['unit_id'].unique()}")
    print(f"\nData summary:")
    print(df.describe())
    print(f"\nError code distribution:")
    print(df['error_code'].value_counts().sort_index())
    print(f"\nData saved to: {output_file}")
    
    # Show short-cycling event details
    short_cycle_data = df[df['error_code'] == 4]
    if not short_cycle_data.empty:
        print(f"\nShort-cycling event detected:")
        print(f"Unit: {short_cycle_data['unit_id'].iloc[0]}")
        print(f"Duration: {len(short_cycle_data)} data points")
        print(f"Time range: {short_cycle_data['timestamp'].min()} to {short_cycle_data['timestamp'].max()}")
        print(f"\nSample of short-cycling data:")
        print(short_cycle_data[['timestamp', 'unit_id', 'temperature', 'energy_consumption', 'error_code']].head(10))

if __name__ == '__main__':
    main()

