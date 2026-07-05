import fastf1
import pandas as pd
from src.config import DATA_DIR,CHANNELS

fastf1.Cache.enable_cache(str(DATA_DIR))

def load_clean_session_laps(year , gp , session_type , driver):
    print(f"Loading {year} {gp} - Session: {session_type} for Driver: {driver}...")
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, laps=True, weather=False)

    clean_laps = session.laps.pick_driver(driver).pick_quicklaps()
    lap_telemetry_list = []
    for lap_idx, lap in clean_laps.iterlaps():
        tel = lap.get_telemetry()
        
        # Ensure required channels exist and extract Time (timedelta)
        if all(col in tel.columns for col in CHANNELS):
            clean_tel = tel[['Time'] + CHANNELS].copy()
            lap_telemetry_list.append(clean_tel)
            
    print(f"Successfully extracted {len(lap_telemetry_list)} clean laps.")
    return lap_telemetry_list

