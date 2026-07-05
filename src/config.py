from pathlib import Path
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok="True")

CHANNELS = ['Speed', 'RPM', 'Throttle', 'Brake', 'nGear']

TARGET_FREQ_HZ = 10
DT_SECONDS = 1.0 / TARGET_FREQ_HZ  # 0.1s grid intervals

WINDOW_SECONDS = 2.0
WINDOW_SIZE = int(WINDOW_SECONDS * TARGET_FREQ_HZ)  # 20
STEP_SIZE = 5  # 0.5s slide during training to prevent over-redundancy