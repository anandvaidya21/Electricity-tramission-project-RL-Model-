"""Configuration for Electricity Transmission Environment"""

import os
from dataclasses import dataclass

@dataclass
class TransmissionConfig:
    """Core simulation parameters"""
    # Grid topology
    NUM_SUBSTATIONS: int = 5
    NUM_CONSUMERS: int = 10
    NUM_GENERATORS: int = 3
    
    # Capacity limits (MW)
    TRANSMISSION_LINE_CAPACITY: float = 100.0
    SUBSTATION_STORAGE_CAPACITY: float = 50.0
    GENERATOR_MAX_OUTPUT: float = 80.0
    CONSUMER_MAX_DEMAND: float = 30.0
    
    # Time parameters
    EPISODE_LENGTH: int = 288  # 24 hours in 5-min steps
    SIMULATION_STEP_MINUTES: int = 5
    
    # Priority levels (1=highest, 5=lowest)
    PRIORITY_CRITICAL: int = 1      # Hospitals, emergency services
    PRIORITY_HIGH: int = 2          # Industrial (contracts)
    PRIORITY_MEDIUM: int = 3        # Commercial
    PRIORITY_LOW: int = 4           # Residential peak
    PRIORITY_LOWEST: int = 5        # Residential off-peak
    
    # Penalties and rewards
    PENALTY_UNMET_DEMAND: float = -1.0  # Per MWh
    PENALTY_CONGESTION: float = -0.5    # Per overload unit
    PENALTY_INSTABILITY: float = -0.3
    REWARD_EFFICIENCY: float = 0.1      # Bonus for balanced load
    REWARD_MET_PRIORITY: float = 0.2    # Bonus per priority level met
    
    # LLM Configuration
    API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

config = TransmissionConfig()