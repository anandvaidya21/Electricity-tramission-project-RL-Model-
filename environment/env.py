"""OpenEnv-compliant environment"""

from typing import Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
import numpy as np
from environment.simulator import PowerGridSimulator, GridState
from environment.reward import RewardCalculator
from environment.config import config

# ===== TYPED MODELS (OpenEnv Spec) =====

class ObservationModel(BaseModel):
    """Observation returned by step() and state()"""
    total_generation: float = Field(description="Total power generated (MW)")
    total_demand: float = Field(description="Total power demanded (MW)")
    total_met: float = Field(description="Total demand satisfied (MW)")
    critical_demand_met: float = Field(description="Critical priority demand satisfied (%)")
    high_demand_met: float = Field(description="High priority demand satisfied (%)")
    medium_demand_met: float = Field(description="Medium priority demand satisfied (%)")
    low_demand_met: float = Field(description="Low priority demand satisfied (%)")
    transmission_losses: float = Field(description="Power lost in transmission (MW)")
    grid_frequency: float = Field(description="Grid frequency (Hz)")
    voltage_stability: float = Field(description="Voltage stability (0-1)")
    time_step: int = Field(description="Current simulation step")
    total_cost: float = Field(description="Total generation cost ($)")

class ActionModel(BaseModel):
    """Action to take on the environment"""
    generator_actions: Dict[int, float] = Field(
        description="Dict mapping generator_id to target output (MW)"
    )

class StepResponseModel(BaseModel):
    """Response from step()"""
    observation: ObservationModel
    reward: float = Field(ge=0.0, le=1.0, description="Reward in [0, 1]")
    terminated: bool = Field(description="Episode complete")
    truncated: bool = Field(description="Max steps reached")
    info: Dict[str, Any] = Field(description="Additional info")

class StateResponseModel(BaseModel):
    """Response from state()"""
    observation: ObservationModel

class ResetResponseModel(BaseModel):
    """Response from reset()"""
    observation: ObservationModel
    info: Dict[str, Any] = Field(description="Initial info")

# ===== ENVIRONMENT CLASS =====

class ElectricityTransmissionEnv:
    """
    OpenEnv-compliant electricity transmission environment
    
    Agents learn to optimally dispatch generators to meet consumer demands
    while respecting transmission constraints and prioritizing critical loads.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.simulator = PowerGridSimulator(seed=seed or 42)
        self.episode_length = config.EPISODE_LENGTH
        self.step_count = 0
        self._episode_done = False
    
    def _convert_state_to_observation(self, state: GridState) -> ObservationModel:
        """Convert GridState to ObservationModel"""
        return ObservationModel(
            total_generation=float(state.total_generation),
            total_demand=float(state.total_demand),
            total_met=float(state.total_met),
            critical_demand_met=float(state.priority_met.get(config.PRIORITY_CRITICAL, 0.0) * 100),
            high_demand_met=float(state.priority_met.get(config.PRIORITY_HIGH, 0.0) * 100),
            medium_demand_met=float(state.priority_met.get(config.PRIORITY_MEDIUM, 0.0) * 100),
            low_demand_met=float(state.priority_met.get(config.PRIORITY_LOW, 0.0) * 100),
            transmission_losses=float(state.transmission_losses),
            grid_frequency=float(state.grid_frequency),
            voltage_stability=float(state.voltage_stability),
            time_step=int(state.time_step),
            total_cost=float(state.total_cost)
        )
    
    def reset(self) -> Tuple[ResetResponseModel, None]:
        """
        Reset environment to initial state
        OpenEnv spec: reset() -> (observation, info)
        """
        state = self.simulator.reset()
        self.step_count = 0
        self._episode_done = False
        
        observation = self._convert_state_to_observation(state)
        
        return ResetResponseModel(
            observation=observation,
            info={
                "episode": 0,
                "num_generators": len(self.simulator.generators),
                "num_consumers": len(self.simulator.consumers)
            }
        ), None
    
    def step(self, action: ActionModel) -> StepResponseModel:
        """
        Execute one environment step
        OpenEnv spec: step(action) -> (observation, reward, terminated, truncated, info)
        """
        # Execute action
        state, reward = self.simulator.step(action.generator_actions)
        
        # Calculate normalized reward
        unmet = state.total_demand - state.total_met
        congestion = sum(1.0 if util > 0.95 else 0.0 
                        for util in state.line_utilization.values())
        normalized_reward = RewardCalculator.calculate_reward(state, unmet, congestion)
        
        # Check termination
        self.step_count += 1
        terminated = self.step_count >= self.episode_length
        truncated = False
        
        observation = self._convert_state_to_observation(state)
        metrics = RewardCalculator.get_metrics(state)
        
        return StepResponseModel(
            observation=observation,
            reward=normalized_reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "step": self.step_count,
                "metrics": metrics,
                "unmet_demand": float(unmet),
                "congestion_score": float(congestion)
            }
        )
    
    def state(self) -> StateResponseModel:
        """
        Get current observation without stepping
        OpenEnv spec: state() -> observation
        """
        grid_state = self.simulator.get_state()
        observation = self._convert_state_to_observation(grid_state)
        return StateResponseModel(observation=observation)