"""Reward calculation and metrics"""

from environment.simulator import GridState
from environment.config import config
from typing import Dict

class RewardCalculator:
    """Compute environment rewards"""
    
    @staticmethod
    def calculate_reward(state: GridState, unmet_demand: float, 
                        congestion_penalty: float) -> float:
        """
        Multi-objective reward function
        Returns normalized reward in [0, 1] range
        """
        reward = 0.0
        max_possible = 100.0
        
        # 1. Demand satisfaction (40%)
        demand_satisfaction = (state.total_met / max(state.total_demand, 0.001)) * 40
        reward += demand_satisfaction
        
        # 2. Priority fulfillment (30%)
        priority_score = 0.0
        for priority in range(1, 6):
            weight = (6 - priority) / 15.0  # Higher priority = higher weight
            priority_score += state.priority_met[priority] * weight
        reward += priority_score * 30
        
        # 3. Grid stability (20%)
        frequency_stability = max(0, 1 - abs(state.grid_frequency - 50.0) / 5.0)
        voltage_quality = state.voltage_stability
        stability_score = (frequency_stability + voltage_quality) / 2.0
        reward += stability_score * 20
        
        # 4. Efficiency (10%)
        loss_efficiency = 1.0 - (state.transmission_losses / max(state.total_generation, 0.001)) * 0.5
        reward += max(0, loss_efficiency * 10)
        
        # Normalize to [0, 1]
        normalized_reward = max(0.0, min(1.0, reward / max_possible))
        return normalized_reward
    
    @staticmethod
    def get_metrics(state: GridState) -> Dict:
        """Extract evaluation metrics from state"""
        return {
            "demand_met_pct": (state.total_met / max(state.total_demand, 0.001)) * 100,
            "critical_met_pct": state.priority_met.get(config.PRIORITY_CRITICAL, 0.0) * 100,
            "high_met_pct": state.priority_met.get(config.PRIORITY_HIGH, 0.0) * 100,
            "transmission_loss_pct": (state.transmission_losses / max(state.total_generation, 0.001)) * 100,
            "grid_frequency_hz": state.grid_frequency,
            "voltage_stability": state.voltage_stability,
            "total_cost_usd": state.total_cost
        }