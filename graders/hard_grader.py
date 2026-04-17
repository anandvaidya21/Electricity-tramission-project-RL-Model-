"""Hard task: Multi-objective optimization with dynamic demand"""

from graders.base_grader import BaseGrader
from environment.env import ElectricityTransmissionEnv, ActionModel
from environment.config import config
import numpy as np

class HardGrader(BaseGrader):
    """
    Hard Task: Full multi-objective optimization
    Goal: Maximize composite score (demand met, priorities, stability, efficiency)
    Expected agent: RL-trained policy or advanced heuristic
    """
    
    def __init__(self):
        super().__init__(
            name="Hard: Multi-Objective Optimization",
            description="Maximize all objectives: demand, priorities, stability, efficiency"
        )
    
    async def grade(self, agent_callable, num_episodes: int = 3) -> float:
        """Grade agent on hard task"""
        scores = []
        
        for episode in range(num_episodes):
            env = ElectricityTransmissionEnv(seed=200 + episode)
            reset_resp, _ = env.reset()
            
            episode_composite_scores = []
            
            for step in range(config.EPISODE_LENGTH):
                current_obs = reset_resp.observation if step == 0 else env.state().observation
                obs_dict = current_obs.dict()
                
                action_dict = agent_callable(obs_dict)
                action = ActionModel(generator_actions=action_dict)
                
                step_resp = env.step(action)
                obs = step_resp.observation
                
                # Composite score from multiple objectives
                demand_met_pct = obs.total_met / max(obs.total_demand, 0.001)
                
                priority_met_pct = (
                    obs.critical_demand_met * 0.4 +
                    obs.high_demand_met * 0.3 +
                    obs.medium_demand_met * 0.2 +
                    obs.low_demand_met * 0.1
                ) / 100.0
                
                # Stability: frequency within ±0.5 Hz, voltage stable
                frequency_ok = 1.0 - abs(obs.grid_frequency - 50.0) / 5.0
                stability_score = (frequency_ok + obs.voltage_stability) / 2.0
                
                # Efficiency: low losses
                loss_ratio = obs.transmission_losses / max(obs.total_generation, 0.001)
                efficiency_score = max(0.0, 1.0 - loss_ratio)
                
                # Weighted composite
                composite = (
                    0.4 * demand_met_pct +
                    0.3 * priority_met_pct +
                    0.2 * stability_score +
                    0.1 * efficiency_score
                )
                
                episode_composite_scores.append(composite)
            
            episode_score = np.mean(episode_composite_scores)
            scores.append(episode_score)
            self.episode_rewards.append(episode_score)
        
        return float(np.mean(scores))