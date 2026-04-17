"""Medium task: Meet priorities while keeping transmission <90% utilization"""

from graders.base_grader import BaseGrader
from environment.env import ElectricityTransmissionEnv, ActionModel
from environment.config import config
import numpy as np

class MediumGrader(BaseGrader):
    """
    Medium Task: Respect priority hierarchy
    Goal: Meet 95% of critical+high priority demand, keep congestion <10%
    Expected agent: Simple heuristic-based
    """
    
    def __init__(self):
        super().__init__(
            name="Medium: Priority-Aware Dispatch",
            description="Meet priority demands while minimizing congestion"
        )
    
    async def grade(self, agent_callable, num_episodes: int = 3) -> float:
        """Grade agent on medium task"""
        scores = []
        
        for episode in range(num_episodes):
            env = ElectricityTransmissionEnv(seed=100 + episode)
            reset_resp, _ = env.reset()
            
            priority_met_scores = []
            congestion_scores = []
            
            for step in range(config.EPISODE_LENGTH):
                current_obs = reset_resp.observation if step == 0 else env.state().observation
                obs_dict = current_obs.dict()
                
                action_dict = agent_callable(obs_dict)
                action = ActionModel(generator_actions=action_dict)
                
                step_resp = env.step(action)
                obs = step_resp.observation
                
                # Priority score: critical + high priority met %
                critical_met = obs.critical_demand_met / 100.0
                high_met = obs.high_demand_met / 100.0
                priority_score = (critical_met + high_met) / 2.0
                priority_met_scores.append(priority_score)
                
                # Congestion score: % of steps with low congestion
                congestion_info = step_resp.info.get("congestion_score", 0.0)
                congestion_ok = 1.0 if congestion_info < 1.0 else 0.0
                congestion_scores.append(congestion_ok)
            
            # Combined score
            priority_component = np.mean(priority_met_scores)
            congestion_component = np.mean(congestion_scores)
            episode_score = 0.6 * priority_component + 0.4 * congestion_component
            
            scores.append(episode_score)
            self.episode_rewards.append(episode_score)
        
        return float(np.mean(scores))