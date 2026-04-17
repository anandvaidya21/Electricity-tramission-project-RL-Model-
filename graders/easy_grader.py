"""Easy task: Meet 80% of demand under stable conditions"""

from graders.base_grader import BaseGrader
from environment.env import ElectricityTransmissionEnv, ActionModel
from environment.config import config
import numpy as np

class EasyGrader(BaseGrader):
    """
    Easy Task: Stable grid with low variability
    Goal: Meet >=80% of total demand
    Expected agent: Random baseline
    """
    
    def __init__(self):
        super().__init__(
            name="Easy: Baseline Demand Met",
            description="Meet 80% of total demand under stable conditions"
        )
    
    async def grade(self, agent_callable, num_episodes: int = 3) -> float:
        """Grade agent on easy task"""
        scores = []
        
        for episode in range(num_episodes):
            env = ElectricityTransmissionEnv(seed=42 + episode)
            reset_resp, _ = env.reset()
            
            episode_rewards = []
            episode_demand_met_pcts = []
            
            for step in range(config.EPISODE_LENGTH):
                # Agent decides actions
                current_obs = reset_resp.observation if step == 0 else env.state().observation
                obs_dict = current_obs.dict()
                
                # Call agent (simple rule-based baseline)
                action_dict = agent_callable(obs_dict)
                action = ActionModel(generator_actions=action_dict)
                
                # Step environment
                step_resp = env.step(action)
                episode_rewards.append(step_resp.reward)
                
                demand_met_pct = (step_resp.observation.total_met / 
                                 max(step_resp.observation.total_demand, 0.001))
                episode_demand_met_pcts.append(demand_met_pct)
            
            # Score: proportion of steps meeting 80% threshold
            steps_meeting_threshold = sum(1 for pct in episode_demand_met_pcts if pct >= 0.8)
            episode_score = steps_meeting_threshold / len(episode_demand_met_pcts)
            scores.append(episode_score)
            self.episode_rewards.append(episode_score)
        
        return float(np.mean(scores))