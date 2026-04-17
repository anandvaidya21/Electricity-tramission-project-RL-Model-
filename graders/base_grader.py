"""Base grader class"""

from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

class BaseGrader(ABC):
    """Abstract base class for task graders"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.episode_rewards: List[float] = []
    
    @abstractmethod
    async def grade(self, agent_callable, num_episodes: int = 3) -> float:
        """
        Run agent and compute score
        Returns: float in [0.0, 1.0]
        """
        pass
    
    def get_score(self) -> float:
        """Return average score across episodes"""
        if not self.episode_rewards:
            return 0.0
        return float(np.mean(self.episode_rewards))