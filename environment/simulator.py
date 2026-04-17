"""Power grid simulation engine"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from environment.config import config

@dataclass
class Consumer:
    """Represents a consumer node"""
    id: int
    priority: int  # 1=critical, 5=lowest
    base_demand: float  # MW
    demand_variance: float  # Std dev
    
    def get_demand(self, time_step: int) -> float:
        """Generate time-varying demand with daily pattern"""
        hour = (time_step * config.SIMULATION_STEP_MINUTES) // 60
        # Peak demand 6-22h, low demand 22-6h
        peak_factor = 1.5 if 6 <= hour <= 22 else 0.5
        noise = np.random.normal(0, self.demand_variance)
        demand = self.base_demand * peak_factor + noise
        return max(0, min(demand, config.CONSUMER_MAX_DEMAND))

@dataclass
class Generator:
    """Represents a power generator"""
    id: int
    max_output: float  # MW
    ramp_rate: float  # MW/step
    cost_per_mwh: float
    fuel_type: str  # 'coal', 'gas', 'renewable', 'nuclear'
    
    current_output: float = 0.0
    
    def set_output(self, target: float) -> float:
        """Set generator output with ramp constraints"""
        max_change = self.ramp_rate * config.SIMULATION_STEP_MINUTES / 60
        new_output = np.clip(target, 
                            self.current_output - max_change,
                            min(self.current_output + max_change, self.max_output))
        self.current_output = new_output
        return new_output

@dataclass
class TransmissionLine:
    """Represents a power line between two nodes"""
    id: int
    from_node: int
    to_node: int
    capacity: float  # MW
    loss_factor: float  # % loss per 100 km
    impedance: float  # Ohms (for stability)
    
    current_flow: float = 0.0
    
    def get_flow_losses(self) -> float:
        """Calculate transmission losses"""
        return abs(self.current_flow) * self.loss_factor / 100.0

@dataclass
class GridState:
    """Complete grid state snapshot"""
    time_step: int
    total_generation: float
    total_demand: float
    total_met: float  # Actual demand met
    priority_met: Dict[int, float]  # % of each priority met
    transmission_losses: float
    line_utilization: Dict[int, float]  # % of capacity used
    grid_frequency: float  # Hz (should be ~50 or 60)
    voltage_stability: float  # 0-1, where 1=stable
    total_cost: float
    
    def to_observation_vector(self) -> np.ndarray:
        """Convert state to RL observation"""
        return np.array([
            self.total_generation,
            self.total_demand,
            self.total_met,
            self.transmission_losses,
            self.grid_frequency,
            self.voltage_stability,
            self.total_cost,
            *[self.priority_met.get(i, 0.0) for i in range(1, 6)]
        ], dtype=np.float32)

class PowerGridSimulator:
    """Main simulation engine"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.consumers: List[Consumer] = []
        self.generators: List[Generator] = []
        self.lines: List[TransmissionLine] = []
        self.time_step = 0
        self.history: List[GridState] = []
        self._initialize_grid()
    
    def _initialize_grid(self):
        """Create default grid topology"""
        # Consumers
        priorities = [config.PRIORITY_CRITICAL, config.PRIORITY_HIGH, 
                     config.PRIORITY_MEDIUM, config.PRIORITY_LOW, config.PRIORITY_LOWEST]
        for i in range(config.NUM_CONSUMERS):
            self.consumers.append(Consumer(
                id=i,
                priority=priorities[i % len(priorities)],
                base_demand=5.0 + np.random.rand() * 10.0,
                demand_variance=1.0
            ))
        
        # Generators (diverse fuel mix)
        fuels = ['coal', 'gas', 'renewable', 'nuclear']
        costs = [30, 50, 10, 25]  # $/MWh
        for i in range(config.NUM_GENERATORS):
            self.generators.append(Generator(
                id=i,
                max_output=config.GENERATOR_MAX_OUTPUT,
                ramp_rate=5.0,  # MW/min
                cost_per_mwh=costs[i % len(costs)],
                fuel_type=fuels[i % len(fuels)]
            ))
        
        # Transmission lines (mesh topology)
        line_id = 0
        for i in range(config.NUM_SUBSTATIONS):
            for j in range(i+1, min(i+3, config.NUM_SUBSTATIONS)):
                self.lines.append(TransmissionLine(
                    id=line_id,
                    from_node=i,
                    to_node=j,
                    capacity=config.TRANSMISSION_LINE_CAPACITY,
                    loss_factor=2.0,
                    impedance=0.1
                ))
                line_id += 1
    
    def step(self, actions: Dict[int, float]) -> Tuple[GridState, float]:
        """
        Execute one simulation step
        actions: {generator_id: target_MW}
        Returns: (new_state, reward)
        """
        # Set generator outputs
        total_gen = 0.0
        gen_costs = 0.0
        for gen_id, target in actions.items():
            if gen_id < len(self.generators):
                output = self.generators[gen_id].set_output(target)
                total_gen += output
                gen_costs += output * self.generators[gen_id].cost_per_mwh
        
        # Get consumer demands
        demands_by_priority = {i: 0.0 for i in range(1, 6)}
        total_demand = 0.0
        for consumer in self.consumers:
            demand = consumer.get_demand(self.time_step)
            total_demand += demand
            demands_by_priority[consumer.priority] += demand
        
        # Allocate power by priority
        available = total_gen
        priority_met = {}
        total_met = 0.0
        
        for priority in range(1, 6):  # 1=highest
            demand = demands_by_priority[priority]
            if available >= demand:
                met = demand
                available -= demand
            else:
                met = available
                available = 0.0
            
            priority_met[priority] = met / max(demand, 0.001)  # % met
            total_met += met
        
        # Calculate losses and grid stability
        transmission_losses = sum(line.get_flow_losses() for line in self.lines)
        unmet_demand = total_demand - total_met
        
        # Simulate frequency (Hz)
        balance = total_gen - total_demand
        grid_frequency = 50.0 + (balance / max(total_demand, 1.0)) * 2.0  # Simplified
        
        # Voltage stability (0-1)
        voltage_stability = max(0.0, 1.0 - abs(balance) / max(total_demand, 1.0) * 0.5)
        
        # Check line congestion
        congestion_penalty = 0.0
        line_utilization = {}
        for line in self.lines:
            utilization = min(1.0, abs(line.current_flow) / line.capacity)
            line_utilization[line.id] = utilization
            if utilization > 0.95:
                congestion_penalty += (utilization - 0.95) * config.PENALTY_CONGESTION
        
        # Compute reward
        reward = 0.0
        
        # Penalty for unmet demand
        reward += unmet_demand * config.PENALTY_UNMET_DEMAND / max(total_demand, 1.0)
        
        # Reward for meeting priorities
        for priority in range(1, 6):
            if priority_met[priority] > 0.95:
                reward += config.REWARD_MET_PRIORITY * (6 - priority) / 5.0
        
        # Penalty for instability
        if abs(grid_frequency - 50.0) > 2.0:
            reward += config.PENALTY_INSTABILITY
        
        # Efficiency bonus
        if 0.9 < total_met / max(total_demand, 0.001) < 1.0:
            reward += config.REWARD_EFFICIENCY
        
        # Add congestion penalty
        reward += congestion_penalty
        
        state = GridState(
            time_step=self.time_step,
            total_generation=total_gen,
            total_demand=total_demand,
            total_met=total_met,
            priority_met=priority_met,
            transmission_losses=transmission_losses,
            line_utilization=line_utilization,
            grid_frequency=grid_frequency,
            voltage_stability=voltage_stability,
            total_cost=gen_costs
        )
        
        self.history.append(state)
        self.time_step += 1
        
        return state, reward
    
    def reset(self):
        """Reset to initial state"""
        self.time_step = 0
        self.history = []
        for gen in self.generators:
            gen.current_output = 0.0
        for line in self.lines:
            line.current_flow = 0.0
        return self.get_state()
    
    def get_state(self) -> GridState:
        """Get current grid state"""
        if not self.history:
            return GridState(
                time_step=0,
                total_generation=0.0,
                total_demand=0.0,
                total_met=0.0,
                priority_met={i: 0.0 for i in range(1, 6)},
                transmission_losses=0.0,
                line_utilization={},
                grid_frequency=50.0,
                voltage_stability=1.0,
                total_cost=0.0
            )
        return self.history[-1]