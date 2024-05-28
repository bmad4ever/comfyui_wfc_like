from dataclasses import dataclass


@dataclass
class TemperatureConfig:
    starting_temperature: float
    min_min_temperature: float
    max_min_temperature: float


@dataclass
class SearchWeights:
    reverse_depth_w: float
    node_cost_w: float
    prev_state_avg_entropy_w: float  # subject to change
