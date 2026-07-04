"""静态设施模型层(仅标准库).

对外主入口: InfrastructureNetwork.load()  ——见 network.py
"""

from derbysim.infra.entities import Platform, Route, Signal, TrackCircuit
from derbysim.infra.network import InfrastructureNetwork

__all__ = ["Signal", "TrackCircuit", "Route", "Platform", "InfrastructureNetwork"]
