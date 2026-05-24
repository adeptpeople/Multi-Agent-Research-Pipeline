from __future__ import annotations

from ..agents.base_agent import BaseAgent


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, name: str, agent: BaseAgent) -> None:
        self._agents[name] = agent

    def get(self, name: str) -> BaseAgent:
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' not found in registry. Registered: {list(self._agents)}")
        return self._agents[name]

    def names(self) -> list[str]:
        return list(self._agents.keys())
