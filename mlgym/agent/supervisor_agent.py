#!/usr/bin/env python3
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Supervisor Agent for MLGym

A research supervisor agent that manages multi-agent workflows by planning, organizing,
and coordinating specialized agents to complete research tasks.
"""

from mlgym.agent.base import BaseAgent
from mlgym.utils.log import get_logger


class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent that manages multi-agent workflows.
    
    This agent:
    1. Plans research workflows and strategies
    2. Creates and manages specialized agents
    3. Handles workflow completion and summarization
    """
    
    def __init__(self, name: str, args):
        """Initialize supervisor agent."""
        super().__init__(name, args)

    def forward(self, observation: str | None, available_actions: list[str], state: str) -> tuple[str, str, str]:
        """
        Forward pass for supervisor agent.
        
        Args:
            observation: Current workflow state and context
            available_actions: Available supervisor actions
            state: Environment state
            
        Returns:
            thought: Agent's reasoning
            action: Action to take (with structured data)
            output: Additional output
        """
        # Use BaseAgent's forward method which handles format errors properly
        thought, action, output = super().forward(observation, available_actions, state)

        self.logger.info(f"Supervisor decision: {action}")

        return thought, action, output 