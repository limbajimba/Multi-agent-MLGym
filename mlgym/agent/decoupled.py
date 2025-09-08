"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Decoupled agent for the MLGym framework.
This agent extends BaseAgent to support decoupled action proposal without environment execution.
"""

from mlgym.agent.base import BaseAgent, AgentArguments


class DecoupledAgent(BaseAgent):
    """
    Decoupled agent that proposes actions without executing them.
    
    This agent extends BaseAgent to support the decoupled workflow where:
    1. Agent proposes actions using BaseAgent.forward()
    2. Run loop decides whether to execute the action
    3. Environment executes approved actions
    
    The key difference from BaseAgent is that this agent doesn't execute
    actions in its _run_step() method - the run loop handles execution.
    """
    
    def __init__(self, name: str, args: AgentArguments):
        super().__init__(name, args)
    
    def _run_step(self, observation: str | None) -> tuple[str | None, bool]:
        """
        Propose an action without executing it.
        
        This overrides BaseAgent._run_step() to only propose actions,
        not execute them. The run loop is responsible for execution.
        
        Args:
            observation (str | None): Current observation

        Returns:
            tuple[str | None, bool]: Tuple containing:
                - observation: None (no execution)
                - done: False (not done, just proposed)
        """
        # Get environment state
        assert self._env is not None
        state = self._env.communicate(self.tools.state_command.name) if self.tools.state_command else ""
        
        # Generate action proposal using BaseAgent.forward()
        thought, action, output = self.forward(observation, self._env.get_available_actions(), state)
        
        # Store in trajectory (BaseAgent already handles this in forward())
        # But we don't execute the action - that's the run loop's job
        
        # Return None observation since we didn't execute anything
        return None, False 