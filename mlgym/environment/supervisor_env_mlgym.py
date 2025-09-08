#!/usr/bin/env python3
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

MLGym-Compliant Supervisor Environment

A multi-agent environment that follows proper MLGym patterns:
- Environment only handles execution, not decision-making
- Uses proper tool system for command parsing
- Clean separation of concerns
- Proper step counting and state management
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json
import time
import re
from pathlib import Path

from mlgym.agent.base import BaseAgent, AgentArguments, AgentConfig
from mlgym.environment.env import MLGymEnv
from mlgym.utils.log import get_logger
from mlgym.constants import DEFAULT_MAX_STEPS
from mlgym.tools.tools import ToolHandler, ToolsConfig, Command


@dataclass
class AgentPlan:
    """Plan for a single agent in the workflow."""
    name: str
    role: str
    instructions: str = ""
    max_steps: int = DEFAULT_MAX_STEPS


class SupervisorEnvMLGym(MLGymEnv):
    """
    MLGym-Compliant Multi-agent environment managed by a supervisor agent.
    
    This environment follows proper MLGym patterns:
    1. Environment only handles execution, not decision-making
    2. Uses proper tool system for command parsing
    3. Clean separation of concerns
    4. Proper step counting and state management
    """
    
    def __init__(self, args, devices, render_mode=None, supervisor_args=None):
        super().__init__(args, devices, render_mode)
        self.logger = get_logger("supervisor_env_mlgym")
        
        # Validate and store supervisor arguments
        if supervisor_args is None:
            self.logger.warning("No supervisor args provided, using defaults")
            supervisor_args = self._create_default_supervisor_args()
        
        self.supervisor_args = supervisor_args
        self._validate_supervisor_args()
        
        # Unified state tracking following MLGym patterns
        self.workflow_plan = {}
        self.agent_plans = []
        self.current_agent = None
        
        # Pre-allocated agent tracking to prevent race conditions
        self.agent_tracking: dict[str, dict] = {}  # agent_id -> {name, role, steps, status}
        self.next_agent_id = 1
        
        # Supervisor step tracking
        self.supervisor_step: int = 0
        
        # Agent completion tracking
        self.completed_agents: list[dict] = []
        
        # Default agent configuration for worker agents
        self.default_agent_args = None
        self._initialized = False
        
        # Store run name for trajectory directories
        self.run_name = None
        
        # Supervisor cost tracking
        self.supervisor_cost = 0.0
        self.supervisor_tokens_sent = 0
        self.supervisor_tokens_received = 0
        self.supervisor_api_calls = 0
        
        # Load supervisor tools (MLGym pattern)
        self.supervisor_tools = self._load_supervisor_tools()
    
    def _create_default_supervisor_args(self):
        """Create default supervisor arguments if none provided."""
        from dataclasses import dataclass
        
        @dataclass
        class DefaultSupervisorArgs:
            max_supervisor_steps: int = 50
            max_agents_per_workflow: int = 10
            max_steps_per_agent: int = 100
        
        return DefaultSupervisorArgs()
    
    def _validate_supervisor_args(self):
        """Validate supervisor arguments have required attributes."""
        required_attrs = ['max_supervisor_steps', 'max_agents_per_workflow', 'max_steps_per_agent']
        for attr in required_attrs:
            if not hasattr(self.supervisor_args, attr):
                raise ValueError(f"Supervisor args missing required attribute: {attr}")
        
        # Validate reasonable ranges
        if self.supervisor_args.max_supervisor_steps <= 0:
            raise ValueError("max_supervisor_steps must be positive")
        if self.supervisor_args.max_agents_per_workflow <= 0:
            raise ValueError("max_agents_per_workflow must be positive")
    
    def _load_supervisor_tools(self) -> ToolHandler:
        """Load supervisor-specific tools following MLGym pattern."""
        # Create actual bash commands that the supervisor can use
        # These are real bash functions that the supervisor can call
        plan_workflow_cmd = Command(
            name="plan_workflow",
            code="""
plan_workflow() {
    # This is a supervisor command that plans the research workflow
    # The actual implementation is in the supervisor agent's prompt
    echo "Planning workflow..."
    # The supervisor LLM will handle the actual planning logic
    # This command just signals the environment to execute workflow planning
};
""",
            docstring="Plan the research workflow strategy and agent sequence"
        )
        
        create_agent_cmd = Command(
            name="create_agent",
            code="""
create_agent() {
    # This is a supervisor command that creates and runs an agent
    # The actual implementation is in the supervisor agent's prompt
    local agent_name="$1"
    local instructions="$2"
    
    if [ -z "$agent_name" ] || [ -z "$instructions" ]; then
        echo "Usage: create_agent <agent_name> '<instructions>'"
        return 1
    fi
    
    echo "Creating agent: $agent_name with instructions: $instructions"
    # The supervisor LLM will handle the actual agent creation logic
    # This command just signals the environment to execute agent creation
};
""",
            docstring="Create and run the next agent in the workflow"
        )
        
        complete_workflow_cmd = Command(
            name="complete_workflow",
            code="""
complete_workflow() {
    # This is a supervisor command that completes the workflow
    # The actual implementation is in the supervisor agent's prompt
    echo "Completing workflow..."
    # The supervisor LLM will handle the actual completion logic
    # This command just signals the environment to execute workflow completion
};
""",
            docstring="Complete the workflow when all agents are done"
        )
        
        supervisor_config = ToolsConfig(
            commands=[
                plan_workflow_cmd,
                create_agent_cmd,
                complete_workflow_cmd,
            ]
        )
        return ToolHandler(supervisor_config)
    
    def _create_supervisor_observation(self) -> str:
        """Create observation for supervisor following MLGym pattern."""
        # Create workflow state information for supervisor
        workflow_state = {
            "supervisor_step": self.supervisor_step,
            "workflow_planned": bool(self.workflow_plan),
            "agents_planned": len(self.agent_plans),
            "agents_completed": len(self.completed_agents),
            "workflow_strategy": self.workflow_plan.get('strategy', 'Not set'),
            "success_criteria": self.workflow_plan.get('success_criteria', 'Not set'),
            "agent_plans": [
                {
                    "name": plan.name,
                    "role": plan.role,
                    "status": "✅ Completed" if i < len(self.completed_agents) else "⏳ Pending"
                }
                for i, plan in enumerate(self.agent_plans)
            ],
            "completed_agents": [
                {
                    "name": agent_result.get("agent_name", "unknown"),
                    "steps": agent_result.get("steps_taken", 0),
                    "status": agent_result.get("exit_status", "unknown")
                }
                for agent_result in self.completed_agents
            ]
        }
        
        return json.dumps(workflow_state, indent=2)
    
    def _cleanup_old_agent_data(self, max_agents_to_keep: int = 100):
        """Clean up old agent data to prevent memory leaks."""
        if len(self.agent_tracking) > max_agents_to_keep:
            # Keep only the most recent agents
            sorted_agents = sorted(self.agent_tracking.items(), key=lambda x: x[1].get("steps", 0))
            agents_to_remove = sorted_agents[:-max_agents_to_keep]
            
            for agent_id, _ in agents_to_remove:
                self.agent_tracking.pop(agent_id, None)
            
            self.logger.info(f"Cleaned up {len(agents_to_remove)} old agent records")
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """Reset the environment following MLGym pattern - always call parent for proper container management."""
        # ALWAYS call parent reset for proper container setup (MLGym pattern)
        observation, info = super().reset(seed=seed, options=options)
        
        # Reset supervisor state (container stays the same)
        self.workflow_plan = {}
        self.agent_plans = []
        self.current_agent = None
        self.agent_tracking = {}
        self.next_agent_id = 1
        self.supervisor_step = 0
        self.completed_agents = []
        
        # Create initial observation for supervisor
        observation = self._create_supervisor_observation()
        
        return observation, info
    
    def reset_for_new_agent(self) -> None:
        """Reset supervisor state for a new agent while keeping the container persistent."""
        # Reset supervisor state but keep container persistent
        self.current_agent = None
        # Don't reset agent_tracking, completed_agents, or workflow_plan
        # This allows sequential agents to see each other's work
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute supervisor action using proper MLGym patterns."""
        try:
            self.logger.info(f"Supervisor action: {action}")
            
            # Check supervisor step limit (MLGym pattern)
            if self.supervisor_step >= self.supervisor_args.max_supervisor_steps:
                self.logger.warning(f"Max supervisor steps reached: {self.supervisor_args.max_supervisor_steps}")
                return "Max supervisor steps reached", 0, True, {"exit_status": "max_supervisor_steps"}
            
            # Use the MLGym tool system to check if this is a supervisor command
            if self._is_supervisor_command(action):
                # Handle supervisor command using tool system
                result = self._handle_supervisor_command(action)
                # Increment supervisor step for supervisor commands
                self.supervisor_step += 1
                
                # Cleanup old agent data to prevent memory leaks
                self._cleanup_old_agent_data()
                
                return result
            else:
                # For all other commands (including worker agent commands), 
                # pass through to parent MLGymEnv.step method
                result = super().step(action)
                # Increment supervisor step for non-supervisor commands
                self.supervisor_step += 1
                return result
                
        except Exception as e:
            self.logger.error(f"Error in supervisor step: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error: {e}", 0, False, {"error": str(e), "dont_count_step": True}
    
    def _is_supervisor_command(self, action: str) -> bool:
        """Check if action is a supervisor command using proper parsing."""
        # Supervisor commands are workflow actions, not bash commands
        # They follow a specific pattern that we can detect
        action_lower = action.strip().lower()
        
        # Check for supervisor command patterns
        if (action_lower.startswith("plan_workflow") or 
            action_lower.startswith("create_agent") or 
            action_lower.startswith("complete_workflow")):
            return True
        
        return False
    
    def _handle_supervisor_command(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Handle supervisor commands using proper parsing."""
        
        # Parse the action to determine which command this is
        action_lower = action.strip().lower()
        
        if action_lower.startswith("plan_workflow"):
            return self._execute_workflow_planning(action)
        elif action_lower.startswith("create_agent"):
            return self._execute_agent_creation(action)
        elif action_lower.startswith("complete_workflow"):
            return self._complete_workflow()
        else:
            # Invalid command - provide error feedback
            error_msg = f"Invalid supervisor command: '{action}'. Available commands: plan_workflow, create_agent, complete_workflow"
            return error_msg, 0, False, {"status": "invalid_command", "dont_count_step": True}
    
    
    def _execute_workflow_planning(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute workflow planning based on supervisor's action."""
        # Parse workflow plan from supervisor's action
        workflow_plan = self._parse_workflow_plan_from_action(action)
        
        return self._plan_workflow(workflow_plan)
    
    def _parse_workflow_plan_from_action(self, action: str) -> Dict[str, Any]:
        """Parse workflow plan from supervisor's action using robust parsing."""
        # Default workflow plan
        workflow_plan = {
            "strategy": "Complete the research task with multiple specialized agents",
            "agent_sequence": [],
            "success_criteria": "Achieve better performance than baseline"
        }
        
        try:
            # Try to extract structured information using multiple approaches
            
            # Approach 1: Look for structured format like [strategy: "..."][agents: "..."][criteria: "..."]
            
            # Extract strategy
            strategy_match = re.search(r'strategy:\s*["\']([^"\']+)["\']', action, re.IGNORECASE)
            if strategy_match:
                workflow_plan["strategy"] = strategy_match.group(1).strip()
            
            # Extract agents
            agents_match = re.search(r'agents:\s*["\']([^"\']+)["\']', action, re.IGNORECASE)
            if agents_match:
                agents_text = agents_match.group(1).strip()
                # Parse comma-separated agent names
                agent_sequence = [agent.strip().strip('"\'') for agent in agents_text.split(",") if agent.strip()]
                workflow_plan["agent_sequence"] = agent_sequence
            
            # Extract criteria
            criteria_match = re.search(r'criteria:\s*["\']([^"\']+)["\']', action, re.IGNORECASE)
            if criteria_match:
                workflow_plan["success_criteria"] = criteria_match.group(1).strip()
            
            # Approach 2: Fallback to keyword-based extraction
            if not workflow_plan["strategy"] or workflow_plan["strategy"] == workflow_plan["strategy"]:
                # Look for strategy in the text
                strategy_keywords = ["strategy", "approach", "method", "plan"]
                for keyword in strategy_keywords:
                    if keyword in action.lower():
                        # Extract sentence containing the keyword
                        sentences = action.split('.')
                        for sentence in sentences:
                            if keyword in sentence.lower():
                                strategy = sentence.strip()
                                if len(strategy) > 10:  # Basic validation
                                    workflow_plan["strategy"] = strategy
                                    break
                        if workflow_plan["strategy"] != workflow_plan["strategy"]:
                            break
            
            # Approach 3: Extract agent roles from context
            if not workflow_plan["agent_sequence"]:
                # Look for common agent role patterns
                agent_patterns = [
                    r'(\w+Agent)',  # StrategyAgent, ResearchAgent, etc.
                    r'(\w+_agent)',  # strategy_agent, research_agent, etc.
                    r'(\w+Agent\d+)',  # StrategyAgent1, ResearchAgent2, etc.
                ]
                
                found_agents = set()
                for pattern in agent_patterns:
                    matches = re.findall(pattern, action, re.IGNORECASE)
                    found_agents.update(matches)
                
                if found_agents:
                    workflow_plan["agent_sequence"] = list(found_agents)
            
            # Validate extracted data
            if not workflow_plan["agent_sequence"]:
                self.logger.warning("No agents found in action, using default")
                workflow_plan["agent_sequence"] = ["StrategyAgent", "ResearchAgent", "ValidationAgent"]
            
            # Limit number of agents to prevent memory issues
            max_agents = getattr(self.supervisor_args, 'max_agents_per_workflow', 10)
            if len(workflow_plan["agent_sequence"]) > max_agents:
                self.logger.warning(f"Limiting agent sequence from {len(workflow_plan['agent_sequence'])} to {max_agents}")
                workflow_plan["agent_sequence"] = workflow_plan["agent_sequence"][:max_agents]
            
            self.logger.info(f"Parsed workflow plan: {workflow_plan}")
            
        except Exception as e:
            self.logger.error(f"Error parsing workflow plan: {e}")
            # Return default plan if parsing fails
            pass
        
        return workflow_plan
    
    def _plan_workflow(self, data: Dict[str, Any]) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Plan the research workflow with supervisor's strategy."""
        # Handle both direct workflow plan and wrapped workflow plan
        if "workflow_plan" in data:
            workflow_plan = data.get("workflow_plan", {})
        else:
            workflow_plan = data  # Direct workflow plan
        
        # If workflow is already planned, remind supervisor
        if self.workflow_plan:
            error_msg = f"Workflow already planned. Strategy: {self.workflow_plan.get('strategy', 'Not set')}. You can proceed to create agents."
            return error_msg, 0, False, {"status": "workflow_already_planned", "dont_count_step": True}
        
        # Store workflow plan
        self.workflow_plan = workflow_plan
        
        # Create agent plans based on supervisor's strategy
        agent_sequence = workflow_plan.get("agent_sequence", [])
        self.agent_plans = []
        
        for i, role in enumerate(agent_sequence):
            agent_plan = AgentPlan(
                name=f"agent_{i+1}",
                role=role,
                instructions=f"Focus on {role} responsibilities for this task",
                max_steps=20
            )
            self.agent_plans.append(agent_plan)
        
        self.logger.info(f"📋 Workflow planned: {workflow_plan.get('strategy', 'Unknown')} with {len(self.agent_plans)} agents")
        
        success_msg = f"Workflow planned successfully. Strategy: {workflow_plan.get('strategy', 'Unknown')}. {len(self.agent_plans)} agents planned."
        return success_msg, 0, False, {"status": "workflow_planned"}
    
    def _execute_agent_creation(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute agent creation based on supervisor's action."""
        # Parse: "create_agent explorer 'Analyze dataset'"
        parts = action.split(" ", 2)
        if len(parts) >= 3:
            name = parts[1]
            instructions = parts[2].strip("'")
            
            agent_creation = {
                "name": name,
                "role": name,
                "instructions": instructions,
                "max_steps": 20
            }
            
            return self._create_and_run_agent({"agent_creation": agent_creation})
        else:
            error_msg = "Invalid create_agent command format. Use: create_agent <name> '<instructions>'"
            return error_msg, 0, False, {"status": "invalid_format", "dont_count_step": True}
    
    def _create_and_run_agent(self, data: Dict[str, Any]) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Create and run next agent with supervisor-defined parameters."""
        # Check if we've reached the maximum number of agents (MLGym pattern)
        if len(self.completed_agents) >= self.supervisor_args.max_agents_per_workflow:
            self.logger.warning(f"Max agents per workflow reached: {self.supervisor_args.max_agents_per_workflow}")
            return "Max agents per workflow reached", 0, True, {"status": "max_agents_reached"}
        
        # Check if we have planned agents to run
        if len(self.completed_agents) >= len(self.agent_plans):
            return "All planned agents completed", 0, True, {"status": "all_agents_done"}
        
        # Get current agent plan
        agent_plan = self.agent_plans[len(self.completed_agents)]
        
        # Override with supervisor's agent creation data if provided
        agent_creation = data.get("agent_creation", {})
        if agent_creation:
            agent_plan.name = agent_creation.get("name", agent_plan.name)
            agent_plan.role = agent_creation.get("role", agent_plan.role)
            agent_plan.instructions = agent_creation.get("instructions", agent_plan.instructions)
            agent_plan.max_steps = agent_creation.get("max_steps", agent_plan.max_steps)
        
        # Pre-allocate agent tracking to prevent race conditions
        agent_id = f"agent_{self.next_agent_id}"
        agent_name = f"{agent_plan.role}_{len(self.completed_agents) + 1}"
        
        # Create tracking entry BEFORE starting execution
        self.agent_tracking[agent_id] = {
            "name": agent_name,
            "role": agent_plan.role,
            "steps": 0,
            "status": "created",
            "max_steps": agent_plan.max_steps,
            "instructions": agent_plan.instructions
        }
        self.next_agent_id += 1
        
        self.logger.info(f"🤖 Creating agent: {agent_name} (Role: {agent_plan.role}) with ID: {agent_id}")
        
        # Create custom ToolsConfig with supervisor variables in env_variables
        from mlgym.tools.tools import ToolsConfig
        # Properly escape the supervisor instructions for bash environment variables
        import shlex
        escaped_instructions = shlex.quote(agent_plan.instructions)
        escaped_role = shlex.quote(agent_plan.role)
        
        # Create collaborative context for the agent
        previous_agent_context = self._create_previous_agent_context()
        agent_position = f"{len(self.completed_agents) + 1} of {len(self.agent_plans)}"
        next_agent_role = self.agent_plans[len(self.completed_agents) + 1].role if len(self.completed_agents) + 1 < len(self.agent_plans) else "Final Agent"
        workflow_objective = self.workflow_plan.get('strategy', 'Complete the research task')
        success_criteria = self.workflow_plan.get('success_criteria', 'Achieve better performance than baseline')
        
        custom_tools_config = ToolsConfig(
            command_files=self.default_agent_args.config.tools.command_files,
            env_variables={
                **self.default_agent_args.config.tools.env_variables,  # Keep existing vars
                "agent_role": escaped_role,                        # Add supervisor's role (escaped)
                "supervisor_instructions": escaped_instructions,   # Add supervisor's instructions (escaped)
                "previous_agent_context": shlex.quote(previous_agent_context),
                "agent_position": agent_position,
                "next_agent_role": next_agent_role,
                "workflow_objective": shlex.quote(workflow_objective),
                "success_criteria": shlex.quote(success_criteria),
            },
            util_functions=self.default_agent_args.config.tools.util_functions,
            submit_command=self.default_agent_args.config.tools.submit_command,
            parser=self.default_agent_args.config.tools.parser,
            state_command=self.default_agent_args.config.tools.state_command,
            blocklist_error_template=self.default_agent_args.config.tools.blocklist_error_template,
            blocklist=self.default_agent_args.config.tools.blocklist,
            blocklist_standalone=self.default_agent_args.config.tools.blocklist_standalone,
            commands=self.default_agent_args.config.tools.commands,
        )
        
        # Load the enhanced template from YAML file instead of using original templates
        enhanced_config = AgentConfig.load_yaml(self.default_agent_args.agent_config_path)
        
        # Create a new agent config with enhanced templates and custom tools
        dynamic_config = AgentConfig(
            system_template=enhanced_config.system_template,  # Use enhanced template
            task_template=enhanced_config.task_template,      # Use enhanced template
            next_step_template=enhanced_config.next_step_template,
            next_step_no_output_template=enhanced_config.next_step_no_output_template,
            tools=custom_tools_config,  # Use custom tools with supervisor variables
            output_parser=enhanced_config.output_parser,
            history_processor="Last5Observations",
            history_processor_args=enhanced_config.history_processor_args
        )
        
        # Create agent with dynamic config
        dynamic_args = AgentArguments(
            model=self.default_agent_args.model,
            agent_config_path=self.default_agent_args.agent_config_path,
            config=dynamic_config,
            log_verbose_to_console=self.default_agent_args.log_verbose_to_console
        )
        
        # Use BaseAgent instead of DecoupledAgent for simplicity
        self.current_agent = BaseAgent(agent_plan.name, dynamic_args)
        
        # Setup agent following MLGym pattern - let the agent handle its own setup
        try:
            self.current_agent.setup(self.task.args)
            # Set environment reference for trajectory saving (MLGym pattern)
            self.current_agent._env = self
        except Exception as e:
            self.logger.error(f"Failed to setup agent {agent_name}: {e}")
            # Update tracking status
            self.agent_tracking[agent_id]["status"] = "setup_failed"
            self.agent_tracking[agent_id]["error"] = str(e)
            raise
        
        # Create agent-specific trajectory directory with dynamic naming
        agent_traj_dir = self._create_safe_trajectory_path(agent_name)
        
        # Let the agent handle its own trajectory directory setup
        self.current_agent.traj_dir = agent_traj_dir
        
        # Initialize environment variables and shell functions (MLGym pattern)
        self.current_agent.init_environment_vars(self)
        
        # Verify shell functions are properly set up
        try:
            result = self.communicate("type edit")
            self.logger.info(f"Edit command setup check: {result}")
        except Exception as e:
            self.logger.warning(f"Edit command setup warning: {e}")
        
        # Run agent with plan parameters
        try:
            agent_result = self._run_agent_with_plan(agent_plan, agent_id)
            
            # Store result and track completion (MLGym pattern)
            # Note: agent_name is already stored in agent_results above
            self.agent_tracking[agent_id]["status"] = "completed"
            
            # Reset for next agent while keeping container persistent
            self.reset_for_new_agent()
            
            # Return success message
            success_msg = f"Agent {agent_name} completed successfully. {self.agent_tracking[agent_id]['steps']} steps taken."
            return success_msg, 0, False, {"status": "agent_completed", "agent_name": agent_name, "agent_id": agent_id}
            
        except Exception as e:
            self.logger.error(f"Error running agent {agent_name}: {e}")
            # Update tracking status
            self.agent_tracking[agent_id]["status"] = "error"
            self.agent_tracking[agent_id]["error"] = str(e)
            # Still track completion even if there was an error
            # Note: agent_name is already stored in agent_results above
            
            # Return error message
            error_msg = f"Agent {agent_name} failed with error: {str(e)}"
            return error_msg, 0, False, {"status": "agent_error", "error": str(e), "agent_name": agent_name, "agent_id": agent_id}
    
    def _run_agent_with_plan(self, agent_plan: AgentPlan, agent_id: str) -> Dict[str, Any]:
        """Run current agent with proper MLGym trajectory management."""
        import time
        from mlgym.types import TrajectoryStep
        
        max_steps = self.supervisor_args.max_steps_per_agent
        agent_name = self.agent_tracking[agent_id]["name"]
        
        self.logger.info(f"🔄 Running agent {self.current_agent.name} with role: {agent_plan.role}")
        
        # Initialize MLGym-style agent execution
        observation = None
        done = False
        info = {"exit_status": "not_started"}
        
        # Run agent following proper MLGym pattern
        while not done and self.agent_tracking[agent_id]["steps"] < max_steps:
            try:
                # Start timing for execution
                execution_t0 = time.perf_counter()
                
                # Create proper state (MLGym pattern)
                current_step = self.agent_tracking[agent_id]["steps"]
                remaining_steps = max(0, max_steps - current_step)
                
                # Get current open file from container state
                try:
                    # Query the container for current file state
                    file_state_output = self.communicate("echo $CURRENT_FILE", timeout_duration=5)
                    open_file = file_state_output.strip() if file_state_output.strip() else "n/a"
                except Exception:
                    open_file = "n/a"
                
                # Create warning message based on remaining steps
                if remaining_steps <= 3:
                    warning_message = "CRITICAL: Only 3 or fewer steps left! SUBMIT IMMEDIATELY if you have a working solution!"
                elif remaining_steps <= 5:
                    warning_message = "WARNING: Only 5 or fewer steps left! Consider submitting your current solution soon."
                else:
                    warning_message = ""
                
                # Create state data (moved outside try-except block)
                state_data = {
                    "open_file": open_file,
                    "working_dir": "/home/agent/workspace",
                    "current_step": current_step,
                    "remaining_steps": remaining_steps,
                    "warning_message": warning_message
                }
                state = json.dumps(state_data)
                
                # Get available actions from agent's tools (MLGym pattern)
                available_actions = [cmd.name for cmd in self.current_agent.tools.commands] if hasattr(self.current_agent, 'tools') and self.current_agent.tools else []
                
                # Agent proposes action
                thought, action, output = self.current_agent.forward(observation, available_actions, state)
                
                # CRITICAL: Process action through the agent's tool handler to handle multi-line commands
                run_action = self.current_agent.tools.guard_multiline_input(action).strip()
                
                # Execute action in container
                observation, reward, done, info = super().step(run_action)
                
                # Update file state if open command was executed
                if action.strip().startswith('open '):
                    try:
                        # Extract file path from open command
                        import shlex
                        parts = shlex.split(action.strip())
                        if len(parts) >= 2:
                            file_path = parts[1]
                            # Update CURRENT_FILE in container
                            self.communicate(f'export CURRENT_FILE="{file_path}"', timeout_duration=5)
                    except Exception as e:
                        self.logger.warning(f"Failed to update file state after open command: {e}")
                
                # Calculate execution time
                execution_time = time.perf_counter() - execution_t0
                
                # Create trajectory step (MLGym pattern)
                trajectory_step = TrajectoryStep(
                    state=state,
                    response=output,
                    thought=thought,
                    action=action,
                    execution_time=execution_time,
                    observation=observation,
                )
                
                # Add to agent's trajectory
                self.current_agent.trajectory.append(trajectory_step)
                
                # Update model statistics (CRITICAL: Capture stats after each forward call)
                from mlgym.backend.base import APIStats
                model_stats: APIStats = self.current_agent.model.stats
                self.current_agent.info["model_stats"] = model_stats.to_dict()
                
                # Log the stats for debugging
                self.logger.debug(f"Agent {agent_name} stats after step {self.agent_tracking[agent_id]['steps']}: "
                                f"cost={model_stats.task_cost:.4f}, tokens_sent={model_stats.tokens_sent}, "
                                f"tokens_received={model_stats.tokens_received}, api_calls={model_stats.api_calls}")
                
                # CRITICAL: Update agent info with environment info (for validation scores)
                if info:
                    self.current_agent.info.update(info)
                
                # Save trajectory after each step (MLGym pattern)
                self.current_agent.save_trajectory()
                
                # Update step counter
                self.agent_tracking[agent_id]["steps"] += 1
                
                # Check completion conditions
                if (done or 
                    action.strip() == "submit" or 
                    self.agent_tracking[agent_id]["steps"] >= max_steps):
                    self.logger.info(f"✅ Agent {agent_name} completed: {info.get('exit_status', 'completed')}")
                    break
                    
            except Exception as e:
                import traceback
                self.logger.error(f"Error in agent execution: {e}")
                self.logger.error(f"Full traceback: {traceback.format_exc()}")
                # Set default values for error case
                observation = f"Error: {e}"
                reward = 0
                done = True
                info = {"error": str(e), "exit_status": "error"}
                break
        
        # Save final results - let MLGym handle score capture from validate command
        try:
            # CRITICAL: Ensure model stats are captured before saving results
            if hasattr(self.current_agent, 'model') and hasattr(self.current_agent.model, 'stats'):
                from mlgym.backend.base import APIStats
                model_stats: APIStats = self.current_agent.model.stats
                self.current_agent.info["model_stats"] = model_stats.to_dict()
                self.logger.info(f"Final model stats for {agent_name}: {model_stats.to_dict()}")
            
            # Only add completion metrics if no validation scores exist
            if not self.current_agent.info.get("score"):
                # If no validation scores, capture completion metrics as fallback
                self.current_agent.info["score"] = [{
                    "completion_rate": 1.0 if info.get("exit_status") == "completed" else 0.0,
                    "steps_taken": self.agent_tracking[agent_id]["steps"],
                    "max_steps": max_steps,
                    "efficiency": self.agent_tracking[agent_id]["steps"] / max_steps if max_steps > 0 else 0.0
                }]
            self.current_agent.save_results()
        except Exception as e:
            self.logger.warning(f"Failed to save agent results: {e}")
        
        # Extract cost information from agent's model stats
        agent_cost = 0.0
        agent_tokens_sent = 0
        agent_tokens_received = 0
        agent_api_calls = 0
        
        if hasattr(self.current_agent, 'model') and hasattr(self.current_agent.model, 'stats'):
            from mlgym.backend.base import APIStats
            model_stats: APIStats = self.current_agent.model.stats
            agent_cost = model_stats.task_cost
            agent_tokens_sent = model_stats.tokens_sent
            agent_tokens_received = model_stats.tokens_received
            agent_api_calls = model_stats.api_calls
        
        # Store agent results for supervisor aggregation
        agent_results = {
            "agent_name": agent_name,
            "agent_id": agent_id,
            "role": agent_plan.role,
            "steps_taken": self.agent_tracking[agent_id]["steps"],
            "exit_status": info.get("exit_status", "completed"),
            "scores": self.current_agent.info.get("score", []),
            "cost": agent_cost,
            "tokens_sent": agent_tokens_sent,
            "tokens_received": agent_tokens_received,
            "api_calls": agent_api_calls
        }
        self.completed_agents.append(agent_results)
        
        # Return agent result
        return {
            "agent_name": agent_name,
            "agent_id": agent_id,
            "role": agent_plan.role,
            "instructions": agent_plan.instructions,
            "steps_taken": self.agent_tracking[agent_id]["steps"],
            "max_steps_planned": max_steps,
            "exit_status": info.get("exit_status", "completed"),
            "final_observation": observation if 'observation' in locals() else "No observation"
        }
    
    
    def _complete_workflow(self) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Complete the workflow."""
        self.logger.info("🎉 Workflow completed successfully")
        
        # Save supervisor results with aggregated agent performance
        try:
            self.save_results()
        except Exception as e:
            self.logger.warning(f"Failed to save supervisor results: {e}")
        
        # Create final summary (MLGym pattern)
        summary = {
            "status": "completed",
            "total_agents": len(self.completed_agents),
            "agents_completed": self.completed_agents,
            "workflow_plan": self.workflow_plan
        }
        
        return "Workflow completed successfully", 1.0, True, summary
    
    def get_available_actions(self) -> list[str]:
        """Get available actions for the supervisor agent."""
        actions = []
        
        # If no workflow is planned, only allow planning
        if not self.workflow_plan:
            actions.append("plan_workflow")
            return actions
        
        # If workflow is planned but no agents created yet, allow agent creation
        if not self.agent_plans:
            actions.append("create_agent")
            return actions
        
        # If agents are planned but not all completed, allow agent creation and completion
        if len(self.completed_agents) < len(self.agent_plans):
            actions.append("create_agent")
            actions.append("complete_workflow")
        else:
            # All agents completed, only allow completion
            actions.append("complete_workflow")
        
        return actions
    
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file system operations."""
        import re
        # Remove or replace unsafe characters
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Replace spaces with underscores
        safe_filename = safe_filename.replace(' ', '_')
        # Limit length
        if len(safe_filename) > 100:
            safe_filename = safe_filename[:100]
        # Ensure it's not empty
        if not safe_filename:
            safe_filename = "unnamed"
        return safe_filename
    
    def _create_safe_trajectory_path(self, agent_name: str) -> Path:
        """Create safe trajectory directory path for an agent."""
        from getpass import getuser
        
        # Use stored run name or fallback to task ID
        run_name = self.run_name if self.run_name else (self.task.args.id if hasattr(self.task.args, 'id') else "supervisor_run")
        
        # Sanitize both run name and agent name
        safe_run_name = self._sanitize_filename(run_name)
        safe_agent_name = self._sanitize_filename(agent_name)
        
        # Follow pattern: trajectories/user/run_name/agent_name/
        agent_traj_dir = Path("trajectories") / Path(getuser()) / safe_run_name / safe_agent_name
        
        # Create directory with error handling
        try:
            agent_traj_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created trajectory directory: {agent_traj_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create trajectory directory: {e}")
            # Fallback to a simpler path
            fallback_dir = Path("trajectories") / "fallback" / safe_agent_name
            fallback_dir.mkdir(parents=True, exist_ok=True)
            agent_traj_dir = fallback_dir
        
        return agent_traj_dir
    
    def save_results(self) -> None:
        """
        Save supervisor results with aggregated agent performance.
        Override the base save_results to include all completed agent results.
        """
        import json
        from pathlib import Path
        from getpass import getuser
        
        results = {}
        
        # Aggregate all agent results by reading their individual result files
        if self.completed_agents:
            results["agent"] = []
            total_cost = 0.0
            total_steps = 0
            
            self.logger.info(f"Processing {len(self.completed_agents)} completed agents for result aggregation")
            
            for agent_result in self.completed_agents:
                agent_name = agent_result.get("agent_name", "unknown")
                agent_id = agent_result.get("agent_id", "unknown")
                steps_taken = agent_result.get("steps_taken", 0)
                exit_status = agent_result.get("exit_status", "unknown")
                
                self.logger.info(f"Processing agent: {agent_name} (ID: {agent_id})")
                
                # Try to read the agent's individual results file
                try:
                    # Construct the agent's result file path using the same logic as agent creation
                    run_name = self.run_name if self.run_name else "supervisor_run"
                    safe_run_name = self._sanitize_filename(run_name)
                    safe_agent_name = self._sanitize_filename(agent_name)
                    agent_traj_dir = Path("trajectories") / Path(getuser()) / safe_run_name / safe_agent_name
                    agent_results_file = agent_traj_dir / "results.json"
                    
                    self.logger.info(f"Looking for agent results at: {agent_results_file}")
                    self.logger.info(f"File exists: {agent_results_file.exists()}")
                    
                    if agent_results_file.exists():
                        with open(agent_results_file, 'r') as f:
                            agent_data = json.load(f)
                        
                        # Extract agent scores and add metadata
                        if "agent" in agent_data and isinstance(agent_data["agent"], list):
                            for score_entry in agent_data["agent"]:
                                # Add agent metadata to each score
                                enhanced_score = {
                                    **score_entry,
                                    "agent_name": agent_name,
                                    "agent_id": agent_id,
                                    "steps_taken": steps_taken,
                                    "exit_status": exit_status
                                }
                                results["agent"].append(enhanced_score)
                        
                        # Extract cost information if available
                        if "cost" in agent_data:
                            cost_info = agent_data["cost"]
                            if isinstance(cost_info, dict):
                                # Extract task_cost from model_stats
                                agent_cost = cost_info.get("task_cost", 0.0)
                                total_cost += float(agent_cost)
                                
                                # Add cost information to the enhanced score
                                enhanced_score["cost"] = agent_cost
                                enhanced_score["tokens_sent"] = cost_info.get("tokens_sent", 0)
                                enhanced_score["tokens_received"] = cost_info.get("tokens_received", 0)
                                enhanced_score["api_calls"] = cost_info.get("api_calls", 0)
                            else:
                                # Fallback for old format
                                total_cost += float(cost_info)
                        
                    else:
                        self.logger.warning(f"Agent results file not found: {agent_results_file}")
                        # Fallback: use the scores from completed_agents if available
                        if "scores" in agent_result and agent_result["scores"]:
                            self.logger.info(f"Using fallback scores from completed_agents for {agent_name}")
                            for score_entry in agent_result["scores"]:
                                enhanced_score = {
                                    **score_entry,
                                    "agent_name": agent_name,
                                    "agent_id": agent_id,
                                    "steps_taken": steps_taken,
                                    "exit_status": exit_status
                                }
                                
                                # Add cost information from completed_agents if available
                                if "cost" in agent_result:
                                    enhanced_score["cost"] = agent_result["cost"]
                                if "tokens_sent" in agent_result:
                                    enhanced_score["tokens_sent"] = agent_result["tokens_sent"]
                                if "tokens_received" in agent_result:
                                    enhanced_score["tokens_received"] = agent_result["tokens_received"]
                                if "api_calls" in agent_result:
                                    enhanced_score["api_calls"] = agent_result["api_calls"]
                                
                                results["agent"].append(enhanced_score)
                                
                                # Add to total cost
                                if "cost" in agent_result:
                                    total_cost += float(agent_result["cost"])
                        else:
                            self.logger.warning(f"No scores available for agent {agent_name}")
                
                except Exception as e:
                    self.logger.warning(f"Failed to read results for agent {agent_name}: {e}")
                    # Add fallback entry
                    results["agent"].append({
                        "agent_name": agent_name,
                        "agent_id": agent_id,
                        "steps_taken": steps_taken,
                        "exit_status": exit_status,
                        "error": "Failed to read agent results"
                    })
                
                total_steps += steps_taken
            
            # Add aggregated statistics
            results["supervisor_stats"] = {
                "total_agents": len(self.completed_agents),
                "total_steps": total_steps,
                "total_agent_cost": total_cost,
                "supervisor_cost": self.supervisor_cost,
                "total_cost": total_cost + self.supervisor_cost,
                "supervisor_tokens_sent": self.supervisor_tokens_sent,
                "supervisor_tokens_received": self.supervisor_tokens_received,
                "supervisor_api_calls": self.supervisor_api_calls,
                "average_score": self._calculate_average_score(results["agent"]) if results["agent"] else 0.0
            }
        
        # Add baseline scores
        assert self.task is not None
        if self.task.args.baseline_scores:
            results["baseline"] = self.task.args.baseline_scores[0]
        
        # Save to supervisor's results file
        # Use current agent's traj_path if available, otherwise create a fallback
        if self.current_agent and hasattr(self.current_agent, 'traj_path') and self.current_agent.traj_path:
            traj_path = self.current_agent.traj_path
        else:
            # Fallback: create a supervisor-specific trajectory path
            run_name = self.run_name if self.run_name else "supervisor_run"
            safe_run_name = self._sanitize_filename(run_name)
            traj_path = Path("trajectories") / Path(getuser()) / safe_run_name / "supervisor" / "supervisor.traj"
            traj_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_path = traj_path.parent / "results.json"
        results_path.write_text(json.dumps(results, indent=2))
        self.logger.info(f"Supervisor results saved to: {results_path}")
    
    def _calculate_average_score(self, agent_scores: list) -> float:
        """Calculate average score from agent scores."""
        if not agent_scores:
            return 0.0
        
        total_score = 0.0
        count = 0
        
        for score_entry in agent_scores:
            # Look for common score keys
            for key in ["Score", "score", "evaluation_score", "performance"]:
                if key in score_entry and isinstance(score_entry[key], (int, float)):
                    total_score += float(score_entry[key])
                    count += 1
                    break
        
        return total_score / count if count > 0 else 0.0
    
    def track_supervisor_cost(self, cost: float, tokens_sent: int, tokens_received: int, api_calls: int = 1) -> None:
        """Track supervisor API costs and usage."""
        self.supervisor_cost += cost
        self.supervisor_tokens_sent += tokens_sent
        self.supervisor_tokens_received += tokens_received
        self.supervisor_api_calls += api_calls
    
    def _create_previous_agent_context(self) -> str:
        """Create context about previous agents' work for the current agent."""
        if not self.completed_agents:
            return "No previous agents have completed work yet. You are the first agent in the workflow."
        
        context_parts = []
        context_parts.append(f"Previous agents completed: {len(self.completed_agents)}")
        
        for i, agent_result in enumerate(self.completed_agents[-3:], 1):  # Show last 3 agents
            agent_name = agent_result.get("agent_name", "unknown")
            role = agent_result.get("role", "unknown")
            steps = agent_result.get("steps_taken", 0)
            status = agent_result.get("exit_status", "unknown")
            scores = agent_result.get("scores", [])
            
            context_parts.append(f"Agent {i}: {agent_name} ({role}) - {steps} steps, {status}")
            
            if scores:
                score_summary = []
                for score in scores:
                    if isinstance(score, dict):
                        for key, value in score.items():
                            if key not in ["agent_name", "agent_id", "steps_taken", "exit_status"]:
                                score_summary.append(f"{key}: {value}")
                if score_summary:
                    context_parts.append(f"  Results: {', '.join(score_summary[:3])}")  # Show first 3 results
        
        context_parts.append("IMPORTANT: Check for files created by previous agents (look for recent timestamps, README files, or output files). Build upon their successful approaches and avoid repeating their failures.")
        
        return "\n".join(context_parts)