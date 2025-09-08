#!/usr/bin/env python3
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Supervisor Environment for MLGym

A multi-agent environment that allows a supervisor agent to orchestrate
specialized worker agents to complete research tasks.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json
import time

from mlgym.agent.decoupled import DecoupledAgent
from mlgym.environment.env import MLGymEnv
from mlgym.utils.log import get_logger
from mlgym.constants import DEFAULT_MAX_STEPS


@dataclass
class AgentPlan:
    """Plan for a single agent in the workflow."""
    name: str
    role: str
    instructions: str = ""
    max_steps: int = DEFAULT_MAX_STEPS  # Will be overridden by supervisor arguments


class SupervisorEnv(MLGymEnv):
    """
    Multi-agent environment managed by a supervisor agent.
    
    This environment allows a supervisor agent to:
    1. Plan research workflows
    2. Create and manage specialized worker agents
    3. Execute agents sequentially in a shared container
    4. Complete end-to-end research tasks
    """
    
    def __init__(self, args, devices, render_mode=None, supervisor_args=None):
        super().__init__(args, devices, render_mode)
        self.logger = get_logger("supervisor_env")
        
        # Store supervisor arguments for configuration
        self.supervisor_args = supervisor_args
        
        # Unified state tracking following MLGym patterns
        self.workflow_plan = {}
        self.agent_plans = []
        self.current_agent = None
        
        # Per-agent step tracking (MLGym pattern)
        self.agent_steps: dict[str, int] = {}  # agent_name -> step_count
        self.supervisor_step: int = 0
        
        # Agent completion tracking
        self.completed_agents: list[str] = []
        
        # Default agent configuration for worker agents
        self.default_agent_args = None
        self._initialized = False
        
        # Store run name for trajectory directories
        self.run_name = None
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """Reset the environment following MLGym pattern - always call parent for proper container management."""
        # ALWAYS call parent reset for proper container setup (MLGym pattern)
        observation, info = super().reset(seed=seed, options=options)
        
        # Reset supervisor state (container stays the same)
        self.workflow_plan = {}
        self.agent_plans = []
        self.current_agent = None
        self.agent_steps = {}
        self.supervisor_step = 0
        self.completed_agents = []
        
        # Create initial observation for supervisor
        observation = self._create_workflow_context()
        
        return observation, info
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute supervisor action and manage workflow."""
        try:
            self.logger.info(f"Supervisor action: {action}")
            
            # Check supervisor step limit (MLGym pattern)
            if self.supervisor_args and hasattr(self.supervisor_args, 'max_supervisor_steps'):
                if self.supervisor_step >= self.supervisor_args.max_supervisor_steps:
                    self.logger.warning(f"Max supervisor steps reached: {self.supervisor_args.max_supervisor_steps}")
                    return "Max supervisor steps reached", 0, True, {"exit_status": "max_supervisor_steps"}
            
            # Only handle supervisor-specific commands
            if action.startswith("plan_workflow"):
                result = self._handle_plan_workflow(action)
                if not result[3].get("dont_count_step", False):
                    self.supervisor_step += 1
                return result
            elif action.startswith("create_agent"):
                result = self._handle_create_agent(action)
                if not result[3].get("dont_count_step", False):
                    self.supervisor_step += 1
                return result
            elif action.startswith("complete_workflow"):
                result = self._handle_complete_workflow()
                if not result[3].get("dont_count_step", False):
                    self.supervisor_step += 1
                return result
            elif action.startswith("exit_format") or action.startswith("exit_error") or action.startswith("exit_api"):
                # Don't count parsing errors as steps
                return f"Parsing error occurred: {action}", 0, False, {"status": "parsing_error", "dont_count_step": True}
            else:
                # For all other commands (including worker agent commands), 
                # pass through to parent MLGymEnv.step method
                result = super().step(action)
                # Increment supervisor step for non-supervisor commands
                self.supervisor_step += 1
                return result
                
        except Exception as e:
            self.logger.error(f"Error in supervisor step: {e}")
            return f"Error: {e}", 0, False, {"error": str(e), "dont_count_step": True}
    
    def _handle_plan_workflow(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Handle plan_workflow command."""
        # Parse workflow plan from supervisor's action
        workflow_plan = self._parse_workflow_plan_from_action(action)
        
        return self._plan_workflow(workflow_plan)
    
    def _parse_workflow_plan_from_action(self, action: str) -> Dict[str, Any]:
        """Parse workflow plan from supervisor's action."""
        # Default workflow plan
        workflow_plan = {
            "strategy": "Complete the research task with multiple specialized agents",
            "agent_sequence": [],
            "success_criteria": "Achieve better performance than baseline"
        }
        
        # Try to extract strategy from the action
        if "strategy:" in action:
            strategy_start = action.find("strategy:")
            if strategy_start != -1:
                strategy_line = action[strategy_start:].split("\n")[0]
                strategy = strategy_line.replace("strategy:", "").strip()
                # Clean up any trailing brackets or quotes
                strategy = strategy.split("]")[0].split("[")[0].strip().strip('"')
                workflow_plan["strategy"] = strategy
        
        # Try to extract agent sequence from the action
        if "agents:" in action:
            agents_start = action.find("agents:")
            if agents_start != -1:
                agents_line = action[agents_start:].split("\n")[0]
                agents_part = agents_line.replace("agents:", "").strip()
                # Clean up any trailing brackets or quotes
                agents_part = agents_part.split("]")[0].split("[")[0].strip().strip('"')
                # Parse comma-separated agent names
                agent_sequence = [agent.strip().strip('"') for agent in agents_part.split(",") if agent.strip()]
                workflow_plan["agent_sequence"] = agent_sequence
        
        # Try to extract success criteria from the action
        if "criteria:" in action:
            criteria_start = action.find("criteria:")
            if criteria_start != -1:
                criteria_line = action[criteria_start:].split("\n")[0]
                criteria = criteria_line.replace("criteria:", "").strip()
                # Clean up any trailing brackets or quotes
                criteria = criteria.split("]")[0].split("[")[0].strip().strip('"')
                workflow_plan["success_criteria"] = criteria
        
        return workflow_plan
    
    def _handle_create_agent(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Handle create_agent command."""
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
            return "Invalid create_agent command format. Use: create_agent <name> '<instructions>'", 0, False, {"status": "invalid_format"}
    
    def _handle_complete_workflow(self) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Handle complete_workflow command."""
        return self._complete_workflow()
    
    def _plan_workflow(self, data: Dict[str, Any]) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Plan the research workflow with supervisor's strategy."""
        # Handle both direct workflow plan and wrapped workflow plan
        if "workflow_plan" in data:
            workflow_plan = data.get("workflow_plan", {})
        else:
            workflow_plan = data  # Direct workflow plan
        
        # If workflow is already planned, remind supervisor
        if self.workflow_plan:
            context = self._create_workflow_context()
            context += f"\n\nWORKFLOW ALREADY PLANNED:\n"
            context += f"  Strategy: {self.workflow_plan.get('strategy', 'Not set')}\n"
            context += f"  Success Criteria: {self.workflow_plan.get('success_criteria', 'Not set')}\n"
            context += f"  Planned Agents: {len(self.agent_plans)} agents\n"
            context += f"\nYou can proceed to create agents or modify the workflow plan."
            return context, 0, False, {"status": "workflow_already_planned"}
        
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
        
        context = self._create_workflow_context()
        return context, 0, False, {"status": "workflow_planned"}
    
    def _organize_agents(self, data: Dict[str, Any]) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Organize agents for the workflow."""
        # If agents are already planned, remind supervisor
        if self.agent_plans:
            context = self._create_workflow_context()
            context += f"\n\nAGENTS ALREADY PLANNED:\n"
            for i, plan in enumerate(self.agent_plans):
                context += f"  Agent {i+1}: {plan.name} ({plan.role})\n"
            context += f"\nYou can proceed to create agents or modify the workflow plan."
            return context, 0, False, {"status": "agents_already_planned"}
        
        # If no agents planned, create default sequence
        agent_organization = data.get("agent_organization", {})
        agent_sequence = agent_organization.get("agent_sequence", [])
        
        self.agent_plans = []
        for i, role in enumerate(agent_sequence):
            agent_plan = AgentPlan(
                name=f"agent_{i+1}",
                role=role,
                instructions=f"Focus on {role} responsibilities for this task",
                max_steps=20
            )
            self.agent_plans.append(agent_plan)
        
        self.logger.info(f"👥 Agents organized: {len(self.agent_plans)} agents")
        
        context = self._create_workflow_context()
        return context, 0, False, {"status": "agents_organized"}
    
    def _create_and_run_agent(self, data: Dict[str, Any]) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Create and run next agent with supervisor-defined parameters."""
        # Check if we've reached the maximum number of agents (MLGym pattern)
        if self.supervisor_args and hasattr(self.supervisor_args, 'max_agents_per_workflow'):
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
        
        # Agent ID is based on completion count (MLGym pattern)
        
        self.logger.info(f"🤖 Creating agent: {agent_plan.name} (Role: {agent_plan.role})")
        
        # Create dynamic task template with supervisor's specifications
        dynamic_task_template = self._create_dynamic_task_template(agent_plan)
        
        # Create a new agent config with the dynamic task template
        from mlgym.agent.base import AgentConfig
        dynamic_config = AgentConfig(
            system_template=self.default_agent_args.config.system_template,
            task_template=dynamic_task_template,
            next_step_template=self.default_agent_args.config.next_step_template,
            next_step_no_output_template=self.default_agent_args.config.next_step_no_output_template,
            tools=self.default_agent_args.config.tools,
            output_parser=self.default_agent_args.config.output_parser,
            history_processor="Last5Observations",
            history_processor_args=self.default_agent_args.config.history_processor_args
        )
        
        # Create agent with dynamic config
        from mlgym.agent.base import AgentArguments
        dynamic_args = AgentArguments(
            model=self.default_agent_args.model,
            agent_config_path=self.default_agent_args.agent_config_path,
            config=dynamic_config,
            log_verbose_to_console=self.default_agent_args.log_verbose_to_console
        )
        self.current_agent = DecoupledAgent(agent_plan.name, dynamic_args)
        
        # Setup agent following MLGym pattern
        self.current_agent.setup(self.task.args)
        self.current_agent._env = self  # Same environment/container
        
        # Create agent-specific trajectory directory with dynamic naming
        from pathlib import Path
        from getpass import getuser
        
        # Use stored run name or fallback to task ID
        run_name = self.run_name if self.run_name else (self.task.args.id if hasattr(self.task.args, 'id') else "supervisor_run")
        
        # Create dynamic agent name based on workflow role and completion count
        agent_name = f"{agent_plan.role}_{len(self.completed_agents) + 1}"
        
        # Follow pattern: trajectories/user/run_name/agent_name/
        agent_traj_dir = Path("trajectories") / Path(getuser()) / run_name / agent_name
        agent_traj_dir.mkdir(parents=True, exist_ok=True)
        self.current_agent.traj_dir = agent_traj_dir
        
        # Initialize agent trajectory and info (MLGym pattern)
        from mlgym.types import Trajectory
        from mlgym.types import AgentInfo
        self.current_agent.trajectory = Trajectory()
        self.current_agent.info = AgentInfo()
        
        # Initialize environment variables and shell functions (MLGym pattern)
        self.current_agent.init_environment_vars(self)
        
        # Verify shell functions are properly set up
        try:
            result = self.communicate("type edit")
            self.logger.info(f"Edit command setup check: {result}")
        except Exception as e:
            self.logger.warning(f"Edit command setup warning: {e}")
        
        # Run agent with plan parameters
        agent_result = self._run_agent_with_plan(agent_plan)
        
        # Store result and track completion (MLGym pattern)
        # Use the same agent name that was used for step tracking
        agent_name = f"{agent_plan.role}_{len(self.completed_agents) + 1}"
        self.completed_agents.append(agent_name)
        
        # Create context for supervisor
        context = self._create_workflow_context()
        return context, 0, False, agent_result
    
    def _run_agent_with_plan(self, agent_plan: AgentPlan) -> Dict[str, Any]:
        """Run current agent according to supervisor's plan."""
        # Use supervisor arguments for max_steps if available, otherwise use agent_plan default
        if self.supervisor_args and hasattr(self.supervisor_args, 'max_steps_per_agent'):
            max_steps = self.supervisor_args.max_steps_per_agent
        else:
            max_steps = agent_plan.max_steps
        
        # Initialize step counter for this agent (MLGym pattern)
        agent_name = f"{agent_plan.role}_{len(self.completed_agents) + 1}"
        self.agent_steps[agent_name] = 0
        
        self.logger.info(f"🔄 Running agent {self.current_agent.name} with role: {agent_plan.role}")
        
        # Initialize observation to None (normal MLGym flow)
        observation = None
        
        # Run agent until completion or max steps (MLGym pattern)
        observation = None
        done = False
        
        while not done and self.agent_steps[agent_name] < max_steps:
            # Start timing (following MLGym pattern)
            execution_t0 = time.perf_counter()
            
            # Get proper environment state for agent
            try:
                # Create proper state with file context (MLGym pattern)
                state_data = {
                    "open_file": "n/a",
                    "working_dir": "/home/agent/workspace",
                    "current_step": self.agent_steps[agent_name],
                    "remaining_steps": max_steps - self.agent_steps[agent_name]
                }
                state = json.dumps(state_data)
            except Exception as e:
                self.logger.warning(f"Failed to create state: {e}")
                state = "{}"
            
            # Get available actions for worker agent (all standard MLGym commands)
            available_actions = self.get_worker_available_actions()
            
            # Agent proposes action using normal MLGym observation flow
            thought, action, output = self.current_agent.forward(
                observation,  # Use normal observation (None initially, then result of previous action)
                available_actions,
                state
            )
            
            # CRITICAL: Process action through the agent's tool handler to handle multi-line commands
            run_action = self.current_agent.tools.guard_multiline_input(action).strip()
            
            # Execute action in container (using MLGymEnv capabilities)
            observation, reward, done, info = super().step(run_action)
            
            # Calculate execution time (following MLGym pattern)
            execution_time = time.perf_counter() - execution_t0
            
            # Create trajectory step (following MLGym pattern)
            from mlgym.types import TrajectoryStep
            trajectory_step = TrajectoryStep(
                state=state,
                response=output,
                thought=thought,
                action=action,
                execution_time=execution_time,
                observation=observation,
            )
            self.current_agent.trajectory.append(trajectory_step)
            
            # Update model statistics (following MLGym pattern)
            from mlgym.backend.base import APIStats
            model_stats: APIStats = self.current_agent.model.stats
            self.current_agent.info["model_stats"] = model_stats.to_dict()
            
            # Save trajectory after each step (following MLGym pattern)
            self.current_agent.save_trajectory()
            
            # Update agent step counter (MLGym pattern)
            self.agent_steps[agent_name] += 1
            
            # Check if agent is done
            if done or action.strip() == self.current_agent.tools.submit_command:
                self.logger.info(f"✅ Agent {self.current_agent.name} completed: {info.get('exit_status', 'completed')}")
                break
        
        # Save final results for the agent
        self.current_agent.save_results()
        
        # Return agent result
        return {
            "agent_name": agent_name,  # Use consistent agent name
            "agent_id": len(self.completed_agents) + 1,
            "role": agent_plan.role,
            "instructions": agent_plan.instructions,
            "steps_taken": self.agent_steps[agent_name],
            "max_steps_planned": max_steps,
            "exit_status": info.get("exit_status", "completed"),
            "final_observation": observation if 'observation' in locals() else "No observation"
        }
    
    def _create_task_observation(self, agent_plan: AgentPlan) -> str:
        """Create task-specific observation for the agent."""
        try:
            # Get task information from the task config
            task_description = self.task.args.description
            baseline_scores = self.task.args.baseline_scores
            
            # Format baseline scores
            baseline_info = ""
            if baseline_scores and len(baseline_scores) > 0:
                baseline_info = f"BASELINE SCORES: {baseline_scores[0]}\n"  # Use first baseline score
            
            # Get submission format from task
            submission_format = "Create appropriate submission file"
            if hasattr(self.task.args, 'evaluation_paths') and self.task.args.evaluation_paths:
                submission_format = f"Follow the evaluation script: {self.task.args.evaluation_paths[0]}"
            
            task_info = f"""
TASK DESCRIPTION:
{task_description}

{baseline_info}
SUBMISSION FORMAT:
{submission_format}

Your Role: {agent_plan.role}
Instructions: {agent_plan.instructions}

WORKING DIRECTORY: /home/agent/workspace
CURRENT STEP: {self.current_step}
REMAINING STEPS: {20 - self.current_step}

Start by exploring the data and understanding the task requirements.
"""
            
            return task_info
            
        except Exception as e:
            self.logger.error(f"Error creating task observation: {e}")
            # Fallback to basic task info
            return f"""
TASK: Complete the assigned research task

Your Role: {agent_plan.role}
Instructions: {agent_plan.instructions}

WORKING DIRECTORY: /home/agent/workspace
CURRENT STEP: {self.current_step}
REMAINING STEPS: {20 - self.current_step}

Start by exploring the workspace and understanding the task requirements.
"""
    
    def _create_custom_task_description(self, agent_plan: AgentPlan) -> str:
        """Create custom task description with agent context."""
        try:
            # Get task information from the task config
            task_description = self.task.args.description
            baseline_scores = self.task.args.baseline_scores
            
            # Format baseline scores
            baseline_info = ""
            if baseline_scores and len(baseline_scores) > 0:
                baseline_info = f"\nBASELINE SCORES: {baseline_scores[0]}"
            
            # Get submission format from task
            submission_format = "Create appropriate submission file"
            if hasattr(self.task.args, 'evaluation_paths') and self.task.args.evaluation_paths:
                submission_format = f"Follow the evaluation script: {self.task.args.evaluation_paths[0]}"
            
            custom_description = f"""
{task_description}

{baseline_info}

SUBMISSION FORMAT: {submission_format}

AGENT CONTEXT:
- Your Role: {agent_plan.role}
- Instructions: {agent_plan.instructions}
- Working Directory: /home/agent/workspace

Start by exploring the data and understanding the task requirements. Focus on your specific role and responsibilities.
"""
            
            return custom_description
            
        except Exception as e:
            self.logger.error(f"Error creating custom task description: {e}")
            # Fallback to basic task info
            return f"""
Complete the assigned research task.

AGENT CONTEXT:
- Your Role: {agent_plan.role}
- Instructions: {agent_plan.instructions}
- Working Directory: /home/agent/workspace

Start by exploring the workspace and understanding the task requirements.
"""
    
    def _create_dynamic_task_template(self, agent_plan: AgentPlan) -> str:
        """Create dynamic task template with supervisor's specifications."""
        return f"""
We're currently solving the following task. Here's the task description:

TASK DESCRIPTION:
{self.task.args.description}

YOUR ROLE:
{agent_plan.role}

SUPERVISOR INSTRUCTIONS:
{agent_plan.instructions}

INSTRUCTIONS:
Now, you're going to train a model to improve performance on this task. Your terminal session has started and you're in the workspace root directory (/home/agent/workspace). You can use any bash commands or the special interface to help you. Edit all the file you need or create a new training script.

Remember, YOU CAN ONLY ENTER ONE COMMAND AT A TIME. You should always wait for feedback after every command.

When you're satisfied with all of the changes you have made, you can run your code. Your code should produce a valid submission artefact. Please follow the instructions in SUBMISSION FORMAT section above and adhere to the guidelines provided for generating submission artefacts. You can also look at the `evaluate.py` provided to you to see if you are following the correct format and naming scheme for your submission artefacts.

Note however that you cannot use any interactive session commands (e.g. python, vim) in this environment, but you can write scripts and run them. E.g. you can write a python script and then run it with `python <script_name>.py`.

NOTE ABOUT THE EDIT AND INSERT COMMANDs: Indentation really matters! When editing a file, make sure to insert appropriate indentation before each line!

IMPORTANT: Do NOT try to open large files (like CSV files) directly with the `open` command. Large files can cause the environment to hang. Instead, use commands like `head`, `tail`, `wc -l`, or `ls -lh` to explore file sizes and get a preview of the data.

WORKING DIRECTORY: /home/agent/workspace
CURRENT STEP: {{current_step}}
REMAINING STEPS: {{remaining_steps}}
OPEN FILE: {{open_file}}
bash-$
"""
    
    def _monitor_current_agent(self) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Monitor current agent progress."""
        if not self.current_agent:
            return "No current agent to monitor", 0, False, {"status": "no_agent"}
        
        # Get agent status (MLGym pattern)
        agent_name = self.current_agent.name
        current_step = self.agent_steps.get(agent_name, 0)
        max_steps = self.supervisor_args.max_steps_per_agent if self.supervisor_args else DEFAULT_MAX_STEPS
        agent_status = {
            "agent_name": agent_name,
            "current_step": current_step,
            "max_steps": max_steps,
            "progress": f"{current_step}/{max_steps} steps completed"
        }
        
        context = self._create_workflow_context()
        return context, 0, False, agent_status
    
    def _complete_workflow(self) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Complete the workflow."""
        self.logger.info("🎉 Workflow completed successfully")
        
        # Create final summary (MLGym pattern)
        summary = {
            "status": "completed",
            "total_agents": len(self.completed_agents),
            "agents_completed": self.completed_agents,
            "workflow_plan": self.workflow_plan
        }
        
        return "Workflow completed successfully", 1.0, True, summary
    
    def _create_workflow_context(self) -> str:
        """Create context string for supervisor."""
        context = f"""
WORKFLOW STATE:
Current Agent Progress: {len(self.completed_agents)}/{len(self.agent_plans)} agents completed
Supervisor Step: {self.supervisor_step}

Workflow Strategy: {self.workflow_plan.get('strategy', 'Not set')}
Success Criteria: {self.workflow_plan.get('success_criteria', 'Not set')}

Planned Agents: {len(self.agent_plans)} agents
"""
        
        if self.agent_plans:
            context += "\nAgent Plans:\n"
            for i, plan in enumerate(self.agent_plans):
                status = "✅ Completed" if i < len(self.completed_agents) else "⏳ Pending"
                context += f"  Agent {i+1}: {plan.name} ({plan.role}) - {status}\n"
        
        if self.completed_agents:
            context += "\nCompleted Agents:\n"
            for i, agent_name in enumerate(self.completed_agents):
                steps = self.agent_steps.get(agent_name, 0)
                context += f"  Agent {i+1}: {agent_name} - {steps} steps completed\n"
        
        return context
    
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
    
    def get_worker_available_actions(self) -> list[str]:
        """Get available actions for worker agents (all standard MLGym commands)."""
        # Return all available commands from the environment
        # This includes: ls, cd, open, edit, insert, create, submit, validate, etc.
        return [
            "ls", "cd", "pwd", "cat", "head", "tail", "wc", "grep", "find",
            "open", "edit", "insert", "create", "submit", "validate",
            "python", "pip", "conda", "git", "mkdir", "rm", "cp", "mv"
        ]