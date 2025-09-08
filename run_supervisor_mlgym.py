#!/usr/bin/env python3
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

MLGym-Compliant Supervisor Run Script

This script runs a supervisor agent that orchestrates multiple worker agents
to complete research tasks end-to-end.

Follows proper MLGym patterns for argument parsing, trajectory handling,
logging, and error management.
"""

import asyncio
import datetime
import logging
import time
import traceback
from dataclasses import dataclass, field
from getpass import getuser
from pathlib import Path
from typing import Any

import gymnasium as gym
import yaml
from simple_parsing import parse
from simple_parsing.helpers.fields import field as simple_field
from simple_parsing.helpers.flatten import FlattenedAccess
from simple_parsing.helpers.serialization.serializable import FrozenSerializable

from mlgym import CONFIG_DIR
from mlgym.agent.base import AgentArguments, BaseAgent
from mlgym.agent.supervisor_agent import SupervisorAgent
from mlgym.backend.base import ModelArguments
from mlgym.environment.env import EnvironmentArguments, MLGymEnv
from mlgym.environment.registration import register_task
from mlgym.environment.supervisor_env_mlgym import SupervisorEnvMLGym
from mlgym.utils.config import load_environment_variables
from mlgym.utils.extras import get_devices, multiline_representer
from mlgym.utils.log import add_file_handler, get_logger
from mlgym.constants import (
    DEFAULT_MAX_STEPS, DEFAULT_MAX_SUPERVISOR_STEPS, DEFAULT_MAX_AGENTS_PER_WORKFLOW,
    DEFAULT_MAX_STEPS_PER_AGENT, DEFAULT_TEMPERATURE, DEFAULT_TOTAL_COST_LIMIT,
    DEFAULT_PER_INSTANCE_COST_LIMIT, DEFAULT_CONTAINER_TYPE, DEFAULT_SEED
)

try:
    import rich  # noqa: F401
except ModuleNotFoundError as e:
    msg = (
        "You probably either forgot to install the dependencies "
        "or forgot to activate your conda or virtual environment."
    )
    raise RuntimeError(msg) from e

from rich.markdown import Markdown

try:
    from rich_argparse import RichHelpFormatter
except ImportError as e:
    msg = "Please install the rich_argparse package with `pip install rich_argparse`."
    raise ImportError(msg) from e

__doc__ = """Run supervisor workflow with proper MLGym patterns."""

logger = get_logger("mlgym-supervisor")
logging.getLogger("simple_parsing").setLevel(logging.WARNING)


@dataclass(frozen=True)
class SupervisorScriptArguments(FlattenedAccess, FrozenSerializable):
    """Configure the control flow of the supervisor run script"""

    environment: EnvironmentArguments
    supervisor_agent: AgentArguments
    # Maximum supervisor steps (different from agent steps)
    max_supervisor_steps: int = DEFAULT_MAX_SUPERVISOR_STEPS
    # Maximum number of agents that can be created in a single workflow
    max_agents_per_workflow: int = DEFAULT_MAX_AGENTS_PER_WORKFLOW
    # Maximum steps per individual agent (overrides environment max_steps for worker agents)
    max_steps_per_agent: int = DEFAULT_MAX_STEPS_PER_AGENT
    # Dump the entire config file to a log
    print_config: bool = False
    # Raise unhandled exceptions during the run (useful for debugging)
    raise_exceptions: bool = False
    # Suffix for the run name (used for example in trajectory directory naming)
    suffix: str = ""

    def run_name(self) -> str:
        """Generate a unique name for this supervisor run based on the arguments."""
        # Get task and model info
        assert self.environment.task is not None
        task_id = self.environment.task.id
        model_name = self.supervisor_agent.model.model_name.replace(":", "-")
        
        # Create timestamp when run starts
        timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        
        # Format: supervisor_mlgym__{task}__{model}__{timestamp}
        base_name = f"supervisor_mlgym__{task_id}__{model_name}__{timestamp}"
        return base_name + (f"__{self.suffix}" if self.suffix else "")

    def register_envs(self) -> None:
        """Register the supervisor environment."""
        register_task(self.environment)

    def __post_init__(self) -> None:
        """Post-initialization validation."""
        self.register_envs()


class _ContinueLoop(Exception):  # noqa: N818
    """Used for internal control flow"""
    pass


class SupervisorMain:
    def __init__(self, args: SupervisorScriptArguments) -> None:
        """Initialize the SupervisorMain class with the given arguments."""
        self.args = args

    def run(self, supervisor: SupervisorAgent, env: SupervisorEnvMLGym) -> None:
        """Run the supervisor workflow with proper MLGym patterns."""
        # Create hierarchical trajectory directory structure
        run_name = self.args.run_name()
        base_traj_dir = Path("trajectories") / Path(getuser()) / run_name
        base_traj_dir.mkdir(parents=True, exist_ok=True)
        
        # Set run name in environment for agent trajectory directories
        env.run_name = run_name
        
        # Create supervisor-specific directory
        supervisor_traj_dir = base_traj_dir / "supervisor"
        supervisor_traj_dir.mkdir(parents=True, exist_ok=True)
        
        # Set supervisor's trajectory directory
        supervisor.traj_dir = supervisor_traj_dir
        
        # Create timestamped log file in base directory (following MLGym pattern)
        timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        log_path = base_traj_dir / f"supervisor-run-{timestamp}.log"
        logger.info("Logging to %s", log_path)
        
        # Add file handler with all relevant loggers (following MLGym pattern)
        add_file_handler(log_path, [
            "mlgym-supervisor", "MLGym", supervisor.name, "api_models", 
            "env_utils", "MLGymEnv", "supervisor_env_mlgym"
        ])
        
        # Print config if requested
        if self.args.print_config:
            logger.info(f"📙 Arguments: {self.args.dumps_yaml()}")
        
        # Save arguments following MLGym pattern
        self._save_arguments(supervisor_traj_dir)

        # Initialize environment and get initial observation FIRST
        observation, info = env.reset()
        if info is None:
            raise _ContinueLoop

        # Now we can safely access env.task
        assert env.task is not None
        task_id = env.task.args.id
        logger.info("▶️  Beginning supervisor task " + str(task_id))

        # Set up supervisor agent properly (following MLGym patterns)
        supervisor.setup(env.task.args)
        supervisor._env = env
        supervisor.init_environment_vars(env)
        
        # Set default agent args for worker agents (use enhanced template)
        env.default_agent_args = AgentArguments(
            model=ModelArguments(model_name="litellm:gpt-4o-mini"),
            agent_config_path=CONFIG_DIR / "agents" / "worker_template_enhanced.yaml",
            log_verbose_to_console=self.args.environment.verbose
        )

        # Set trajectory directory for supervisor (already set above)
        # supervisor.traj_dir is already set to supervisor_traj_dir

        # Run supervisor workflow
        logger.info("🚀 Starting supervisor workflow")
        
        done = False
        
        while not done and env.supervisor_step < self.args.max_supervisor_steps:
            logger.info(f"🔄 Supervisor step {env.supervisor_step + 1}")
            
            # Start timing (following MLGym pattern)
            execution_t0 = time.perf_counter()
            
            # Get available actions and state
            available_actions = env.get_available_actions()
            
            # Create a proper JSON state for the supervisor
            # Use the environment's supervisor_step counter for consistency
            state_data = {
                "working_dir": "/home/agent/workspace",
                "current_step": env.supervisor_step,  # Use environment's counter
                "remaining_steps": self.args.max_supervisor_steps - env.supervisor_step  # Use environment's counter
            }
            import json
            state = json.dumps(state_data)  # Convert to proper JSON string
            
            # Supervisor proposes action
            thought, action, output = supervisor.forward(observation, available_actions, state)
            
            # Execute action in environment
            observation, reward, done, info = env.step(action)
            
            # Calculate execution time (following MLGym pattern)
            execution_time = time.perf_counter() - execution_t0
            
            # Check if this step should be counted
            if info.get("dont_count_step", False):
                # Don't count this step - environment already handles this
                logger.info(f"🔄 Parsing error - not counting step {env.supervisor_step + 1}")
            else:
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
                supervisor.trajectory.append(trajectory_step)
                
                # Update model statistics (following MLGym pattern)
                from mlgym.backend.base import APIStats
                model_stats: APIStats = supervisor.model.stats
                supervisor.info["model_stats"] = model_stats.to_dict()
                
                # Track supervisor costs in environment
                env.track_supervisor_cost(
                    cost=model_stats.task_cost,
                    tokens_sent=model_stats.tokens_sent,
                    tokens_received=model_stats.tokens_received,
                    api_calls=model_stats.api_calls
                )
                
                # Save trajectory after each step (following MLGym pattern)
                supervisor.save_trajectory()
            
            # Log supervisor decision
            logger.info(f"Supervisor decision: {action}")
            
            if done:
                logger.info("✅ Supervisor workflow completed")
                break
        
        # Log final results
        logger.info("📊 Final Results:")
        logger.info(f"   Workflow Status: {info.get('status', 'completed')}")
        logger.info(f"   Agents Planned: {len(env.agent_plans)}")
        logger.info(f"   Agents Completed: {len(env.completed_agents)}")
        logger.info(f"   Supervisor Steps: {env.supervisor_step}")
        logger.info(f"   Supervisor Cost: ${env.supervisor_cost:.4f}")
        logger.info(f"   Supervisor Tokens: {env.supervisor_tokens_sent:,} sent, {env.supervisor_tokens_received:,} received")
        
        if env.workflow_plan:
            logger.info(f"   Workflow Plan: {env.workflow_plan}")
        
        for i, agent_result in enumerate(env.completed_agents):
            # Extract agent information from the completed_agents list
            agent_name = agent_result.get("agent_name", f"agent_{i+1}")
            agent_id = agent_result.get("agent_id", f"agent_{i+1}")
            steps_taken = agent_result.get("steps_taken", 0)
            exit_status = agent_result.get("exit_status", "unknown")
            role = agent_result.get("role", "unknown")
            
            # Try to get additional info from agent_tracking if available
            agent_info = None
            for tracking in env.agent_tracking.values():
                if tracking.get("name") == agent_name or tracking.get("agent_id") == agent_id:
                    agent_info = tracking
                    break
            
            # Use agent_result data first, fallback to agent_tracking
            final_steps = steps_taken if steps_taken > 0 else (agent_info.get("steps", 0) if agent_info else 0)
            final_status = exit_status if exit_status != "unknown" else (agent_info.get("status", "unknown") if agent_info else "unknown")
            
            # Extract cost information
            agent_cost = agent_result.get("cost", 0.0)
            agent_tokens_sent = agent_result.get("tokens_sent", 0)
            agent_tokens_received = agent_result.get("tokens_received", 0)
            
            logger.info(f"   Agent {i+1}: {agent_name} ({role}) - {final_steps} steps completed - {final_status} - Cost: ${agent_cost:.4f} - Tokens: {agent_tokens_sent:,} sent, {agent_tokens_received:,} received")
        
        # Save final results for supervisor (trajectory already saved per-step)
        supervisor.save_results()
        
        # Save environment results with agent aggregation
        env.save_results()
        
        logger.info(f"✅ Supervisor workflow completed. Results saved to: {base_traj_dir}")

    async def run_supervisor(self) -> None:
        """Run the supervisor workflow with proper error handling following MLGym pattern."""
        # Create supervisor agent
        supervisor = SupervisorAgent("supervisor", self.args.supervisor_agent)
        
        # Create MLGym-compliant supervisor environment
        env = SupervisorEnvMLGym(self.args.environment, devices=[], render_mode=None, supervisor_args=self.args)
        
        try:
            await asyncio.to_thread(self.run, supervisor, env)
        except _ContinueLoop:
            pass
        except KeyboardInterrupt:
            logger.info("Exiting MLGym supervisor environment...")
            env.close()
        except SystemExit:
            logger.critical("❌ Exiting because SystemExit was called")
            env.close()
            logger.info("Container closed")
            raise
        except Exception as e:
            logger.warning(traceback.format_exc())
            if self.args.raise_exceptions:
                env.close()
                raise
            if env.task:
                logger.warning(f"❌ Failed on {env.task.args.id}: {e}")
            else:
                logger.warning("❌ Failed on unknown instance")
            env.reset_container()  # Reset container but keep it alive (MLGym pattern)
        finally:
            env.close()  # Always close container (MLGym pattern)

    def _save_arguments(self, traj_dir: Path) -> None:
        """Save the arguments to a yaml file to the run's trajectory directory."""
        log_path = traj_dir / "args.yaml"

        if log_path.exists():
            try:
                other_args = self.args.load_yaml(log_path)
                if self.args.dumps_yaml() != other_args.dumps_yaml():
                    logger.warning("**************************************************")
                    logger.warning("Found existing args.yaml with different arguments!")
                    logger.warning("**************************************************")
            except Exception as e:
                logger.warning(f"Failed to load existing args.yaml: {e}")

        with log_path.open("w") as f:
            self.args.dump_yaml(f)


def get_supervisor_args(args: list[str] | None = None) -> SupervisorScriptArguments:
    """Parse command line arguments and return a SupervisorScriptArguments object."""
    defaults = SupervisorScriptArguments(
        environment=EnvironmentArguments(
            task_config_path="tasks/battleOfSexes.yaml",
            max_steps=DEFAULT_MAX_STEPS,
            seed=DEFAULT_SEED,
            container_type=DEFAULT_CONTAINER_TYPE,
            verbose=True,
        ),
        supervisor_agent=AgentArguments(
            model=ModelArguments(
                model_name="litellm:gpt-4o-mini",
                total_cost_limit=DEFAULT_TOTAL_COST_LIMIT,
                per_instance_cost_limit=DEFAULT_PER_INSTANCE_COST_LIMIT,
                temperature=DEFAULT_TEMPERATURE,
            ),
            agent_config_path=CONFIG_DIR / "agents" / "supervisor_enhanced.yaml",
        ),
        max_supervisor_steps=DEFAULT_MAX_SUPERVISOR_STEPS,
        max_agents_per_workflow=DEFAULT_MAX_AGENTS_PER_WORKFLOW,
        max_steps_per_agent=DEFAULT_MAX_STEPS_PER_AGENT,
    )
    yaml.add_representer(str, multiline_representer)

    new_args = parse(
        SupervisorScriptArguments,
        default=defaults,
        add_config_path_arg=False,
        args=args,
        formatter_class=RichHelpFormatter,
        description=Markdown(__doc__ or ""),
    )

    return new_args


def main(args: SupervisorScriptArguments) -> None:
    """Main entry point following MLGym pattern."""
    asyncio.run(SupervisorMain(args).run_supervisor())


if __name__ == "__main__":
    load_environment_variables()
    main(get_supervisor_args()) 