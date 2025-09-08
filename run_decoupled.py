"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Main script for running MLGym with decoupled supervisor workflow.

Adapted from run.py
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import traceback
from dataclasses import dataclass
from getpass import getuser
from pathlib import Path

import gymnasium as gym
import yaml
from simple_parsing import parse
from simple_parsing.helpers.fields import field
from simple_parsing.helpers.flatten import FlattenedAccess
from simple_parsing.helpers.serialization.serializable import FrozenSerializable

from mlgym import CONFIG_DIR
from mlgym.agent.decoupled import DecoupledAgent, AgentArguments
from mlgym.agent.supervisor import SupervisorAgent
from mlgym.backend.base import ModelArguments
from mlgym.environment.env import EnvironmentArguments, MLGymEnv
from mlgym.environment.registration import register_task
from mlgym.utils.config import load_environment_variables
from mlgym.utils.extras import get_devices, multiline_representer
from mlgym.utils.log import add_file_handler, get_logger

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

__doc__ = """Run inference with decoupled supervisor workflow."""

logger = get_logger("mlgym-decoupled-run")
logging.getLogger("simple_parsing").setLevel(logging.WARNING)
logger.info(f"🍟 DOCKER_HOST: {os.environ.get('DOCKER_HOST')}")


@dataclass(frozen=True)
class DecoupledScriptArguments(FlattenedAccess, FrozenSerializable):
    """Configure the control flow of the decoupled supervisor run script"""

    environment: EnvironmentArguments
    agent: AgentArguments
    supervisor: AgentArguments
    # if None, envArgs.task_args should be set to appropriate task config file
    benchmark: str | None = None
    # Dump the entire config file to a log
    print_config: bool = False
    # skip tasks with existing trajectories
    skip_existing: bool = False
    # Suffix for the run name (used for example in trajectory directory naming)
    suffix: str = ""
    # Raise unhandled exceptions during the run (useful for debugging)
    raise_exceptions: bool = False
    # number of GPUs per agent, if 0, CPU will be used
    gpus_per_agent: int = 0
    # number of agents to run in parallel
    num_agents: int = 1
    # List of GPU Ids to use, if empty, all available GPUs will be used
    gpus: list[int] = field(default_factory=list)  # noqa: RUF009

    def run_name(self) -> str:
        """Generate a unique name for this run based on the arguments."""
        model_name = self.agent.model.model_name.replace(":", "-")
        supervisor_model_name = self.supervisor.model.model_name.replace(":", "-")
        assert self.environment.task is not None
        task_id = self.environment.task.id
        assert self.agent.agent_config_path is not None
        config_stem = Path(self.agent.agent_config_path).stem
        assert self.supervisor.agent_config_path is not None
        supervisor_config_stem = Path(self.supervisor.agent_config_path).stem

        temp = self.agent.model.temperature
        top_p = self.agent.model.top_p

        per_instance_cost_limit = self.agent.model.per_instance_cost_limit
        install_env = False

        return (
            f"DECOUPLED_{model_name}_{supervisor_model_name}__{task_id}__{config_stem}_{supervisor_config_stem}__t-{temp:.2f}__p-{top_p:.2f}"
            f"__cost-{per_instance_cost_limit:.2f}__env-{install_env}__{self.suffix}"
        )

    def register_envs(self) -> None:
        # Assume we are not using benchmark for now. So we only need to register the env for the task specified in task_args.
        register_task(self.environment)

    def __post_init__(self) -> None:
        # check whether benchmark or env_args.task_args is set
        if self.benchmark is None and self.environment.task_config_path is None:
            msg = "Either benchmark or environment.task_config_path must be set."
            raise ValueError(msg)


class Main:
    def __init__(self, args: DecoupledScriptArguments) -> None:
        self.args = args

    def run_decoupled_workflow(self, agent: DecoupledAgent, supervisor: SupervisorAgent, env: MLGymEnv, devices: list[str], run_idx: int) -> None:
        """Run the decoupled supervisor workflow"""
        
        # Create run directory
        main_run_name = f"{self.args.run_name()}_{datetime.datetime.now().strftime('%y%m%d%H%M%S')}"
        main_run_dir = Path("trajectories") / Path(getuser()) / main_run_name
        main_run_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        log_path = main_run_dir / f"decoupled-run-{timestamp}.log"
        logger.info("Logging to %s", log_path)
        add_file_handler(log_path, ["mlgym-decoupled-run", "MLGym", "decoupled-agent", "supervisor"])

        # Save arguments
        yaml.add_representer(str, multiline_representer)
        args_path = main_run_dir / "args.yaml"
        with args_path.open("w") as f:
            yaml.dump({
                "environment": self.args.environment.asdict(),
                "agent": self.args.agent.asdict(),
                "supervisor": self.args.supervisor.asdict()
            }, f)

        # Initialize agents
        agent.setup(env.task.args)
        agent._env = env
        agent.init_environment_vars(env)
        
        supervisor.setup(env.task.args)
        supervisor._env = env
        supervisor.init_environment_vars(env)

        # Run the decoupled workflow
        try:
            logger.info("Starting decoupled supervisor workflow")
            
            observation = env.reset()[0]["observation"]
            step = 0
            max_steps = self.args.environment.max_steps
            done = False
            
            while step < max_steps and not done:
                step += 1
                logger.info(f"Step {step}/{max_steps}")
                
                # Step 1: Agent proposes action
                logger.info("Agent proposing action...")
                thought, action, output = agent.propose_action(observation)
                logger.info(f"Agent proposed: {action[:100]}...")
                
                # Step 2: Supervisor reviews action
                logger.info("Supervisor reviewing action...")
                context = {
                    "agent_name": agent.name,
                    "thought": thought,
                    "observation": observation,
                    "output": output
                }
                approved, feedback = supervisor.review_action(action, context)
                
                # Step 3: Handle supervisor decision
                if approved:
                    logger.info("Supervisor APPROVED action")
                    # Execute action in environment
                    env_result, _, done, _info = env.step(action)
                    # Handle different return formats
                    if isinstance(env_result, dict) and "observation" in env_result:
                        observation = env_result["observation"]
                    else:
                        observation = str(env_result)
                    
                    # Check if agent wants to submit
                    if action.strip() == agent.tools.submit_command:
                        logger.info("Agent submitted solution")
                        break
                        
                else:
                    logger.info("Supervisor REJECTED action")
                    # Provide feedback as new observation
                    observation = f"""
SUPERVISOR FEEDBACK:
Your proposed action has been REJECTED.

PROPOSED ACTION:
{action}

REJECTION REASON:
{feedback}

Please reconsider your approach based on this feedback.
"""
                    # Add feedback to agent's history
                    agent.receive_feedback(feedback)
                    
            logger.info("Decoupled supervisor workflow completed")
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            if self.args.raise_exceptions:
                raise
            traceback.print_exc()
        finally:
            env.close()
            logger.info("Environment closed.")

    async def run_agent(self, devices: list[str], run_idx: int) -> None:
        # Reset environment
        env: MLGymEnv = gym.make(f"mlgym/{self.args.environment.task.id}", devices=devices).unwrapped  # type: ignore

        # Create agents
        agent = DecoupledAgent(f"decoupled_agent_{run_idx}", self.args.agent)
        supervisor = SupervisorAgent(f"supervisor_{run_idx}", self.args.supervisor)

        try:
            self.run_decoupled_workflow(agent, supervisor, env, devices, run_idx)
        except Exception as e:
            logger.error(f"Agent {run_idx} failed: {e}")
            if self.args.raise_exceptions:
                raise
            traceback.print_exc()

    async def main(self) -> None:
        """Main entry point for the decoupled supervisor workflow"""
        
        if self.args.print_config:
            logger.info("Configuration:")
            logger.info(yaml.dump(self.args.asdict(), default_flow_style=False))
            return

        # Register environments
        self.args.register_envs()

        # Get devices (same logic as original run.py)
        if self.args.gpus_per_agent > 0:
            # get all the devices available
            _devices = get_devices() if len(self.args.gpus) == 0 else self.args.gpus
            devices = [str(x) for x in _devices]
            if self.args.gpus_per_agent * self.args.num_agents > len(devices):
                msg = f"Not enough GPUs available. Required: {self.args.gpus_per_agent * self.args.num_agents}, Available: {len(devices)}"
                raise RuntimeError(msg)
            agent_devices = []
            for i in range(self.args.num_agents):
                gpus = devices[self.args.gpus_per_agent * i : self.args.gpus_per_agent * (i + 1)]
                agent_devices.append(gpus)
        else:
            agent_devices = [[f"cpu_{i}"] for i in range(self.args.num_agents)]

        # Run agents
        tasks = []
        for run_idx in range(self.args.num_agents):
            task = asyncio.create_task(self.run_agent(agent_devices[run_idx], run_idx))
            tasks.append(task)

        await asyncio.gather(*tasks)

    def _save_arguments(self, traj_dir: Path) -> None:
        """Save the arguments to a YAML file in the trajectory directory"""
        yaml.add_representer(str, multiline_representer)
        args_path = traj_dir / "args.yaml"
        with args_path.open("w") as f:
            yaml.dump(self.args.asdict(), f)


def get_args(args: list[str] | None = None) -> DecoupledScriptArguments:
    """Parse command line arguments and return a DecoupledScriptArguments object.

    Args:
        args: Optional list of arguments to parse. If not provided, uses sys.argv.
    """
    defaults = DecoupledScriptArguments(
        environment=EnvironmentArguments(
            task_config_path="tasks/regressionKaggleHousePrice.yaml",
            max_steps=10,
            seed=42,
            container_type="docker",
            verbose=True,
        ),
        agent=AgentArguments(
            model=ModelArguments(
                model_name="litellm:gpt-4o",
                total_cost_limit=0.0,
                per_instance_cost_limit=3.0,
                temperature=0.0,
                top_p=0.95,
            ),
            agent_config_path=CONFIG_DIR / "agents" / "default.yaml",
        ),
        supervisor=AgentArguments(
            model=ModelArguments(
                model_name="litellm:gpt-4o-mini",
                total_cost_limit=0.0,
                per_instance_cost_limit=3.0,
                temperature=0.0,
                top_p=0.95,
            ),
            agent_config_path=CONFIG_DIR / "agents" / "supervisor.yaml",
        ),
    )
    yaml.add_representer(str, multiline_representer)

    new_args = parse(
        DecoupledScriptArguments,
        default=defaults,
        add_config_path_arg=False,
        args=args,
        formatter_class=RichHelpFormatter,
        description=Markdown(__doc__ or ""),
    )

    return new_args


def main(args: DecoupledScriptArguments) -> None:
    """Main entry point"""
    try:
        main_instance = Main(args)
        asyncio.run(main_instance.main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Failed to run: {e}")
        if args.raise_exceptions:
            raise
        traceback.print_exc()


if __name__ == "__main__":
    load_environment_variables()
    args = get_args()
    main(args) 