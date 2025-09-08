
from __future__ import annotations

import asyncio
import datetime
import logging
import os
import traceback
from dataclasses import dataclass
from getpass import getuser
from pathlib import Path
from typing import List

import gymnasium as gym
import yaml
from simple_parsing import parse
from simple_parsing.helpers.fields import field
from simple_parsing.helpers.flatten import FlattenedAccess
from simple_parsing.helpers.serialization.serializable import FrozenSerializable

from mlgym import CONFIG_DIR
from mlgym.agent.base import AgentArguments, BaseAgent
from mlgym.agent.supervisor_aware import SupervisorAwareAgent
from mlgym.backend.base import ModelArguments
from mlgym.environment.env import EnvironmentArguments, MLGymEnv
from mlgym.environment.registration import register_task
from mlgym.utils.config import load_environment_variables
from mlgym.utils.extras import get_devices, multiline_representer
from mlgym.utils.log import add_file_handler, get_logger

try:
    import rich  # noqa: F401
except ModuleNotFoundError as e:
    raise RuntimeError(
        "You probably either forgot to install the dependencies "
        "or forgot to activate your conda or virtual environment."
    ) from e

from rich.markdown import Markdown

try:
    from rich_argparse import RichHelpFormatter
except ImportError as e:
    raise ImportError(
        "Please install the rich_argparse package with `pip install rich_argparse`."
    ) from e

__doc__ = """Run inference."""

logger = get_logger("mlgym-run")
logging.getLogger("simple_parsing").setLevel(logging.WARNING)
logger.info(f"🍟 DOCKER_HOST: {os.environ.get('DOCKER_HOST')}")


@dataclass(frozen=True)
class ScriptArguments(FlattenedAccess, FrozenSerializable):
    """Configure the control flow of the run.py script"""

    environment: EnvironmentArguments
    agent: AgentArguments
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
    # Maximum number of steps per agent, if None, will use the environment's max_steps
    per_agent_max_steps: int | None = None
    
    def run_name(self) -> str:
        """Generate a unique name for this run based on the arguments."""
        model_name = self.agent.model.model_name.replace(":", "-")
        assert self.environment.task is not None
        task_id = self.environment.task.id
        assert self.agent.agent_config_path is not None
        config_stem = Path(self.agent.agent_config_path).stem

        temp = self.agent.model.temperature
        top_p = self.agent.model.top_p

        per_instance_cost_limit = self.agent.model.per_instance_cost_limit
        # install_env = self.environment.install_environment
        install_env = False

        return (
            f"FINAL_EXPERIMENT_{model_name}__{task_id}__{config_stem}__t-{temp:.2f}__p-{top_p:.2f}"
            + f"__c-{per_instance_cost_limit:.2f}__install-{int(install_env)}"
            + (f"__{self.suffix}" if self.suffix else "")
        )

    def register_envs(self) -> None:
        # Assume we are not using benchmark for now. So we only need to register the env for the task specified in task_args.
        register_task(self.environment)

    def __post_init__(self) -> None:
        # check whether benchmark or env_args.task_args is set
        if self.benchmark is not None and self.environment.task is not None:
            msg = "Please set either benchmark or task_args parameter in EnvironmentArguments"
            raise ValueError(msg)

        self.register_envs()


class Main:
    def __init__(self, args: ScriptArguments) -> None:
        self.args = args

        # assign a unique container_name when the user didn’t specify one, so the container becomes persistent
        # and can be reused by subsequent agents
        if self.args.environment.container_name is None:
            object.__setattr__(                                  
                self.args.environment,
                "container_name",
                self.args.run_name() + f"_{datetime.datetime.now():%y%m%d%H%M%S}",
            )


    def _prepare_devices(self) -> list[list[str]]:
        if self.args.gpus_per_agent > 0:
            _devices = (
                get_devices() if len(self.args.gpus) == 0 else self.args.gpus
            )
            devices: List[str] = [str(x) for x in _devices]
            if self.args.gpus_per_agent * self.args.num_agents > len(devices):
                raise RuntimeError(
                    f"Not enough GPUs: need "
                    f"{self.args.gpus_per_agent * self.args.num_agents}, "
                    f"have {len(devices)}"
                )
            agent_devices = [
                devices[
                    self.args.gpus_per_agent * i : self.args.gpus_per_agent * (i + 1)
                ]
                for i in range(self.args.num_agents)
            ]
        else:
            agent_devices = [[f"cpu_{i}"] for i in range(self.args.num_agents)]
        return agent_devices

   
    def _single_agent_run(
        self,
        agent: BaseAgent,
        #agent: SupervisorAwareAgent,
        env: MLGymEnv,
        observation: str | None,
        traj_dir: Path,
    ) -> None:
        """Helper to execute one agent inside existing env."""

        env.current_step = 0
        if self.args.per_agent_max_steps is not None:
            env.max_steps = self.args.per_agent_max_steps
            
        traj_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        log_path = traj_dir / f"run-{timestamp}.log"
        logger.info("Logging to %s", log_path)

        add_file_handler(
            log_path,
            ["mlgym-run", "MLGym", agent.name, "api_models", "env_utils", "MLGymEnv"],
        )
        if self.args.print_config:
            logger.info(f"Arguments: {self.args.dumps_yaml()}")

        self._save_arguments(traj_dir)

        try:
            agent.run(
                env=env,
                observation=observation,
                traj_dir=traj_dir,
                return_type="info_trajectory",
            )
            logger.info("Agent %s finished", agent.name)
        except Exception as e:
            logger.warning(traceback.format_exc())
            if self.args.raise_exceptions:
                env.close()
                raise
            logger.warning("❌ Agent %s crashed: %s", agent.name, e)
            env.reset_container()  # non-destructive reset


    async def main(self) -> None:
        agent_devices = self._prepare_devices()

        assert self.args.environment.task is not None
        # create 1 environment that will stay alive for all agents 
        env: MLGymEnv = gym.make(f"mlgym/{self.args.environment.task.id}", devices=agent_devices[0]).unwrapped  # type: ignore

        # only call env.reset() once, so workspace persists afterwards
        init_observation, _ = env.reset()
        init_observation = init_observation["observation"]

        # sequential agents build on the previous ones
        for idx in range(self.args.num_agents):
            agent = BaseAgent(f"agent_{idx}", self.args.agent)
            traj_dir = (
                Path("trajectories")
                / Path(getuser())
                / (self.args.run_name() + f"_agent_{idx}")
            )
           
            await asyncio.to_thread(
                self._single_agent_run,
                agent,
                env,
                init_observation if idx == 0 else None,
                traj_dir,
            )

        # close
        env.close()
        logger.info("Environment closed (container persisted/paused).")

    def _save_arguments(self, traj_dir: Path) -> None:
        """Save the arguments to a yaml file to the run's trajectory directory."""
        log_path = traj_dir / "args.yaml"

        if log_path.exists():
            try:
                other_args = self.args.load_yaml(log_path)
                if self.args.dumps_yaml() != other_args.dumps_yaml():  # check yaml equality instead of object equality
                    logger.warning("**************************************************")
                    logger.warning("Found existing args.yaml with different arguments!")
                    logger.warning("**************************************************")
            except Exception as e:
                logger.warning(f"Failed to load existing args.yaml: {e}")

        with log_path.open("w") as f:
            self.args.dump_yaml(f)

def get_args(args: list[str] | None = None) -> ScriptArguments:
    defaults = ScriptArguments(
        environment=EnvironmentArguments(
            task_config_path="tasks/regressionKaggleHousePrice.yaml",
            max_steps=30,
            seed=42,
            container_type="podman",
            verbose=True,
        ),
        agent=AgentArguments(
            model=ModelArguments(
                model_name="litellm:gpt-4o",
                total_cost_limit=0.0,
                per_instance_cost_limit=3.0,
                temperature=1.0,
                top_p=0.95,
            ),
            agent_config_path=CONFIG_DIR / "agents" / "default.yaml",
        ),
        
    )
    yaml.add_representer(str, multiline_representer)

    return parse(
        ScriptArguments,
        default=defaults,
        add_config_path_arg=False,
        args=args,
        formatter_class=RichHelpFormatter,
        description=Markdown(__doc__ or ""),
    )


def main(args: ScriptArguments) -> None:
    asyncio.run(Main(args).main())


if __name__ == "__main__":
    load_environment_variables()
    main(get_args())