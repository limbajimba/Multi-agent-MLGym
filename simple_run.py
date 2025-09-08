from pathlib import Path
import datetime
import logging
import os
from getpass import getuser

import gymnasium as gym
import yaml

from mlgym import CONFIG_DIR
from mlgym.agent.base import AgentArguments, BaseAgent
from mlgym.backend.base import ModelArguments
from mlgym.environment.env import EnvironmentArguments, MLGymEnv
from mlgym.environment.registration import register_task
from mlgym.utils.config import load_environment_variables
from mlgym.utils.extras import multiline_representer
from mlgym.utils.log import add_file_handler, get_logger

# --- AGENT SEQUENCE PLACEHOLDER ---
# You can add different agent classes and their arguments here
AGENT_SEQUENCE = [
        (BaseAgent, AgentArguments(
        model=ModelArguments(
            model_name="litellm:gpt-4o-mini",
            total_cost_limit=0.0,
            per_instance_cost_limit=3.0,
            temperature=0.0,
            top_p=0.95,
        ),
        agent_config_path=CONFIG_DIR / "agents" / "default.yaml",
    )),
        (BaseAgent, AgentArguments(
        model=ModelArguments(
            model_name="litellm:gpt-4o-mini",
            total_cost_limit=0.0,
            per_instance_cost_limit=3.0,
            temperature=0.0,
            top_p=0.95,
        ),
        agent_config_path=CONFIG_DIR / "agents" / "default.yaml",
    )),
        (BaseAgent, AgentArguments(
        model=ModelArguments(
            model_name="litellm:gpt-4o-mini",
            total_cost_limit=0.0,
            per_instance_cost_limit=3.0,
            temperature=0.0,
            top_p=0.95,
        ),
        agent_config_path=CONFIG_DIR / "agents" / "default.yaml",
    ))
]

PERSISTENT_CONTAINER_NAME = "mlgym_shared_container"

def run_sequential_agents():
    logger = get_logger("mlgym-simple-run")
    logging.getLogger("simple_parsing").setLevel(logging.WARNING)
    logger.info(f"🍟 DOCKER_HOST: {os.environ.get('DOCKER_HOST')}")

    # Minimal environment arguments with persistent container
    env_args = EnvironmentArguments(
        task_config_path="tasks/regressionKaggleHousePrice.yaml",
        max_steps=50,
        seed=42,
        container_type="podman",
        verbose=True,
        container_name=PERSISTENT_CONTAINER_NAME,
    )

    # Register environment
    register_task(env_args)

    # Create environment (persistent container)
    env: MLGymEnv = gym.make(f"mlgym/{env_args.task.id}", devices=["cpu"]).unwrapped  # type: ignore
    init_observation, _ = env.reset()
    init_observation = init_observation["observation"]

    # Create a main run directory for this sequence
    main_run_name = f"{env_args.task.id}_{datetime.datetime.now().strftime('%y%m%d%H%M%S')}"
    main_run_dir = Path("trajectories") / Path(getuser()) / main_run_name
    main_run_dir.mkdir(parents=True, exist_ok=True)

    # Run each agent in sequence, reusing the same environment/container
    for idx, (AgentClass, agent_args) in enumerate(AGENT_SEQUENCE):
        agent = AgentClass(f"agent_{idx}", agent_args)
        agent_dir = main_run_dir / f"agent_{idx}_{agent_args.model.model_name.replace(':', '-')}"
        agent_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        log_path = agent_dir / f"run-{timestamp}.log"
        logger.info("Logging to %s", log_path)
        add_file_handler(log_path, ["mlgym-simple-run", "MLGym", agent.name])

        # Save arguments
        yaml.add_representer(str, multiline_representer)
        args_path = agent_dir / "args.yaml"
        with args_path.open("w") as f:
            yaml.dump({"env": env_args.asdict(), "agent": agent_args.asdict()}, f)

        # Reset max_steps for each agent
        env.max_steps = 10
        env.current_step = 0

        # Run agent
        try:
            agent.run(
                env=env,
                observation=init_observation if idx == 0 else None,
                traj_dir=agent_dir,
                return_type="info_trajectory",
            )
            logger.info("Agent %s finished", agent.name)
        except Exception as e:
            logger.warning(f"Agent crashed: {e}")
            env.reset_container()
        # Do not close the environment here; keep container alive for next agent

    # Close environment after all agents are done
    env.close()
    logger.info("Environment closed.")

if __name__ == "__main__":
    load_environment_variables()
    run_sequential_agents() 