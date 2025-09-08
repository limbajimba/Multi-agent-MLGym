"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Decoupled supervisor run script for the MLGym framework.
This script demonstrates a decoupled workflow where:
1. Agent proposes actions without executing them
2. Supervisor reviews and approves/rejects actions
3. Environment executes only approved actions

Adapted from simple_run.py
"""

from pathlib import Path
import datetime
import logging
import os
from getpass import getuser

import gymnasium as gym
import yaml

from mlgym import CONFIG_DIR
from mlgym.agent.decoupled import DecoupledAgent, AgentArguments
from mlgym.agent.supervisor import SupervisorAgent
from mlgym.backend.base import ModelArguments
from mlgym.environment.env import EnvironmentArguments, MLGymEnv
from mlgym.environment.registration import register_task
from mlgym.utils.config import load_environment_variables
from mlgym.utils.extras import multiline_representer
from mlgym.utils.log import add_file_handler, get_logger

def run_decoupled_supervisor():
    """Run a decoupled supervisor workflow"""
    
    logger = get_logger("mlgym-decoupled-supervisor")
    logging.getLogger("simple_parsing").setLevel(logging.WARNING)
    logger.info(f"🍟 DOCKER_HOST: {os.environ.get('DOCKER_HOST')}")

    # Environment setup
    env_args = EnvironmentArguments(
        task_config_path="tasks/regressionKaggleHousePrice.yaml",
        max_steps=10,  # Small number for testing
        seed=42,
        container_type="podman",
        verbose=True,
        container_name="mlgym_decoupled_supervisor_container",
    )

    # Register environment
    register_task(env_args)

    # Create environment
    env: MLGymEnv = gym.make(f"mlgym/{env_args.task.id}", devices=["cpu"]).unwrapped
    init_observation, _ = env.reset()
    init_observation = init_observation["observation"]

    # Create decoupled agent
    agent_args = AgentArguments(
        model=ModelArguments(
            model_name="litellm:gpt-4o-mini",
            total_cost_limit=0.0,
            per_instance_cost_limit=3.0,
            temperature=0.0,
            top_p=0.95,
        ),
        agent_config_path=CONFIG_DIR / "agents" / "default.yaml",
    )

    # Create supervisor agent
    supervisor_args = AgentArguments(
        model=ModelArguments(
            model_name="litellm:gpt-4o-mini",
            total_cost_limit=0.0,
            per_instance_cost_limit=3.0,
            temperature=0.0,
            top_p=0.95,
        ),
        agent_config_path=CONFIG_DIR / "agents" / "supervisor.yaml",
    )

    agent = DecoupledAgent("decoupled_agent", agent_args)
    supervisor = SupervisorAgent("supervisor", supervisor_args)

    # Create run directory
    main_run_name = f"decoupled_supervisor_{env_args.task.id}_{datetime.datetime.now().strftime('%y%m%d%H%M%S')}"
    main_run_dir = Path("trajectories") / Path(getuser()) / main_run_name
    main_run_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    log_path = main_run_dir / f"decoupled-supervisor-run-{timestamp}.log"
    logger.info("Logging to %s", log_path)
    add_file_handler(log_path, ["mlgym-decoupled-supervisor", "MLGym", "decoupled-agent", "supervisor"])

    # Save arguments
    yaml.add_representer(str, multiline_representer)
    args_path = main_run_dir / "args.yaml"
    with args_path.open("w") as f:
        yaml.dump({
            "env": env_args.asdict(),
            "agent": agent_args.asdict(),
            "supervisor": supervisor_args.asdict()
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
        
        observation = init_observation
        step = 0
        max_steps = 10
        done = False
        
        while step < max_steps:
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
            
            # Check if done
            if done:
                logger.info("Environment marked as done")
                break
                
        logger.info("Decoupled supervisor workflow completed")
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        raise
    finally:
        env.close()
        logger.info("Environment closed.")

if __name__ == "__main__":
    load_environment_variables()
    run_decoupled_supervisor() 