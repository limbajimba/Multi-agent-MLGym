#!/usr/bin/env python3
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Decoupled supervisor workflow for MLGym.
This script implements a decoupled agent-supervisor-environment workflow with periodic check-ins.
"""

import datetime
import logging
import os
from dataclasses import dataclass
from getpass import getuser
from pathlib import Path

import gymnasium as gym
import yaml

from mlgym import CONFIG_DIR
from mlgym.agent.decoupled import DecoupledAgent
from mlgym.agent.supervisor import SupervisorAgent
from mlgym.agent.base import AgentArguments
from mlgym.backend.base import ModelArguments
from mlgym.environment.env import EnvironmentArguments, MLGymEnv
from mlgym.environment.registration import register_task
from mlgym.utils.config import load_environment_variables
from mlgym.utils.extras import multiline_representer
from mlgym.utils.log import add_file_handler, get_logger


@dataclass
class DecoupledScriptArguments:
    """Arguments for the decoupled supervisor workflow."""
    
    environment: EnvironmentArguments
    agent: AgentArguments
    supervisor: AgentArguments
    check_in_interval: int = 10  # Check-in every N steps
    raise_exceptions: bool = False


def run_decoupled_supervisor(args: DecoupledScriptArguments):
    """Run the decoupled supervisor workflow with periodic check-ins."""
    
    # Setup logging following MLGym pattern
    logger = get_logger("mlgym-decoupled-simple")
    logging.getLogger("simple_parsing").setLevel(logging.WARNING)
    logger.info(f"🍟 DOCKER_HOST: {os.environ.get('DOCKER_HOST')}")
    logger.info("Starting decoupled supervisor workflow with periodic check-ins")
    
    # Register and create environment
    register_task(args.environment)
    env: MLGymEnv = gym.make(f"mlgym/{args.environment.task.id}", devices=["cpu"]).unwrapped  # type: ignore
    
    # Initialize environment
    init_observation, _ = env.reset()
    if isinstance(init_observation, dict) and "observation" in init_observation:
        init_observation = init_observation["observation"]
    
    # Create trajectory directory following MLGym pattern
    timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    traj_dir = Path("trajectories") / getuser() / f"decoupled_simple_{args.environment.task.id}_{timestamp}"
    traj_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file following MLGym pattern
    log_path = traj_dir / f"decoupled-simple-run-{timestamp}.log"
    logger.info(f"Logging to {log_path}")
    add_file_handler(log_path, [
        "mlgym-decoupled-simple", "MLGym", "decoupled_agent", "supervisor", 
        "api_models", "env_utils", "MLGymEnv"
    ])
    
    # Save arguments following MLGym pattern
    yaml.add_representer(str, multiline_representer)
    args_path = traj_dir / "args.yaml"
    with args_path.open("w") as f:
        yaml.dump({
            "env": args.environment.asdict(),
            "agent": args.agent.asdict(),
            "supervisor": args.supervisor.asdict(),
            "check_in_interval": args.check_in_interval
        }, f)
    
    # Initialize agents
    agent = DecoupledAgent("decoupled_agent", args.agent)
    supervisor = SupervisorAgent("supervisor", args.supervisor)
    
    # Setup agents following MLGym pattern
    agent.setup(env.task.args)
    agent._env = env
    agent.init_environment_vars(env)
    
    supervisor.setup(env.task.args)
    supervisor._env = env
    supervisor.init_environment_vars(env)
    
    # Set check-in interval
    supervisor.set_check_in_interval(args.check_in_interval)
    
    # Initialize task context from environment
    task_context = {}
    if hasattr(env.task_args, 'description'):
        task_context["description"] = env.task_args.description
    if hasattr(env.task_args, 'baseline_scores'):
        # Handle both list and dictionary formats for baseline scores
        baseline_scores = env.task_args.baseline_scores
        if isinstance(baseline_scores, list):
            # Convert list of dicts to single dict
            task_context["baseline_scores"] = {}
            for score_dict in baseline_scores:
                if isinstance(score_dict, dict):
                    task_context["baseline_scores"].update(score_dict)
        else:
            task_context["baseline_scores"] = baseline_scores
    
    supervisor.update_task_context(task_context)
    
    # Get submit command from agent's tools configuration
    submit_command = agent.tools.submit_command if hasattr(agent.tools, 'submit_command') else "submit"
    
    # Log task information
    logger.info(f"▶️  Beginning task {args.environment.task.id}")
    if task_context.get("baseline_scores"):
        logger.info(f"🎯 Target: Beat baseline score")
        baseline_scores = task_context["baseline_scores"]
        if isinstance(baseline_scores, dict):
            for metric, value in baseline_scores.items():
                logger.info(f"   Baseline {metric}: {value}")
        else:
            logger.info(f"   Baseline scores: {baseline_scores}")
    logger.info(f"📝 Submit command: {submit_command}")
    logger.info(f"🔍 Check-in interval: every {args.check_in_interval} steps")
    
    # Run the decoupled workflow with periodic check-ins
    try:
        observation = init_observation
        done = False
        submission_made = False
        
        # Use environment's step counter and max_steps (MLGym pattern)
        while env.current_step < env.max_steps and not done:
            logger.info(f"Step {env.current_step + 1}/{env.max_steps}")
            
            # Step 1: Agent proposes and executes action
            logger.info("Agent proposing action...")
            thought, action, output = agent.propose_action(observation)
            logger.info(f"Agent proposed: {action[:100]}...")
            
            # Execute action directly (no approval needed)
            try:
                env_result, _, done, _info = env.step(action)
                
                # Handle different return formats
                if isinstance(env_result, dict) and "observation" in env_result:
                    observation = env_result["observation"]
                else:
                    observation = str(env_result)
                
                # Track agent activity for supervisor
                supervisor.add_agent_activity(
                    agent.name, action, observation, env.current_step
                )
                
                # Check for submission
                if action.strip() == submit_command:
                    logger.info("🎉 Agent submitted solution!")
                    submission_made = True
                    break
                    
            except Exception as e:
                logger.error(f"Environment step failed: {e}")
                if args.raise_exceptions:
                    raise
                observation = f"ERROR: {str(e)}"
                supervisor.add_agent_activity(
                    agent.name, action, observation, env.current_step
                )
            
            # Step 2: Periodic supervisor check-in
            if supervisor.should_check_in(env.current_step):
                logger.info(f"🔍 SUPERVISOR CHECK-IN at step {env.current_step}")
                
                # Perform check-in
                assessment = supervisor.perform_check_in(agent.name, env.current_step)
                
                # Log assessment
                logger.info(f"📊 Check-in Assessment:")
                logger.info(f"   Progress: {assessment['progress']}")
                logger.info(f"   Strategy: {assessment['strategy']}")
                logger.info(f"   Urgency: {assessment['urgency']}")
                
                if assessment['guidance']:
                    logger.info(f"   Guidance: {assessment['guidance'][:200]}...")
                
                if assessment['next_steps']:
                    logger.info(f"   Next Steps: {assessment['next_steps'][:200]}...")
                
                # If high urgency, provide guidance as observation
                if assessment['urgency'] == 'High':
                    guidance_observation = f"""
SUPERVISOR GUIDANCE (URGENT):
{assessment['guidance']}

RECOMMENDED NEXT STEPS:
{assessment['next_steps']}

Please consider this guidance in your next actions.
"""
                    # Add guidance to agent's history
                    agent.receive_feedback(guidance_observation)
        
        # Log final results
        logger.info("Decoupled supervisor workflow completed")
        
        # Get supervision summary
        supervision_summary = supervisor.get_supervision_summary()
        agent_progress = agent.get_progress_summary()
        
        logger.info(f"📊 SUPERVISION SUMMARY:")
        logger.info(f"   Total check-ins: {supervision_summary['total_check_ins']}")
        logger.info(f"   Total agent actions: {supervision_summary['total_agent_actions']}")
        logger.info(f"   Check-in interval: {supervision_summary['check_in_interval']}")
        logger.info(f"   High urgency issues: {supervision_summary['high_urgency_count']}")
        
        logger.info(f"🤖 AGENT PROGRESS:")
        logger.info(f"   Actions proposed: {agent_progress['total_actions_proposed']}")
        logger.info(f"   Progress: {agent_progress['progress_percentage']:.1f}%")
        logger.info(f"   Trajectory length: {agent_progress['trajectory_length']}")
        logger.info(f"   History length: {agent_progress['history_length']}")
        
        if submission_made:
            logger.info("✅ SUCCESS: Agent submitted solution!")
        else:
            logger.warning("❌ Agent did not submit solution within step limit")
            
        # Log recent assessments for analysis
        if supervision_summary['recent_assessments']:
            logger.info(f"   Recent assessments:")
            for assessment in supervision_summary['recent_assessments']:
                logger.info(f"     Step {assessment['step']}: {assessment['progress']} progress, {assessment['strategy']} strategy")
        
        # Save trajectories following MLGym pattern
        agent.traj_dir = traj_dir / "agent"
        agent.traj_dir.mkdir(parents=True, exist_ok=True)
        agent.save_trajectory()
        agent.save_results()
        
        supervisor.traj_dir = traj_dir / "supervisor"
        supervisor.traj_dir.mkdir(parents=True, exist_ok=True)
        supervisor.save_trajectory()
        supervisor.save_results()
        
        logger.info(f"📁 Trajectories saved to {traj_dir}")
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        if args.raise_exceptions:
            raise
    finally:
        env.close()
        logger.info("Environment closed.")


def get_args() -> DecoupledScriptArguments:
    """Get default arguments for the decoupled workflow."""
    
    # Set up default arguments following MLGym pattern
    env_args = EnvironmentArguments(
        task_config_path="tasks/battleOfSexes.yaml",
        max_steps=50,
        seed=42,
        container_type="podman",
        verbose=True,
        container_name="mlgym_decoupled_simple_container",
    )
    
    agent_args = AgentArguments(
        model=ModelArguments(
            model_name="litellm:gpt-4o-mini",
            total_cost_limit=0.0,
            per_instance_cost_limit=4.0,
            temperature=1.0,
            top_p=0.95,
        ),
        agent_config_path=CONFIG_DIR / "agents" / "default.yaml",
    )
    
    supervisor_args = AgentArguments(
        model=ModelArguments(
            model_name="litellm:gpt-4o-mini",
            total_cost_limit=0.0,
            per_instance_cost_limit=4.0,
            temperature=1.0,
            top_p=0.95,
        ),
        agent_config_path=CONFIG_DIR / "agents" / "supervisor.yaml",
    )
    
    return DecoupledScriptArguments(
        environment=env_args,
        agent=agent_args,
        supervisor=supervisor_args,
        check_in_interval=10,  # Check-in every 10 steps
        raise_exceptions=False
    )


def main() -> None:
    """Main entry point following MLGym pattern."""
    load_environment_variables()
    args = get_args()
    run_decoupled_supervisor(args)


if __name__ == "__main__":
    main() 