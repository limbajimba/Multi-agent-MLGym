#!/usr/bin/env python3
"""
Single Sequential Agents Experiment Runner

Run ONE experiment at a time. Change the parameters below and run it.
Much simpler and more reliable than batch experiments.

Usage:
    python single_experiment.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict

# Add parent directory to path for MLGym imports
sys.path.append(str(Path(__file__).parent))

# MLGym imports
from mlgym import CONFIG_DIR
from mlgym.agent.base import AgentArguments, BaseAgent
from mlgym.backend.base import ModelArguments
from mlgym.environment.env import EnvironmentArguments, MLGymEnv
from mlgym.environment.registration import register_task
from mlgym.utils.config import load_environment_variables
from mlgym.utils.log import get_logger

# Import MLGym's multi-run functionality
from multi_run import ScriptArguments, Main

# Set up logging
logger = get_logger("single-experiment")

def run_single_experiment() -> Dict:
    """
    Run a single experiment with the parameters defined below.
    """
    print("🔬 Single Sequential Agents Experiment")
    print("=" * 50)
    
    # Load environment variables
    load_environment_variables()
    
    # Register the task
    register_task(EnvironmentArguments(task_config_path=EXPERIMENT_CONFIG['task']))
    
    print(f"🤖 Model: {EXPERIMENT_CONFIG['model']}")
    print(f"👥 Agents: {EXPERIMENT_CONFIG['num_agents']}")
    print(f"📊 Task: {EXPERIMENT_CONFIG['task']}")
    print(f"⏱️  Max steps per agent: {EXPERIMENT_CONFIG['max_steps']}")
    print(f"💰 Cost limit per agent: ${EXPERIMENT_CONFIG['cost_limit']}")
    print(f"🌡️  Temperature: {EXPERIMENT_CONFIG['temperature']}")
    print(f"🎯 Top-p: {EXPERIMENT_CONFIG['top_p']}")
    print(f"🎲 Seed: {EXPERIMENT_CONFIG['seed']}")
    print()
    
    # Create experiment arguments using MLGym's standard approach
    args = ScriptArguments(
        environment=EnvironmentArguments(
            task_config_path=EXPERIMENT_CONFIG['task'],
            max_steps=EXPERIMENT_CONFIG['max_steps'],
            seed=EXPERIMENT_CONFIG['seed'],
            container_type="podman",
            verbose=True,
        ),
        agent=AgentArguments(
            model=ModelArguments(
                model_name=EXPERIMENT_CONFIG['model'],
                total_cost_limit=EXPERIMENT_CONFIG['total_cost_limit'],
                per_instance_cost_limit=EXPERIMENT_CONFIG['cost_limit'],
                temperature=EXPERIMENT_CONFIG['temperature'],
                top_p=EXPERIMENT_CONFIG['top_p'],
            ),
            agent_config_path=CONFIG_DIR / "agents" / "default.yaml",
        ),
        num_agents=EXPERIMENT_CONFIG['num_agents'],
        per_agent_max_steps=EXPERIMENT_CONFIG['max_steps'],
        suffix=EXPERIMENT_CONFIG['suffix']
    )
    
    # Run the experiment using MLGym's Main class
    main = Main(args)
    
    try:
        print("🚀 Starting experiment...")
        # This runs the sequential agents and saves all data automatically
        asyncio.run(main.main())
        
        # Extract results from MLGym's saved files
        results = extract_mlgym_results(args, EXPERIMENT_CONFIG['num_agents'])
        results['success'] = True
        results['error'] = None
        
        print("✅ Experiment completed successfully!")
        print(f"   Total cost: ${results['total_cost']:.2f}")
        print(f"   Final score: {results.get('final_score', 'N/A')}")
        print(f"   Total tokens: {results['total_tokens']:,}")
        print(f"   API calls: {results['api_calls']}")
        print(f"   Run name: {results['run_name']}")
        
        # Save results to a simple file
        save_results(results)
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        results = {
            'success': False,
            'error': str(e),
            'model': EXPERIMENT_CONFIG['model'],
            'num_agents': EXPERIMENT_CONFIG['num_agents'],
            'total_cost': 0.0,
            'final_score': None
        }
        save_results(results)
    
    return results

def extract_mlgym_results(args: ScriptArguments, num_agents: int) -> Dict:
    """
    Extract results from MLGym's automatically saved files.
    """
    # Get the run name that MLGym created
    run_name = args.run_name()
    traj_base = Path("trajectories") / os.getenv('USER', 'unknown')
    
    # Find the trajectory directory
    traj_dirs = list(traj_base.glob(f"*{run_name}*"))
    if not traj_dirs:
        raise FileNotFoundError(f"No trajectory directory found for {run_name}")
    
    # Get the last agent's results (final performance)
    last_agent_dir = traj_base / f"{run_name}_agent_{num_agents - 1}"
    if not last_agent_dir.exists():
        raise FileNotFoundError(f"Last agent directory not found: {last_agent_dir}")
    
    # Load MLGym's automatically saved results.json
    results_file = last_agent_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file) as f:
        mlgym_results = json.load(f)
    
    # Load the trajectory to get model stats
    traj_file = last_agent_dir / f"{run_name.split('__')[1]}.traj"
    if not traj_file.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_file}")
    
    with open(traj_file) as f:
        trajectory_data = json.load(f)
    
    # Extract the data we need
    model_stats = trajectory_data.get('info', {}).get('model_stats', {})
    
    return {
        'model': args.agent.model.model_name,
        'num_agents': num_agents,
        'total_cost': model_stats.get('total_cost', 0.0),
        'total_tokens': model_stats.get('tokens_sent', 0) + model_stats.get('tokens_received', 0),
        'api_calls': model_stats.get('api_calls', 0),
        'final_score': mlgym_results.get('agent', [{}])[-1] if mlgym_results.get('agent') else None,
        'baseline_score': mlgym_results.get('baseline'),
        'run_name': run_name,
        'trajectory_dir': str(last_agent_dir)
    }

def save_results(results: Dict, output_file: str = "single_experiment_result.json"):
    """Save experiment results to a JSON file."""
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"📊 Results saved to: {output_path}")

if __name__ == "__main__":
    # ============================================================================
    # EXPERIMENT CONFIGURATION - CHANGE THESE PARAMETERS FOR EACH RUN
    # ============================================================================
    
    EXPERIMENT_CONFIG = {
        # Model to test
        'model': "litellm:gpt-4o-mini",
        # 'model': "litellm:gpt-4o",
        # 'model': "litellm:claude-3-5-sonnet",
        
        # Number of agents to test
        'num_agents': 1,
        # 'num_agents': 2,
        # 'num_agents': 3,
        
        # Task configuration
        'task': "tasks/regressionKaggleHousePrice.yaml",
        # 'task': "tasks/battleOfSexes.yaml",
        # 'task': "tasks/imageClassificationCifar10.yaml",
        
        # Experiment parameters
        'max_steps': 30,        # Steps per agent
        'cost_limit': 2.0,      # Cost limit per agent
        'total_cost_limit': 5.0, # Total cost limit for all agents
        'temperature': 0.0,     # Model temperature
        'top_p': 0.95,         # Model top_p
        'seed': 42,            # Random seed
        
        # Suffix for run name (to distinguish different experiments)
        'suffix': "test_run"
    }
    
    # Run the single experiment
    run_single_experiment()




