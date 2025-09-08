#!/usr/bin/env python3
"""
Simple Sequential Agents Experiment Runner

This script runs simple experiments with sequential agents using MLGym's built-in functionality.
Just change the parameters at the bottom and run it.

Usage:
    python simple_sequential_experiment.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

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
logger = get_logger("simple-sequential-experiment")

def run_single_experiment(
    model_name: str,
    num_agents: int, 
    task: str,
    max_steps: int = 50,
    cost_limit: float = 3.0,
    temperature: float = 0.0,
    top_p: float = 0.95,
    seed: int = 42
) -> Dict:
    """
    Run a single experiment with sequential agents.
    
    Returns:
        Dict with experiment results including costs, scores, and metadata
    """
    logger.info(f"🚀 Running experiment: {model_name} with {num_agents} agents")
    
    # Create experiment arguments using MLGym's standard approach
    args = ScriptArguments(
        environment=EnvironmentArguments(
            task_config_path=task,
            max_steps=max_steps,
            seed=seed,
            container_type="podman",
            verbose=True,
        ),
        agent=AgentArguments(
            model=ModelArguments(
                model_name=model_name,
                total_cost_limit=5.0,
                per_instance_cost_limit=cost_limit,
                temperature=temperature,
                top_p=top_p,
            ),
            agent_config_path=CONFIG_DIR / "agents" / "default.yaml",
        ),
        num_agents=num_agents,
        per_agent_max_steps=max_steps,
        suffix=f"seq_{num_agents}agents"
    )
    
    # Run the experiment using MLGym's Main class
    main = Main(args)
    
    try:
        # This runs the sequential agents and saves all data automatically
        asyncio.run(main.main())
        
        # Extract results from MLGym's saved files
        results = extract_mlgym_results(args, num_agents)
        results['success'] = True
        results['error'] = None
        
        logger.info(f"✅ Experiment completed successfully")
        logger.info(f"   Total cost: ${results['total_cost']:.2f}")
        logger.info(f"   Final score: {results.get('final_score', 'N/A')}")
        
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        results = {
            'success': False,
            'error': str(e),
            'model': model_name,
            'num_agents': num_agents,
            'total_cost': 0.0,
            'final_score': None
        }
    
    return results

def extract_mlgym_results(args: ScriptArguments, num_agents: int) -> Dict:
    """
    Extract results from MLGym's automatically saved files.
    This is much simpler than the original complex extraction.
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

def save_experiment_results(results: List[Dict], output_file: str = "experiment_results.json"):
    """Save experiment results to a JSON file."""
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"📊 Results saved to: {output_path}")

def main():
    """Main function to run experiments."""
    
    print("🔬 Simple Sequential Agents Experiment")
    print("=" * 50)
    
    # Load environment variables
    load_environment_variables()
    
    # Register the task
    register_task(EnvironmentArguments(task_config_path=EXPERIMENT_CONFIG['task']))
    
    # Run experiments
    all_results = []
    
    for model in EXPERIMENT_CONFIG['models']:
        for num_agents in EXPERIMENT_CONFIG['agent_counts']:
            for run_id in range(EXPERIMENT_CONFIG['num_runs']):
                print(f"\n🤖 Model: {model}, Agents: {num_agents}, Run: {run_id + 1}")
                
                result = run_single_experiment(
                    model_name=model,
                    num_agents=num_agents,
                    task=EXPERIMENT_CONFIG['task'],
                    max_steps=EXPERIMENT_CONFIG['max_steps'],
                    cost_limit=EXPERIMENT_CONFIG['cost_limit'],
                    temperature=EXPERIMENT_CONFIG['temperature'],
                    top_p=EXPERIMENT_CONFIG['top_p'],
                    seed=EXPERIMENT_CONFIG['seed'] + run_id  # Different seed per run
                )
                
                result['run_id'] = run_id
                all_results.append(result)
                
                # Save results after each experiment
                save_experiment_results(all_results)
                
                if result['success']:
                    print(f"   ✅ Cost: ${result['total_cost']:.2f}, Score: {result.get('final_score', 'N/A')}")
                else:
                    print(f"   ❌ Failed: {result['error']}")
    
    print(f"\n🎉 All experiments completed!")
    print(f"📊 Total experiments: {len(all_results)}")
    print(f"✅ Successful: {len([r for r in all_results if r['success']])}")
    print(f"❌ Failed: {len([r for r in all_results if not r['success']])}")

if __name__ == "__main__":
    # ============================================================================
    # EXPERIMENT CONFIGURATION - CHANGE THESE PARAMETERS AS NEEDED
    # ============================================================================
    
    EXPERIMENT_CONFIG = {
        # Models to test
        'models': [
            "litellm:gpt-4o-mini",
            # "litellm:gpt-4o",
            # "litellm:claude-3-5-sonnet",
        ],
        
        # Number of agents to test
        'agent_counts': [1, 2, 3],
        # 'agent_counts': [1, 2, 3, 4, 5],  # Uncomment for more agents
        
        # Number of runs per condition
        'num_runs': 2,
        # 'num_runs': 5,  # Uncomment for more runs
        
        # Task configuration
        'task': "tasks/regressionKaggleHousePrice.yaml",
        
        # Experiment parameters
        'max_steps': 30,        # Steps per agent
        'cost_limit': 2.0,      # Cost limit per agent
        'temperature': 0.0,     # Model temperature
        'top_p': 0.95,         # Model top_p
        'seed': 42,            # Base seed (will be incremented per run)
    }
    
    # Run the experiments
    main()




