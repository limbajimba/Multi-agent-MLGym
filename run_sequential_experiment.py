#!/usr/bin/env python3
"""
Sequential Agents Academic Experiment Runner

Usage:
    python run_sequential_experiment.py

This script runs experiments to study how increasing the number of sequential agents
affects performance, cost, and success rates across different LLM models.
"""

import asyncio
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway

# Add parent directory to path for MLGym imports
sys.path.append(str(Path(__file__).parent.parent))

# MLGym imports
from mlgym import CONFIG_DIR
from mlgym.agent.base import AgentArguments, BaseAgent
from mlgym.backend.base import ModelArguments
from mlgym.environment.env import EnvironmentArguments, MLGymEnv
from mlgym.environment.registration import register_task
from mlgym.utils.config import load_environment_variables
from mlgym.utils.log import get_logger

# Import our modified multi_run components
from multi_run import ScriptArguments, Main

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger("academic-experiment")

class AcademicExperimentRunner:
    """Modified version of Main class for academic experiments with proper isolation"""
    
    def __init__(self, model_name: str, num_agents: int, run_id: int, seed: int = 42):
        self.model_name = model_name
        self.num_agents = num_agents
        self.run_id = run_id
        self.seed = seed
        
        # Create unique container name for this experiment
        # MLGym will use this for persistent container management
        timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        self.container_name = f"academic_exp_{model_name.replace(':', '-')}_{num_agents}agents_run{run_id}_{timestamp}"
        
        # Create experiment arguments
        self.args = ScriptArguments(
            environment=EnvironmentArguments(
                task_config_path=EXPERIMENT_CONFIG['task'],
                max_steps=EXPERIMENT_CONFIG['per_agent_max_steps'],
                seed=seed,
                container_type="podman",  # Use podman for containerization
                verbose=True,
                container_name=self.container_name  # Force unique container (persistence controlled by container_name)
            ),
            agent=AgentArguments(
                model=ModelArguments(
                    model_name=model_name,
                    total_cost_limit=0.0,
                    per_instance_cost_limit=EXPERIMENT_CONFIG['per_agent_cost_limit'],
                    temperature=0.0,
                    top_p=0.95,
                ),
                agent_config_path=CONFIG_DIR / "agents" / "default.yaml",
            ),
            num_agents=num_agents,
            per_agent_max_steps=EXPERIMENT_CONFIG['per_agent_max_steps'],
            suffix=f"academic_exp_run{run_id}"
        )
        
        # Initialize main runner
        self.main = Main(self.args)
        
    async def run_experiment(self) -> Dict:
        """Run single experiment and return results"""
        start_time = time.time()
        
        try:
            # Run the experiment
            await self.main.main()
            
            # Extract results
            results = self._extract_results()
            results['success'] = True
            results['error'] = None
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            results = {
                'success': False,
                'error': str(e),
                'model': self.model_name,
                'num_agents': self.num_agents,
                'run_id': self.run_id,
                'r2': 0.0,
                'rmse': float('inf'),
                'cost': 0.0,
                'time': time.time() - start_time
            }
        
        results['time'] = time.time() - start_time
        
        # Note: MLGym's Main.main() already handles container cleanup via env.close()
        # No manual cleanup needed - MLGym manages container lifecycle properly
        
        return results
    
    def _extract_results(self) -> Dict:
        """Extract comprehensive results including raw data and metadata"""
        # Find the trajectory directory
        run_name = self.args.run_name()
        traj_base = Path("trajectories") / os.getenv('USER', 'unknown')
        
        # Look for the most recent trajectory directory matching our run
        traj_dirs = list(traj_base.glob(f"*{run_name}*"))
        if not traj_dirs:
            raise FileNotFoundError(f"No trajectory directory found for {run_name}")
        
        # Get all agent directories for this run
        agent_dirs = []
        for i in range(self.num_agents):
            agent_dir = traj_base / f"{run_name}_agent_{i}"
            if agent_dir.exists():
                agent_dirs.append(agent_dir)
        
        if not agent_dirs:
            raise FileNotFoundError(f"No agent directories found for {run_name}")
        
        # Extract comprehensive results
        results = {
            'model': self.model_name,
            'num_agents': self.num_agents,
            'run_id': self.run_id,
            'container_name': self.container_name,
            'timestamp': datetime.datetime.now().isoformat(),
            'agents': [],
            'total_cost': 0.0,  # Will be populated by MLGym
            'total_tokens': 0,
            'total_api_calls': 0,
            'success': False,
            'error': None,
            'time': None
        }
        
        # Process each agent's results
        for i, agent_dir in enumerate(agent_dirs):
            agent_results = {
                'agent_id': i,
                'directory': str(agent_dir),
                'files': []
            }
            
            # List all files in agent directory
            for file_path in agent_dir.rglob('*'):
                if file_path.is_file():
                    agent_results['files'].append({
                        'name': file_path.name,
                        'path': str(file_path.relative_to(traj_base)),
                        'size': file_path.stat().st_size
                    })
            
            # Extract results.json if it exists
            results_file = agent_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    agent_data = json.load(f)
                
                # Extract detailed metrics
                agent_scores = agent_data.get('agent', [])
                if agent_scores:
                    best_score = agent_scores[-1] if agent_scores else {}
                    agent_results.update({
                        'r2': best_score.get('r2', 0.0),
                        'rmse': best_score.get('rmse', float('inf')),
                        'cost': agent_data.get('total_cost', 0.0),
                        'tokens_sent': agent_data.get('total_tokens_sent', 0),
                        'tokens_received': agent_data.get('total_tokens_received', 0),
                        'api_calls': agent_data.get('total_api_calls', 0)
                    })
                    
                    # Accumulate totals (MLGym already provides these)
                    results['total_tokens'] += agent_results.get('tokens_sent', 0) + agent_results.get('tokens_received', 0)
                    results['total_api_calls'] += agent_results.get('api_calls', 0)
            
            # Extract trajectory file if it exists
            traj_file = agent_dir / f"{run_name.split('__')[1]}.traj"
            if traj_file.exists():
                agent_results['trajectory_file'] = str(traj_file.relative_to(traj_base))
                agent_results['trajectory_size'] = traj_file.stat().st_size
            
            results['agents'].append(agent_results)
        
        # Get overall performance (last agent's performance)
        if results['agents']:
            last_agent = results['agents'][-1]
            results.update({
                'r2': last_agent.get('r2', 0.0),
                'rmse': last_agent.get('rmse', float('inf')),
                'cost': results.get('total_cost', 0.0)  # Use MLGym's total cost
            })
        
        return results
    
    def _reference_trajectory_files(self, result: Dict, experiment_id: int):
        """Reference MLGym's trajectory files instead of copying them"""
        try:
            # Create a reference file that points to MLGym's trajectories
            ref_file = EXPERIMENT_CONFIG['results_dir'] / f"experiment_{experiment_id:03d}_trajectory_references.json"
            
            trajectory_refs = {
                'experiment_id': experiment_id,
                'model': result.get('model'),
                'num_agents': result.get('num_agents'),
                'run_id': result.get('run_id'),
                'timestamp': datetime.datetime.now().isoformat(),
                'trajectory_locations': []
            }
            
            # Reference MLGym's trajectory files
            for agent in result.get('agents', []):
                agent_id = agent.get('agent_id', 0)
                agent_dir = Path(agent.get('directory', ''))
                
                if agent_dir.exists():
                    # Find the main trajectory file
                    traj_files = list(agent_dir.glob('*.traj'))
                    results_files = list(agent_dir.glob('results.json'))
                    log_files = list(agent_dir.glob('run-*.log'))
                    
                    agent_refs = {
                        'agent_id': agent_id,
                        'trajectory_directory': str(agent_dir.relative_to(Path('trajectories'))),
                        'main_trajectory': str(traj_files[0].relative_to(Path('trajectories'))) if traj_files else None,
                        'results_file': str(results_files[0].relative_to(Path('trajectories'))) if results_files else None,
                        'log_files': [str(f.relative_to(Path('trajectories'))) for f in log_files],
                        'file_count': len(list(agent_dir.rglob('*')))
                    }
                    
                    trajectory_refs['trajectory_locations'].append(agent_refs)
            
            # Save the reference file
            with open(ref_file, 'w') as f:
                json.dump(trajectory_refs, f, indent=2)
            
            print(f"         📁 Referenced MLGym trajectories: {ref_file}")
            
        except Exception as e:
            print(f"         ⚠️  Warning: Could not reference trajectory files: {e}")


def _reference_trajectory_files(result: Dict, experiment_id: int, results_dir: Path):
    """Standalone function to reference MLGym's trajectory files instead of copying them"""
    try:
        # Create a reference file that points to MLGym's trajectories
        ref_file = results_dir / f"experiment_{experiment_id:03d}_trajectory_references.json"
        
        trajectory_refs = {
            'experiment_id': experiment_id,
            'model': result.get('model'),
            'num_agents': result.get('num_agents'),
            'run_id': result.get('run_id'),
            'timestamp': datetime.datetime.now().isoformat(),
            'trajectory_locations': []
        }
        
        # Reference MLGym's trajectory files
        for agent in result.get('agents', []):
            agent_id = agent.get('agent_id', 0)
            agent_dir = Path(agent.get('directory', ''))
            
            if agent_dir.exists():
                # Find the main trajectory file
                traj_files = list(agent_dir.glob('*.traj'))
                results_files = list(agent_dir.glob('results.json'))
                log_files = list(agent_dir.glob('run-*.log'))
                
                agent_refs = {
                    'agent_id': agent_id,
                    'trajectory_directory': str(agent_dir.relative_to(Path('trajectories'))),
                    'main_trajectory': str(traj_files[0].relative_to(Path('trajectories'))) if traj_files else None,
                    'results_file': str(results_files[0].relative_to(Path('trajectories'))) if results_files else None,
                    'log_files': [str(f.relative_to(Path('trajectories'))) for f in log_files],
                    'file_count': len(list(agent_dir.rglob('*')))
                }
                
                trajectory_refs['trajectory_locations'].append(agent_refs)
        
        # Save the reference file
        with open(ref_file, 'w') as f:
            json.dump(trajectory_refs, f, indent=2)
        
        print(f"         📁 Referenced MLGym trajectories: {ref_file}")
        
    except Exception as e:
        print(f"         ⚠️  Warning: Could not reference trajectory files: {e}")


async def run_experiment_batch(models: List[str], agent_counts: List[int], num_runs: int, results_dir: Path) -> pd.DataFrame:
    """Run all experiments with progress tracking and resume capability"""
    results = []
    total_experiments = len(models) * len(agent_counts) * num_runs
    completed = 0
    
    # Check if we can resume from previous run
    progress_file = results_dir / "experiment_progress.json"
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                if progress_data.get('completed', 0) > 0:
                    print(f"🔄 Found previous progress: {progress_data['completed']}/{total_experiments} completed")
                    resume = input("Do you want to resume from where you left off? (y/N): ")
                    if resume.lower() == 'y':
                        # Load previous results
                        results_file = results_dir / "results_progress.csv"
                        if results_file.exists():
                            previous_df = pd.read_csv(results_file)
                            results = previous_df.to_dict('records')
                            completed = len(results)
                            print(f"✅ Resuming from experiment {completed + 1}")
                        else:
                            print("⚠️  Previous results file not found, starting fresh")
                    else:
                        print("🆕 Starting fresh experiment")
        except Exception as e:
            print(f"⚠️  Could not load previous progress: {e}, starting fresh")
    
    # Create progress tracking file
    progress_file = results_dir / "experiment_progress.json"
    
    print(f"🚀 Starting batch experiment: {total_experiments} total experiments")
    print(f"📁 Results will be saved to: {results_dir}")
    
    for model in models:
        print(f"\n🤖 Testing model: {model}")
        
        for num_agents in agent_counts:
            print(f"  👥 Agent count: {num_agents}")
            
            for run_id in range(num_runs):
                print(f"    🔄 Run {run_id + 1}/{num_runs}")
                
                # Create and run experiment
                runner = AcademicExperimentRunner(model, num_agents, run_id, EXPERIMENT_CONFIG['seed'])
                result = await runner.run_experiment()
                results.append(result)
                
                completed += 1
                progress = completed / total_experiments * 100
                
                print(f"      ✅ Completed ({completed}/{total_experiments}, {progress:.1f}%)")
                if result['success']:
                    print(f"         R²: {result['r2']:.4f}, RMSE: {result['rmse']:.2f}, Cost: ${result['cost']:.2f}")
                    print(f"         💾 Saved: progress.csv, metadata.json, individual result, trajectory references")
                else:
                    print(f"         ❌ Failed: {result['error']}")
                    print(f"         💾 Saved: progress.csv, metadata.json (with error)")
                
                # Save progress after EVERY successful run
                df = pd.DataFrame(results)
                df.to_csv(results_dir / "results_progress.csv", index=False)
                
                # Save progress metadata
                with open(progress_file, 'w') as f:
                    json.dump({
                        'completed': completed,
                        'total': total_experiments,
                        'progress_percent': progress,
                        'last_update': datetime.datetime.now().isoformat(),
                        'last_experiment': {
                            'model': model,
                            'num_agents': num_agents,
                            'run_id': run_id,
                            'success': result['success'],
                            'timestamp': datetime.datetime.now().isoformat()
                        }
                    }, f, indent=2)
                
                # Also save individual experiment result
                experiment_file = results_dir / f"experiment_{completed:03d}_{model.replace(':', '-')}_{num_agents}agents_run{run_id}.json"
                with open(experiment_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                
                # Reference MLGym's trajectory files for reproducibility
                if result['success'] and 'agents' in result:
                    _reference_trajectory_files(result, completed, results_dir)
    
    # Final save
    df = pd.DataFrame(results)
    df.to_csv(results_dir / "results_final.csv", index=False)
    
    print(f"\n🎉 All experiments completed!")
    print(f"📊 Results saved to: {results_dir / 'results_final.csv'}")
    
    # Note: MLGym handles container lifecycle automatically
    # Each experiment gets a fresh persistent container for its sequential agents
    logger.info("MLGym handles container lifecycle automatically")
    logger.info("Each experiment gets a fresh persistent container for its sequential agents")
    
    return df

def analyze_results(df: pd.DataFrame) -> Dict:
    """Comprehensive statistical analysis of experiment results"""
    
    # Filter successful experiments
    successful_df = df[df['success'] == True].copy()
    
    if len(successful_df) == 0:
        return {'error': 'No successful experiments found'}
    
    analysis = {}
    
    # 1. Basic descriptive statistics
    analysis['descriptive'] = {
        'total_experiments': len(df),
        'successful_experiments': len(successful_df),
        'success_rate': len(successful_df) / len(df),
        'models_tested': successful_df['model'].nunique(),
        'agent_counts_tested': successful_df['num_agents'].nunique()
    }
    
    # 2. Performance by number of agents (ANOVA)
    agent_groups = [successful_df[successful_df['num_agents'] == n]['r2'].values 
                   for n in sorted(successful_df['num_agents'].unique())]
    
    if len(agent_groups) > 1:
        f_stat, p_value = f_oneway(*agent_groups)
        
        # Calculate effect size (eta-squared)
        total_ss = sum((x - np.mean(np.concatenate(agent_groups)))**2 for x in agent_groups)
        between_ss = sum(len(g) * (np.mean(g) - np.mean(np.concatenate(agent_groups)))**2 for g in agent_groups)
        eta_squared = between_ss / total_ss if total_ss > 0 else 0
        
        analysis['anova'] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size_eta_squared': eta_squared,
            'effect_size_interpretation': 'large' if eta_squared > 0.14 else 'medium' if eta_squared > 0.06 else 'small'
        }
    
    # 3. Performance by model (ANOVA)
    model_groups = [successful_df[successful_df['model'] == m]['r2'].values 
                   for m in successful_df['model'].unique()]
    
    if len(model_groups) > 1:
        f_stat_model, p_value_model = f_oneway(*model_groups)
        
        # Calculate effect size for model comparison
        total_ss_model = sum((x - np.mean(np.concatenate(model_groups)))**2 for x in model_groups)
        between_ss_model = sum(len(g) * (np.mean(g) - np.mean(np.concatenate(model_groups)))**2 for g in model_groups)
        eta_squared_model = between_ss_model / total_ss_model if total_ss_model > 0 else 0
        
        analysis['model_anova'] = {
            'f_statistic': f_stat_model,
            'p_value': p_value_model,
            'significant': p_value_model < 0.05,
            'effect_size_eta_squared': eta_squared_model,
            'effect_size_interpretation': 'large' if eta_squared_model > 0.14 else 'medium' if eta_squared_model > 0.06 else 'small'
        }
    
    # 4. Multiple comparison correction (Bonferroni)
    if 'anova' in analysis and 'model_anova' in analysis:
        num_tests = 2  # agent count ANOVA + model ANOVA
        bonferroni_alpha = 0.05 / num_tests
        
        analysis['multiple_comparison_correction'] = {
            'method': 'Bonferroni',
            'corrected_alpha': bonferroni_alpha,
            'agent_count_significant': analysis['anova']['p_value'] < bonferroni_alpha,
            'model_significant': analysis['model_anova']['p_value'] < bonferroni_alpha
        }
    
    # 4. Cost-benefit analysis
    successful_df['performance_per_dollar'] = successful_df['r2'] / (successful_df['cost'] + 0.01)
    
    analysis['cost_benefit'] = {
        'best_performance_per_dollar': successful_df['performance_per_dollar'].max(),
        'worst_performance_per_dollar': successful_df['performance_per_dollar'].min(),
        'avg_performance_per_dollar': successful_df['performance_per_dollar'].mean(),
        'cost_efficiency_std': successful_df['performance_per_dollar'].std(),
        'cost_efficiency_cv': successful_df['performance_per_dollar'].std() / successful_df['performance_per_dollar'].mean() if successful_df['performance_per_dollar'].mean() > 0 else 0
    }
    
    # 5. Optimal agent count per model
    optimal_agents = {}
    for model in successful_df['model'].unique():
        model_data = successful_df[successful_df['model'] == model]
        best_agents = model_data.loc[model_data['r2'].idxmax(), 'num_agents']
        optimal_agents[model] = int(best_agents)
    
    analysis['optimal_agents_per_model'] = optimal_agents
    
    return analysis

def create_visualizations(df: pd.DataFrame, results_dir: Path):
    """Create comprehensive visualizations of results"""
    
    successful_df = df[df['success'] == True].copy()
    
    if len(successful_df) == 0:
        print("No successful experiments to visualize")
        return
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig_size = (15, 10)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=fig_size)
    
    # 1. Performance vs Number of Agents
    for model in successful_df['model'].unique():
        model_data = successful_df[successful_df['model'] == model]
        agent_performance = model_data.groupby('num_agents')['r2'].mean()
        axes[0, 0].plot(agent_performance.index, agent_performance.values, 
                marker='o', label=model.split(':')[-1], linewidth=2)
    
    axes[0, 0].set_xlabel('Number of Agents')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('Performance vs Number of Agents')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Cost vs Number of Agents
    for model in successful_df['model'].unique():
        model_data = successful_df[successful_df['model'] == model]
        agent_cost = model_data.groupby('num_agents')['cost'].mean()
        axes[0, 1].plot(agent_cost.index, agent_cost.values, 
                marker='s', label=model.split(':')[-1], linewidth=2)
    
    axes[0, 1].set_xlabel('Number of Agents')
    axes[0, 1].set_ylabel('Total Cost ($)')
    axes[0, 1].set_title('Cost vs Number of Agents')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Success Rate by Agent Count
    success_by_agents = df.groupby('num_agents')['success'].mean() * 100
    axes[0, 2].bar(success_by_agents.index, success_by_agents.values, alpha=0.7)
    axes[0, 2].set_xlabel('Number of Agents')
    axes[0, 2].set_ylabel('Success Rate (%)')
    axes[0, 2].set_title('Success Rate vs Number of Agents')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Performance Distribution by Model
    model_names = [m.split(':')[-1] for m in successful_df['model'].unique()]
    axes[1, 0].boxplot([successful_df[successful_df['model'] == m]['r2'].values 
                for m in successful_df['model'].unique()], 
               labels=model_names)
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title('Performance Distribution by Model')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Cost-Benefit Analysis
    successful_df['performance_per_dollar'] = successful_df['r2'] / (successful_df['cost'] + 0.01)
    
    for model in successful_df['model'].unique():
        model_data = successful_df[successful_df['model'] == model]
        agent_efficiency = model_data.groupby('num_agents')['performance_per_dollar'].mean()
        axes[1, 1].plot(agent_efficiency.index, agent_efficiency.values, 
                marker='^', label=model.split(':')[-1], linewidth=2)
    
    axes[1, 1].set_xlabel('Number of Agents')
    axes[1, 1].set_ylabel('R² per Dollar')
    axes[1, 1].set_title('Cost-Benefit Analysis')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Execution Time vs Agents
    for model in successful_df['model'].unique():
        model_data = successful_df[successful_df['model'] == model]
        agent_time = model_data.groupby('num_agents')['time'].mean()
        axes[1, 2].plot(agent_time.index, agent_time.values, 
                marker='d', label=model.split(':')[-1], linewidth=2)
    
    axes[1, 2].set_xlabel('Number of Agents')
    axes[1, 2].set_ylabel('Execution Time (seconds)')
    axes[1, 2].set_title('Execution Time vs Number of Agents')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualizations saved to: {results_dir / 'comprehensive_analysis.png'}")

def export_results_for_publication(df: pd.DataFrame, results_dir: Path):
    """Export results in formats suitable for academic publication"""
    
    successful_df = df[df['success'] == True].copy()
    
    # 1. Main results table
    results_table = successful_df.groupby(['model', 'num_agents']).agg({
        'r2': ['mean', 'std', 'count'],
        'rmse': ['mean', 'std'],
        'cost': ['mean', 'std'],
        'time': ['mean', 'std']
    }).round(4)
    
    results_table.to_csv(results_dir / "results_table.csv")
    
    # 2. Statistical test results
    analysis = analyze_results(df)
    
    with open(results_dir / "statistical_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    # 3. Raw data for reproducibility
    df.to_csv(results_dir / "raw_data.csv", index=False)
    
    # 4. Experiment metadata
    metadata = {
        'experiment_date': datetime.datetime.now().isoformat(),
        'models_tested': EXPERIMENT_CONFIG['models'],
        'agent_counts_tested': EXPERIMENT_CONFIG['agent_counts'],
        'runs_per_condition': EXPERIMENT_CONFIG['num_runs'],
        'total_experiments': len(df),
        'successful_experiments': len(successful_df),
        'task': EXPERIMENT_CONFIG['task'],
        'seed': EXPERIMENT_CONFIG['seed'],
        'cost_limit_per_agent': EXPERIMENT_CONFIG['per_agent_cost_limit'],
        'max_steps_per_agent': EXPERIMENT_CONFIG['per_agent_max_steps']
    }
    
    with open(results_dir / "experiment_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📄 Results exported for publication to: {results_dir}")
    print(f"   - results_table.csv: Main results table")
    print(f"   - statistical_analysis.json: Statistical test results")
    print(f"   - raw_data.csv: Raw experimental data")
    print(f"   - experiment_metadata.json: Experiment configuration")
    print(f"   - comprehensive_analysis.png: Visualizations")

async def main():
    """Main function to run the experiment"""
    
    print("🔬 Sequential Agents Academic Experiment")
    print("=" * 50)
    
    # Load environment variables
    load_environment_variables()
    
    # Create results directory
    EXPERIMENT_CONFIG['results_dir'].mkdir(parents=True, exist_ok=True)
    
    # Register environment
    register_task(EnvironmentArguments(task_config_path=EXPERIMENT_CONFIG['task']))
    
    # Calculate total experiments
    total_experiments = len(EXPERIMENT_CONFIG['models']) * len(EXPERIMENT_CONFIG['agent_counts']) * EXPERIMENT_CONFIG['num_runs']
    
    print(f"\n📊 Experiment Configuration:")
    print(f"   Models: {len(EXPERIMENT_CONFIG['models'])} ({', '.join(EXPERIMENT_CONFIG['models'])})")
    print(f"   Agent Counts: {EXPERIMENT_CONFIG['agent_counts']}")
    print(f"   Runs per condition: {EXPERIMENT_CONFIG['num_runs']}")
    print(f"   Total experiments: {total_experiments}")
    print(f"   Results directory: {EXPERIMENT_CONFIG['results_dir']}")
    
    # Confirm before running
    if total_experiments > 10:
        response = input(f"\n⚠️  This will run {total_experiments} experiments. Continue? (y/N): ")
        if response.lower() != 'y':
            print("❌ Experiment cancelled.")
            return
    
    # Run experiments
    print(f"\n🚀 Starting experiments...")
    results_df = await run_experiment_batch(
        EXPERIMENT_CONFIG['models'],
        EXPERIMENT_CONFIG['agent_counts'],
        EXPERIMENT_CONFIG['num_runs'],
        EXPERIMENT_CONFIG['results_dir']
    )
    
    # Analyze results
    print(f"\n📈 Analyzing results...")
    analysis = analyze_results(results_df)
    
    print(f"\n📊 Results Summary:")
    print(f"   Total experiments: {len(results_df)}")
    print(f"   Successful: {len(results_df[results_df['success'] == True])}")
    print(f"   Success rate: {len(results_df[results_df['success'] == True]) / len(results_df):.1%}")
    
    if 'anova' in analysis:
        print(f"   Agent count effect (ANOVA): p = {analysis['anova']['p_value']:.4f}")
        print(f"   Significant: {'Yes' if analysis['anova']['significant'] else 'No'}")
        print(f"   Effect size (η²): {analysis['anova']['effect_size_eta_squared']:.4f} ({analysis['anova']['effect_size_interpretation']})")
        
        if 'multiple_comparison_correction' in analysis:
            print(f"   Bonferroni corrected: {'Yes' if analysis['multiple_comparison_correction']['agent_count_significant'] else 'No'}")
    
    # Power analysis recommendation
    if len(EXPERIMENT_CONFIG['agent_counts']) > 1 and EXPERIMENT_CONFIG['num_runs'] < 10:
        print(f"\n⚠️  Power Analysis Recommendation:")
        print(f"   Current runs per condition: {EXPERIMENT_CONFIG['num_runs']}")
        print(f"   For publication, consider: 10+ runs per condition")
        print(f"   This will improve statistical power and effect size reliability")
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    create_visualizations(results_df, EXPERIMENT_CONFIG['results_dir'])
    
    # Export results
    print(f"\n📄 Exporting results...")
    export_results_for_publication(results_df, EXPERIMENT_CONFIG['results_dir'])
    
    print(f"\n🎉 Experiment completed successfully!")
    print(f"📁 All results saved to: {EXPERIMENT_CONFIG['results_dir']}")

if __name__ == "__main__":
    # ============================================================================
    # EXPERIMENT CONFIGURATION - MODIFY THESE PARAMETERS FOR DIFFERENT EXPERIMENTS
    # ============================================================================
    
    EXPERIMENT_CONFIG = {
        # Models to test
        'models': [
            "litellm:gpt-4o-mini",
            # "litellm:gpt-4o",
            # "litellm:claude-3-5-sonnet",
            # "litellm:gemini-1.5-pro",
            # "litellm:llama-3.1-8b-instruct"
        ],
        
        # Number of agents to test
        'agent_counts': [1, 2, 3, 4, 5],
        # 'agent_counts': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Full range
        
        # Number of runs per condition
        'num_runs': 3,
        # 'num_runs': 10,  # Full experiment
        
        # Task configuration
        'task': "tasks/regressionKaggleHousePrice.yaml",
        
        # Cost and step limits
        'per_agent_cost_limit': 3.0,
        'per_agent_max_steps': 50,
        
        # Random seed for reproducibility
        'seed': 42,
        
        # Results directory
        'results_dir': Path("results/academic_experiment")
    }
    
    # Run the experiment
    asyncio.run(main())
