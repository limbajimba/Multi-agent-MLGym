"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Research Supervisor agent for the MLGym framework.
This agent extends BaseAgent to provide intelligent research supervision and coordination.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from mlgym.agent.base import BaseAgent, AgentArguments
from mlgym.utils.log import get_logger


class ResearchSupervisor(BaseAgent):
    """
    Research Supervisor that provides intelligent guidance for research tasks.
    
    This agent can:
    - Understand research tasks and success criteria
    - Analyze code quality and progress
    - Provide strategic guidance and intervention
    - Coordinate multi-agent research workflows
    """
    
    def __init__(self, name: str, args: AgentArguments):
        super().__init__(name, args)
        self.research_context = {}      # Research task context and requirements
        self.agent_progress = {}        # Track progress of supervised agents
        self.code_analysis = {}         # Store code quality assessments
        self.research_phases = []       # Track research workflow phases
        self.intervention_history = []  # Track interventions and their impact
        
        # Load configurable parameters from YAML
        self.research_phases_config = getattr(args, 'research_phases', [])
        self.file_patterns_config = getattr(args, 'file_patterns', {})
        self.command_patterns_config = getattr(args, 'command_patterns', {})
        self.intervention_triggers_config = getattr(args, 'intervention_triggers', {})
        
    def analyze_research_task(self, task_description: str, baseline_scores: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a research task to understand requirements and success criteria.
        
        Args:
            task_description: Description of the research task
            baseline_scores: Current baseline performance
            
        Returns:
            Dictionary containing task analysis
        """
        self.logger.info("🔬 Analyzing research task requirements")
        
        # Extract task type and requirements
        task_type = self._identify_task_type(task_description)
        success_criteria = self._extract_success_criteria(task_description, baseline_scores)
        research_phases = self._plan_research_phases(task_type, task_description)
        
        analysis = {
            "task_type": task_type,
            "success_criteria": success_criteria,
            "research_phases": research_phases,
            "baseline_performance": baseline_scores,
            "improvement_targets": self._calculate_improvement_targets(baseline_scores)
        }
        
        self.research_context = analysis
        self.logger.info(f"Task analysis complete: {task_type} task with {len(research_phases)} phases")
        
        return analysis
    
    def analyze_code_quality(self, agent_name: str, file_path: str, code_content: str) -> Dict[str, Any]:
        """
        Analyze code quality and provide assessment.
        
        Args:
            agent_name: Name of the agent being supervised
            file_path: Path to the file being analyzed
            code_content: Content of the code to analyze
            
        Returns:
            Dictionary containing code quality assessment
        """
        self.logger.info(f"📝 Analyzing code quality for {agent_name} in {file_path}")
        
        # Perform code analysis
        syntax_check = self._check_syntax(code_content)
        completeness_check = self._check_completeness(code_content, file_path)
        correctness_check = self._check_correctness(code_content, file_path)
        efficiency_check = self._check_efficiency(code_content)
        
        assessment = {
            "syntax_valid": syntax_check["valid"],
            "syntax_errors": syntax_check["errors"],
            "completeness_score": completeness_check["score"],
            "completeness_issues": completeness_check["issues"],
            "correctness_score": correctness_check["score"],
            "correctness_issues": correctness_check["issues"],
            "efficiency_score": efficiency_check["score"],
            "efficiency_issues": efficiency_check["issues"],
            "overall_quality": self._calculate_overall_quality(syntax_check, completeness_check, correctness_check, efficiency_check)
        }
        
        # Store analysis
        self.code_analysis[f"{agent_name}_{file_path}"] = assessment
        
        self.logger.info(f"Code quality assessment: {assessment['overall_quality']:.2f}/1.0")
        
        return assessment
    
    def provide_strategic_guidance(self, agent_name: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide strategic guidance based on current research state.
        
        Args:
            agent_name: Name of the agent being supervised
            current_state: Current state including progress, files, etc.
            
        Returns:
            Dictionary containing strategic guidance
        """
        self.logger.info(f"🎯 Providing strategic guidance for {agent_name}")
        
        # Analyze current progress
        progress_analysis = self._analyze_progress(agent_name, current_state)
        
        # Determine if intervention is needed
        intervention_needed = self._assess_intervention_needs(progress_analysis)
        
        # Generate strategic guidance
        if intervention_needed:
            guidance = self._generate_intervention_guidance(progress_analysis)
        else:
            guidance = self._generate_encouragement_guidance(progress_analysis)
        
        # Track intervention
        if intervention_needed:
            self.intervention_history.append({
                "agent": agent_name,
                "step": current_state.get("current_step", 0),
                "reason": guidance["reason"],
                "guidance": guidance["message"]
            })
        
        self.logger.info(f"Strategic guidance: {guidance['type']} - {guidance['reason']}")
        
        return guidance
    
    def assess_research_readiness(self, agent_name: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess if research is ready for submission or needs more work.
        
        Args:
            agent_name: Name of the agent being supervised
            current_state: Current state including files, validation results, etc.
            
        Returns:
            Dictionary containing readiness assessment
        """
        self.logger.info(f"✅ Assessing research readiness for {agent_name}")
        
        # Check implementation completeness
        implementation_check = self._check_implementation_completeness(current_state)
        
        # Check validation results
        validation_check = self._check_validation_results(current_state)
        
        # Check performance against baseline
        performance_check = self._check_performance_improvement(current_state)
        
        # Determine readiness
        readiness_score = self._calculate_readiness_score(implementation_check, validation_check, performance_check)
        
        assessment = {
            "ready_for_submission": readiness_score >= 0.8,
            "readiness_score": readiness_score,
            "implementation_complete": implementation_check["complete"],
            "validation_passed": validation_check["passed"],
            "performance_improved": performance_check["improved"],
            "recommendations": self._generate_readiness_recommendations(implementation_check, validation_check, performance_check)
        }
        
        self.logger.info(f"Readiness assessment: {readiness_score:.2f}/1.0 - {'Ready' if assessment['ready_for_submission'] else 'Needs work'}")
        
        return assessment
    
    def _identify_task_type(self, task_description: str) -> str:
        """Identify the type of research task."""
        task_lower = task_description.lower()
        
        if "game theory" in task_lower or "battle" in task_lower or "strategy" in task_lower:
            return "game_theory"
        elif "machine learning" in task_lower or "model" in task_lower or "training" in task_lower:
            return "machine_learning"
        elif "optimization" in task_lower or "minimize" in task_lower or "maximize" in task_lower:
            return "optimization"
        elif "regression" in task_lower or "prediction" in task_lower:
            return "regression"
        elif "classification" in task_lower or "classify" in task_lower:
            return "classification"
        else:
            return "general_research"
    
    def _extract_success_criteria(self, task_description: str, baseline_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Extract success criteria from task description."""
        criteria = {
            "primary_metric": None,
            "target_score": None,
            "constraints": [],
            "requirements": []
        }
        
        # Extract baseline score as target
        if baseline_scores:
            if isinstance(baseline_scores, dict):
                for metric, score in baseline_scores.items():
                    criteria["primary_metric"] = metric
                    criteria["target_score"] = score
                    break
            elif isinstance(baseline_scores, list) and len(baseline_scores) > 0:
                if isinstance(baseline_scores[0], dict):
                    for metric, score in baseline_scores[0].items():
                        criteria["primary_metric"] = metric
                        criteria["target_score"] = score
                        break
        
        # Extract requirements from description
        if "function" in task_description.lower():
            criteria["requirements"].append("implement_correct_function")
        if "validate" in task_description.lower():
            criteria["requirements"].append("validate_solution")
        if "submit" in task_description.lower():
            criteria["requirements"].append("submit_solution")
        
        return criteria
    
    def _plan_research_phases(self, task_type: str, task_description: str) -> List[Dict[str, Any]]:
        """Plan research phases based on task type and configurable parameters."""
        # Use configurable phases from YAML, fallback to defaults if not configured
        if self.research_phases_config:
            phases = self.research_phases_config
        else:
            # Default phases if not configured
            phases = [
                {"name": "understanding", "description": "Understand task requirements and constraints"},
                {"name": "implementation", "description": "Implement core solution components"},
                {"name": "testing", "description": "Validate and test the implementation"},
                {"name": "optimization", "description": "Optimize and improve the solution"},
                {"name": "submission", "description": "Prepare and submit final solution"}
            ]
        
        self.research_phases = phases
        return phases
    
    def _check_syntax(self, code_content: str) -> Dict[str, Any]:
        """Check code syntax validity."""
        try:
            compile(code_content, '<string>', 'exec')
            return {"valid": True, "errors": []}
        except SyntaxError as e:
            return {"valid": False, "errors": [str(e)]}
        except Exception as e:
            return {"valid": False, "errors": [f"Compilation error: {str(e)}"]}
    
    def _check_completeness(self, code_content: str, file_path: str) -> Dict[str, Any]:
        """Check if code implementation is complete."""
        issues = []
        score = 1.0
        
        # Check for required function signatures
        if "strategy.py" in file_path or "row_strategy" in code_content:
            if "def row_strategy" not in code_content:
                issues.append("Missing row_strategy function definition")
                score -= 0.5
            if "return" not in code_content:
                issues.append("Missing return statement in strategy function")
                score -= 0.3
        
        # Check for basic structure
        if len(code_content.strip()) < 10:
            issues.append("Code appears incomplete or empty")
            score -= 0.4
        
        return {"score": max(0.0, score), "issues": issues}
    
    def _check_correctness(self, code_content: str, file_path: str) -> Dict[str, Any]:
        """Check code correctness based on task requirements."""
        issues = []
        score = 1.0
        
        # Basic correctness checks
        if "def row_strategy" in code_content:
            # Check for proper function signature
            if "def row_strategy(history):" not in code_content:
                issues.append("Incorrect function signature for row_strategy")
                score -= 0.3
            
            # Check for proper parameter usage
            if "history" not in code_content:
                issues.append("Function doesn't use history parameter")
                score -= 0.2
        
        return {"score": max(0.0, score), "issues": issues}
    
    def _check_efficiency(self, code_content: str) -> Dict[str, Any]:
        """Check code efficiency."""
        issues = []
        score = 1.0
        
        # Check for potential inefficiencies
        if code_content.count("for") > 3:
            issues.append("Multiple nested loops detected - potential inefficiency")
            score -= 0.1
        
        if len(code_content) > 1000:
            issues.append("Code is quite long - consider simplification")
            score -= 0.1
        
        return {"score": max(0.0, score), "issues": issues}
    
    def _calculate_overall_quality(self, syntax_check: Dict, completeness_check: Dict, 
                                 correctness_check: Dict, efficiency_check: Dict) -> float:
        """Calculate overall code quality score."""
        if not syntax_check["valid"]:
            return 0.0  # Syntax errors make code unusable
        
        # Weighted average of different aspects
        weights = {
            "syntax": 0.3,
            "completeness": 0.3,
            "correctness": 0.3,
            "efficiency": 0.1
        }
        
        overall_score = (
            weights["syntax"] * 1.0 +  # Syntax is valid
            weights["completeness"] * completeness_check["score"] +
            weights["correctness"] * correctness_check["score"] +
            weights["efficiency"] * efficiency_check["score"]
        )
        
        return overall_score
    
    def _analyze_progress(self, agent_name: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze agent progress toward research goals."""
        current_step = current_state.get("current_step", 0)
        max_steps = current_state.get("max_steps", 50)
        files = current_state.get("files", [])
        agent_action = current_state.get("agent_action", "")
        agent_state = current_state.get("agent_state", {})
        
        # Calculate progress metrics
        step_progress = current_step / max_steps if max_steps > 0 else 0
        
        # Use configurable file patterns
        file_patterns = self.file_patterns_config or {
            "implementation_files": ["strategy", "model", "algorithm", "solution", "main"],
            "validation_files": ["validate", "test", "evaluate", "check"],
            "configuration_files": ["config", "settings", "params", "hyperparameters"],
            "documentation_files": ["readme", "doc", "documentation", "notes"]
        }
        
        # Analyze file activity using configurable patterns
        has_implementation = any(
            any(pattern in f.lower() for pattern in file_patterns.get("implementation_files", []))
            for f in files
        )
        has_validation = any(
            any(pattern in f.lower() for pattern in file_patterns.get("validation_files", []))
            for f in files
        )
        has_evaluate = any(
            any(pattern in f.lower() for pattern in file_patterns.get("validation_files", []))
            for f in files
        )
        has_target = any(
            any(pattern in f.lower() for pattern in file_patterns.get("configuration_files", []))
            for f in files
        )
        
        # Analyze agent actions
        action_analysis = self._analyze_agent_actions(agent_action)
        
        # Analyze decoupled agent state
        agent_analysis = self._analyze_decoupled_agent_state(agent_state)
        
        # Determine current research phase based on more factors
        current_phase = self._determine_current_phase(
            has_implementation, has_validation, has_evaluate, has_target, 
            step_progress, action_analysis, agent_analysis
        )
        
        # Calculate more nuanced progress score
        progress_score = self._calculate_progress_score(
            step_progress, has_implementation, has_validation, has_evaluate, 
            has_target, action_analysis, agent_analysis
        )
        
        return {
            "step_progress": step_progress,
            "current_phase": current_phase,
            "has_implementation": has_implementation,
            "has_validation": has_validation,
            "has_evaluation": has_evaluate,
            "has_target": has_target,
            "action_analysis": action_analysis,
            "agent_analysis": agent_analysis,
            "progress_score": progress_score
        }
    
    def _analyze_decoupled_agent_state(self, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze decoupled agent state for better supervision."""
        analysis = {
            "total_actions_proposed": agent_state.get("total_actions_proposed", 0),
            "total_feedback_received": agent_state.get("total_feedback_received", 0),
            "recent_actions": agent_state.get("recent_actions", []),
            "feedback_ratio": 0.0,
            "action_pattern": "unknown"
        }
        
        # Calculate feedback ratio
        if analysis["total_actions_proposed"] > 0:
            analysis["feedback_ratio"] = analysis["total_feedback_received"] / analysis["total_actions_proposed"]
        
        # Analyze action patterns
        recent_actions = analysis["recent_actions"]
        if recent_actions:
            action_text = " ".join(recent_actions).lower()
            if "edit" in action_text or "write" in action_text:
                analysis["action_pattern"] = "implementation"
            elif "validate" in action_text or "test" in action_text:
                analysis["action_pattern"] = "validation"
            elif "submit" in action_text:
                analysis["action_pattern"] = "submission"
            elif "ls" in action_text or "cat" in action_text:
                analysis["action_pattern"] = "exploration"
        
        return analysis
    
    def _analyze_agent_actions(self, agent_action: str) -> Dict[str, Any]:
        """Analyze agent actions to understand behavior."""
        action_lower = agent_action.lower()
        
        # Use configurable command patterns
        command_patterns = self.command_patterns_config or {
            "exploration_commands": ["ls", "cat", "head", "find", "grep", "search"],
            "implementation_commands": ["edit", "insert", "create", "write", "echo"],
            "validation_commands": ["validate", "test", "run", "execute", "check"],
            "submission_commands": ["submit", "finalize", "complete"]
        }
        
        analysis = {
            "action_type": "unknown",
            "is_productive": False,
            "is_exploratory": False,
            "is_implementation": False,
            "is_validation": False,
            "is_submission": False,
            "has_format_issues": False
        }
        
        # Check for format issues first
        if "```" in agent_action and "```" in agent_action:
            # Check if it's properly formatted
            parts = agent_action.split("```")
            if len(parts) < 3 or not parts[1].strip():
                analysis["has_format_issues"] = True
        
        # Check for malformed output patterns
        if "discussion" in action_lower and "```" not in action_lower:
            analysis["has_format_issues"] = True
        
        if "```" in action_lower and "discussion" not in action_lower:
            analysis["has_format_issues"] = True
        
        # Categorize actions using configurable patterns
        exploration_commands = command_patterns.get("exploration_commands", [])
        implementation_commands = command_patterns.get("implementation_commands", [])
        validation_commands = command_patterns.get("validation_commands", [])
        submission_commands = command_patterns.get("submission_commands", [])
        
        if any(cmd in action_lower for cmd in exploration_commands):
            analysis["action_type"] = "exploration"
            analysis["is_exploratory"] = True
        elif any(cmd in action_lower for cmd in implementation_commands):
            analysis["action_type"] = "implementation"
            analysis["is_implementation"] = True
            analysis["is_productive"] = True
        elif any(cmd in action_lower for cmd in validation_commands):
            analysis["action_type"] = "validation"
            analysis["is_validation"] = True
            analysis["is_productive"] = True
        elif any(cmd in action_lower for cmd in submission_commands):
            analysis["action_type"] = "submission"
            analysis["is_submission"] = True
            analysis["is_productive"] = True
        elif "python" in action_lower or "run" in action_lower:
            analysis["action_type"] = "execution"
            analysis["is_productive"] = True
        elif "open" in action_lower:
            analysis["action_type"] = "navigation"
            analysis["is_exploratory"] = True
        
        return analysis
    
    def _determine_current_phase(self, has_implementation: bool, has_validation: bool, 
                               has_evaluate: bool, has_target: bool, 
                               step_progress: float, action_analysis: Dict[str, Any], agent_analysis: Dict[str, Any]) -> str:
        """Determine current research phase with more context."""
        # If agent is doing exploration, likely in understanding phase
        if action_analysis["is_exploratory"] and step_progress < 0.3:
            return "understanding"
        
        # If agent is implementing, in implementation phase
        if action_analysis["is_implementation"] or has_implementation:
            if not has_validation:
                return "implementation"
            else:
                return "testing"
        
        # If agent is validating, in testing phase
        if action_analysis["is_validation"] or has_validation:
            return "testing"
        
        # If agent is submitting, in submission phase
        if action_analysis["is_submission"]:
            return "submission"
        
        # Default phase determination based on progress
        if step_progress < 0.2:
            return "understanding"
        elif step_progress < 0.6:
            return "implementation"
        elif step_progress < 0.8:
            return "testing"
        else:
            return "submission"
    
    def _calculate_progress_score(self, step_progress: float, has_implementation: bool, 
                                has_validation: bool, has_evaluate: bool, has_target: bool,
                                action_analysis: Dict[str, Any], agent_analysis: Dict[str, Any]) -> float:
        """Calculate overall progress score with more factors."""
        score = 0.0
        
        # Step progress weight (25%)
        score += step_progress * 0.25
        
        # Implementation weight (20%)
        if has_implementation:
            score += 0.2
        
        # Validation weight (15%)
        if has_validation:
            score += 0.15
        
        # Evaluation weight (10%)
        if has_evaluate:
            score += 0.1
        
        # Productive actions weight (10%)
        if action_analysis["is_productive"]:
            score += 0.1
        
        # Decoupled agent state factors (20%)
        agent_score = 0.0
        
        # Reward for good action patterns
        action_pattern = agent_analysis.get("action_pattern", "unknown")
        if action_pattern == "implementation":
            agent_score += 0.1
        elif action_pattern == "validation":
            agent_score += 0.15
        elif action_pattern == "submission":
            agent_score += 0.2
        
        # Penalty for too much exploration
        if action_pattern == "exploration" and step_progress > 0.3:
            agent_score -= 0.05
        
        # Reward for balanced feedback ratio
        feedback_ratio = agent_analysis.get("feedback_ratio", 0)
        if 0.1 <= feedback_ratio <= 0.3:  # Good balance
            agent_score += 0.05
        elif feedback_ratio > 0.5:  # Too much feedback
            agent_score -= 0.05
        
        score += agent_score
        
        return min(1.0, max(0.0, score))
    
    def _assess_intervention_needs(self, progress_analysis: Dict[str, Any]) -> bool:
        """Assess if intervention is needed."""
        progress_score = progress_analysis["progress_score"]
        current_phase = progress_analysis["current_phase"]
        step_progress = progress_analysis["step_progress"]
        agent_analysis = progress_analysis.get("agent_analysis", {})
        action_analysis = progress_analysis.get("action_analysis", {})
        
        # Use configurable thresholds
        thresholds = self.intervention_triggers_config or {
            "low_progress_threshold": 0.2,
            "stuck_phase_threshold": 0.6,
            "no_implementation_threshold": 0.4,
            "no_validation_threshold": 0.7,
            "high_progress_low_score_threshold": 0.8,
            "feedback_ratio_threshold": 0.5,
            "exploration_stuck_threshold": 0.3,
            "repeated_actions_threshold": 20,
            "low_progress_with_actions_threshold": 0.3
        }
        
        # DEBUG: Log the values being checked
        self.logger.info(f"🔍 DEBUG - Intervention check values:")
        self.logger.info(f"   progress_score: {progress_score}")
        self.logger.info(f"   current_phase: {current_phase}")
        self.logger.info(f"   step_progress: {step_progress}")
        self.logger.info(f"   action_analysis: {action_analysis}")
        self.logger.info(f"   agent_analysis: {agent_analysis}")
        
        # More sophisticated intervention detection
        intervention_triggers = []
        
        # 1. Very low progress
        if progress_score < thresholds.get("low_progress_threshold", 0.2):
            intervention_triggers.append("Very low progress score")
        
        # 2. Stuck in early phases too long
        if current_phase in ["understanding", "implementation"] and step_progress > thresholds.get("stuck_phase_threshold", 0.6):
            intervention_triggers.append(f"Stuck in {current_phase} phase too long")
        
        # 3. No implementation despite being in later steps
        if not progress_analysis["has_implementation"] and step_progress > thresholds.get("no_implementation_threshold", 0.4):
            intervention_triggers.append("No implementation despite being in later steps")
        
        # 4. No validation despite having implementation
        if progress_analysis["has_implementation"] and not progress_analysis["has_validation"] and step_progress > thresholds.get("no_validation_threshold", 0.7):
            intervention_triggers.append("No validation despite having implementation")
        
        # 5. Very high step progress but low overall progress
        if step_progress > thresholds.get("high_progress_low_score_threshold", 0.8) and progress_score < 0.5:
            intervention_triggers.append("High step progress but low research progress")
        
        # 6. Decoupled agent specific triggers
        if agent_analysis.get("feedback_ratio", 0) > thresholds.get("feedback_ratio_threshold", 0.5):
            intervention_triggers.append("Agent receiving too much feedback - may need redirection")
        
        if agent_analysis.get("action_pattern") == "exploration" and step_progress > thresholds.get("exploration_stuck_threshold", 0.3):
            intervention_triggers.append("Agent stuck in exploration mode")
        
        if agent_analysis.get("total_actions_proposed", 0) > thresholds.get("repeated_actions_threshold", 20) and progress_score < thresholds.get("low_progress_with_actions_threshold", 0.3):
            intervention_triggers.append("Agent making many actions but little progress")
        
        # 7. Command format issues (NEW - based on observed problems)
        if action_analysis.get("has_format_issues", False):
            intervention_triggers.append("Agent has command format issues")
        elif action_analysis.get("action_type") == "unknown" and step_progress > 0.1:
            intervention_triggers.append("Agent may have command format issues")
        
        # 8. Repeated failed attempts (NEW - based on observed pattern)
        recent_actions = agent_analysis.get("recent_actions", [])
        if len(recent_actions) >= 3:
            # Check if agent is repeating similar actions (indicating stuck)
            action_text = " ".join(recent_actions).lower()
            if action_text.count("edit") > 2 or action_text.count("insert") > 2:
                intervention_triggers.append("Agent repeating failed edit attempts")
        
        # Log intervention triggers for debugging
        if intervention_triggers:
            self.logger.info(f"🔍 Intervention triggers: {intervention_triggers}")
        else:
            self.logger.info(f"🔍 No intervention triggers detected")
        
        return len(intervention_triggers) > 0
    
    def _generate_intervention_guidance(self, progress_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intervention guidance based on detailed progress analysis."""
        current_phase = progress_analysis["current_phase"]
        action_analysis = progress_analysis["action_analysis"]
        step_progress = progress_analysis["step_progress"]
        agent_analysis = progress_analysis.get("agent_analysis", {})
        
        # Check for specific intervention triggers
        recent_actions = agent_analysis.get("recent_actions", [])
        action_text = " ".join(recent_actions).lower()
        
        # Command format issues
        if action_analysis.get("action_type") == "unknown" and step_progress > 0.1:
            return {
                "type": "intervention",
                "reason": "Command format issues detected",
                "message": "It appears you're having trouble with MLGym's command format. Remember to use the exact format: DISCUSSION followed by a command block with ```. For example: DISCUSSION [your thoughts] ``` [your command] ```",
                "priority": "high"
            }
        
        # Repeated failed edit attempts
        if action_text.count("edit") > 2 or action_text.count("insert") > 2:
            return {
                "type": "intervention",
                "reason": "Repeated failed edit attempts",
                "message": "You seem to be having trouble with the edit commands. Try using simpler approaches: 1) Use 'echo' to write files directly, 2) Use 'insert' instead of 'edit' for new content, 3) Check the file content first with 'cat' to understand the current state.",
                "priority": "high"
            }
        
        if current_phase == "understanding":
            if action_analysis["is_exploratory"]:
                return {
                    "type": "intervention",
                    "reason": "Need to move beyond exploration to implementation",
                    "message": "You've been exploring the task for a while. It's time to start implementing your solution. Focus on understanding the key requirements and begin coding your strategy function.",
                    "priority": "high"
                }
            else:
                return {
                    "type": "intervention",
                    "reason": "Need to understand task requirements better",
                    "message": "Take time to thoroughly understand the task requirements. Read the task description carefully and identify the key components you need to implement.",
                    "priority": "high"
                }
        elif current_phase == "implementation":
            if not progress_analysis["has_implementation"]:
                return {
                    "type": "intervention", 
                    "reason": "Implementation phase needs focus",
                    "message": "You're in the implementation phase but haven't created the core strategy function yet. Focus on implementing the row_strategy function in strategy.py. Don't get distracted by optimization - get a working solution first.",
                    "priority": "high"
                }
            else:
                return {
                    "type": "intervention",
                    "reason": "Implementation needs testing",
                    "message": "You have an implementation but haven't tested it yet. Use the validate command to test your strategy function and see how it performs.",
                    "priority": "medium"
                }
        elif current_phase == "testing":
            if not progress_analysis["has_validation"]:
                return {
                    "type": "intervention",
                    "reason": "Need to validate current implementation",
                    "message": "Test your current implementation using the validate command. This will help you understand how well your solution performs and identify areas for improvement.",
                    "priority": "medium"
                }
            else:
                return {
                    "type": "intervention",
                    "reason": "Ready for optimization or submission",
                    "message": "Your implementation has been tested. Consider whether you need to optimize further or if you're ready to submit your solution.",
                    "priority": "low"
                }
        else:
            return {
                "type": "intervention",
                "reason": "General guidance needed",
                "message": "Consider the next steps in your research process. Are you ready to submit, or do you need to optimize further?",
                "priority": "low"
            }
    
    def _generate_encouragement_guidance(self, progress_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate encouragement guidance based on progress."""
        current_phase = progress_analysis["current_phase"]
        action_analysis = progress_analysis["action_analysis"]
        progress_score = progress_analysis["progress_score"]
        
        if current_phase == "optimization":
            return {
                "type": "encouragement",
                "reason": "Good progress, focus on optimization",
                "message": "Great progress! You have a working implementation. Now focus on optimizing your solution to improve performance and beat the baseline score.",
                "priority": "low"
            }
        elif current_phase == "submission":
            return {
                "type": "encouragement",
                "reason": "Ready for submission",
                "message": "Excellent work! Your solution appears ready for submission. Make sure to validate one final time before submitting to ensure the best performance.",
                "priority": "low"
            }
        elif action_analysis["is_productive"]:
            return {
                "type": "encouragement",
                "reason": "Making productive progress",
                "message": "Good work! You're making productive progress. Continue with your current approach and focus on the next phase of your research.",
                "priority": "low"
            }
        elif progress_score > 0.5:
            return {
                "type": "encouragement",
                "reason": "Good overall progress",
                "message": "You're making good overall progress. Keep up the momentum and focus on completing the current phase.",
                "priority": "low"
            }
        else:
            return {
                "type": "encouragement",
                "reason": "Continue current approach",
                "message": "You're making steady progress. Continue with your current approach and focus on the next phase of your research.",
                "priority": "low"
            }
    
    def _calculate_improvement_targets(self, baseline_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate improvement targets based on baseline."""
        targets = {}
        
        if baseline_scores:
            if isinstance(baseline_scores, dict):
                for metric, score in baseline_scores.items():
                    # Aim for 10% improvement over baseline
                    targets[metric] = score * 1.1
            elif isinstance(baseline_scores, list) and len(baseline_scores) > 0:
                if isinstance(baseline_scores[0], dict):
                    for metric, score in baseline_scores[0].items():
                        targets[metric] = score * 1.1
        
        return targets
    
    def _check_implementation_completeness(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check if implementation is complete."""
        files = current_state.get("files", [])
        has_strategy = any("strategy" in f.lower() for f in files)
        
        return {
            "complete": has_strategy,
            "missing": [] if has_strategy else ["strategy implementation"]
        }
    
    def _check_validation_results(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check validation results."""
        # This would need to be implemented based on actual validation output
        # For now, assume validation is needed but not yet done
        return {
            "passed": False,
            "results": None,
            "message": "Validation not yet performed"
        }
    
    def _check_performance_improvement(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check if performance has improved over baseline."""
        # This would need actual performance metrics
        # For now, assume no improvement yet
        return {
            "improved": False,
            "current_score": None,
            "baseline_score": self.research_context.get("baseline_performance", {})
        }
    
    def _calculate_readiness_score(self, implementation_check: Dict, validation_check: Dict, performance_check: Dict) -> float:
        """Calculate readiness score for submission."""
        score = 0.0
        
        if implementation_check["complete"]:
            score += 0.4
        
        if validation_check["passed"]:
            score += 0.3
        
        if performance_check["improved"]:
            score += 0.3
        
        return score
    
    def _generate_readiness_recommendations(self, implementation_check: Dict, validation_check: Dict, performance_check: Dict) -> List[str]:
        """Generate recommendations for submission readiness."""
        recommendations = []
        
        if not implementation_check["complete"]:
            recommendations.append("Complete the core implementation before submitting")
        
        if not validation_check["passed"]:
            recommendations.append("Run validation to test your solution")
        
        if not performance_check["improved"]:
            recommendations.append("Consider optimizing your solution to improve performance")
        
        return recommendations
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get summary of research supervision activities."""
        return {
            "task_type": self.research_context.get("task_type", "unknown"),
            "total_interventions": len(self.intervention_history),
            "code_analyses": len(self.code_analysis),
            "research_phases": len(self.research_phases),
            "recent_interventions": self.intervention_history[-3:] if self.intervention_history else [],
            "code_quality_scores": [
                {"file": k, "quality": v["overall_quality"]} 
                for k, v in self.code_analysis.items()
            ]
        } 