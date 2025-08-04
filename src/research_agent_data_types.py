from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal, Callable
from datetime import datetime


@dataclass
class Goal:
    """Represents the research goal"""
    description: str
    success_criteria: str
    max_iterations: int = 10


@dataclass
class Plan:
    """Represents a research plan"""
    id: str
    description: str
    action: str
    expected_outcome: str
    test_criteria: Callable | str
    created_at: str


@dataclass
class ActionResult:
    """Represents the result of an action"""
    action: str
    success: bool
    output: Optional[str] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class ReviewResult:
    """Represents the result of a review"""
    plan_id: str
    test_results: List[Dict[str, Any]]
    overall_success: bool
    progress_score: float  # 0.0 to 1.0
    feedback: str
    next_steps: List[str]


@dataclass
class TestCriterion:
    description: str
    mode: str  # "function" or "prompt"
    function_code: Optional[str] = None
    prompt_template: Optional[str] = None


@dataclass
class Argument:
    name: str
    value: Any
    description: str
    type: Literal["string", "number", "boolean"]
    required: bool
    default: Optional[Any] = None


@dataclass
class AgentStatus:
    """Current status of the research agent to inform planning decisions"""
    iteration_count: int
    current_goal: Goal
    latest_review: Optional[ReviewResult] = None
    cumulative_progress: float = 0.0
    accumulated_knowledge: List[str] = None
    successful_actions_count: int = 0
    failed_actions_count: int = 0
    workspace_files: List[str] = None
    key_findings: List[str] = None
    current_challenges: List[str] = None
    last_updated: str = None
    
    def __post_init__(self):
        if self.accumulated_knowledge is None:
            self.accumulated_knowledge = []
        if self.workspace_files is None:
            self.workspace_files = []
        if self.key_findings is None:
            self.key_findings = []
        if self.current_challenges is None:
            self.current_challenges = []
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()
    
    def update_from_review(self, review_result: ReviewResult, action_results: List[ActionResult]):
        """Update status based on latest review and action results"""
        self.latest_review = review_result
        self.cumulative_progress = min(1.0, self.cumulative_progress + review_result.progress_score * 0.3)
        
        # Update action counts
        successful_actions = [r for r in action_results if r.success]
        failed_actions = [r for r in action_results if not r.success]
        self.successful_actions_count += len(successful_actions)
        self.failed_actions_count += len(failed_actions)
        
        # Extract key findings from successful actions
        for action_result in successful_actions:
            if action_result.output:
                self.key_findings.append(f"Iteration {self.iteration_count}: {action_result.output}...")
        
        # Extract challenges from failed actions
        for action_result in failed_actions:
            if action_result.error_message:
                self.current_challenges.append(f"Iteration {self.iteration_count}: {action_result.error_message}")
        
        self.last_updated = datetime.now().isoformat()
    
    def get_status_summary(self) -> str:
        """Get a concise summary of current status for planning"""
        summary_parts = [
            f"Iteration: {self.iteration_count}",
            f"Progress: {self.cumulative_progress:.2f}",
            f"Actions: {self.successful_actions_count} successful, {self.failed_actions_count} failed"
        ]
        
        if self.latest_review:
            summary_parts.append(f"Last review score: {self.latest_review.progress_score:.2f}")
        
        if self.key_findings:
            summary_parts.append(f"Key findings: {len(self.key_findings)} items")
        
        if self.current_challenges:
            summary_parts.append(f"Challenges: {len(self.current_challenges)} items")
        
        return " | ".join(summary_parts)
    
    def get_work_to_continue(self) -> Dict[str, Any]:
        """Get work to continue based on current status"""
        return {
            "completed_steps": self.key_findings,
            "current_focus": self.current_goal.description,
        }
    
    def should_adjust_goal(self) -> bool:
        """Determine if goal should be adjusted based on current status"""
        # Adjust if making slow progress after multiple iterations
        if self.iteration_count >= 3 and self.cumulative_progress < 0.4:
            return True
        
        # Adjust if hitting repeated challenges
        if len(self.current_challenges) >= 3:
            return True
        
        return False
    
    def get_goal_adjustments(self) -> Dict[str, Any]:
        """Suggest goal adjustments based on current status"""
        adjustments = {}
        
        if self.cumulative_progress < 0.3 and self.iteration_count >= 2:
            # Simplify goal if making slow progress
            adjustments["description"] = f"Simplified: {self.current_goal.description}"
            adjustments["max_iterations"] = min(self.current_goal.max_iterations + 2, 10)
        
        if len(self.current_challenges) >= 2:
            # Add challenge-specific criteria
            challenge_text = "; ".join([f"Address challenge: {c.split(':')[1].strip()}" for c in self.current_challenges[-2:]])
            adjustments["success_criteria"] = f"{self.current_goal.success_criteria}; {challenge_text}"
        
        return adjustments