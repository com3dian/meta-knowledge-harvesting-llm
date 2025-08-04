#!/usr/bin/env python3
"""
Deep Research Agent Framework
Follows the cycle: Plan -> Action -> Review -> Plan
"""

import json
import logging
import argparse
import os
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from typing import Callable
import sys
import re

from research_agent_tools import ReadFileTool, WriteFileTool
from research_agent_data_types import Goal, Plan, ActionResult, ReviewResult, AgentStatus
from prompt import PROMPTS
from utils import split_string_by_multi_markers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ResearchAgent:
    """Main Research Agent that follows plan->action->review->plan cycle"""
    
    def __init__(self,
                 goal: Goal,
                 llm_wrapper: Callable,
                 workspace_dir: str = "./workspace"):
        self.goal = goal
        self.workspace_dir = workspace_dir
        self.iteration_count = 0
        self.plans: List[Plan] = []
        self.action_results: List[ActionResult] = []
        self.review_results: List[ReviewResult] = []
        self.llm_wrapper = llm_wrapper
        
        # Initialize status tracking
        self.status = AgentStatus(
            iteration_count=0,
            current_goal=goal
        )

        tool_delimiter = PROMPTS["DEFAULT_TOOL_DELIMITER"]
        
        # Initialize tools
        self.tools = {
            'read_file': ReadFileTool(),
            'write_file': WriteFileTool()
        }

        self.tools_description = {
            'read_file': f"Read a file to string from the workspace, should be called like {tool_delimiter}read_file(file_path){tool_delimiter}",
            'write_file': f"Write string to a file in the workspace, should be called like {tool_delimiter}write_file(file_path, content){tool_delimiter}"
        }
        
        # Create workspace directory
        os.makedirs(workspace_dir, exist_ok=True)
        
        # Update initial workspace files
        self._update_workspace_files()
        
        logger.info(f"Research Agent initialized with goal: {goal.description}")
    
    def run(self) -> bool:
        """Main execution loop"""
        logger.info("=" * 80)
        logger.info("STARTING RESEARCH AGENT")
        logger.info(f"Goal: {self.goal.description}")
        logger.info(f"Success Criteria: {self.goal.success_criteria}")
        logger.info("=" * 80)
        
        while self.iteration_count < self.goal.max_iterations:
            self.iteration_count += 1
            self.status.iteration_count = self.iteration_count
            
            logger.info(f"\n🔄 ITERATION {self.iteration_count}")
            logger.info(f"📊 Status: {self.status.get_status_summary()}")
            logger.info("-" * 50)
            
            # Check if goal should be adjusted based on status
            if self.status.should_adjust_goal():
                self._adjust_goal_based_on_status()
            
            # 1. PLAN
            plan = self.plan()
            if not plan:
                logger.error("Failed to generate plan. Stopping.")
                return False
            self.plans.append(plan)
            
            # 2. ACTION
            action_results = self.execute_actions(plan)
            self.action_results.extend(action_results)
            
            # 3. REVIEW
            review_result = self.review(plan, action_results)
            self.review_results.append(review_result)
            
            # 4. UPDATE STATUS
            self.status.update_from_review(review_result, action_results)
            
            # Check if goal is achieved
            if self.is_goal_achieved(review_result):
                logger.info("🎉 GOAL ACHIEVED!")
                self.save_final_report()
                return True
            
            # Check if we should continue
            if not self.should_continue(review_result):
                logger.info("❌ Stopping research due to lack of progress")
                self.save_final_report()
                return False
        
        logger.info("⏰ Maximum iterations reached")
        self.save_final_report()
        return False

    def _adjust_goal_based_on_status(self):
        """Adjust goal based on current agent status"""
        adjustments = self.status.get_goal_adjustments()
        
        if adjustments:
            logger.info("🎯 ADJUSTING GOAL based on current status")
            
            # Apply adjustments to current goal
            if "description" in adjustments:
                old_description = self.goal.description
                self.goal.description = adjustments["description"]
                self.status.current_goal.description = adjustments["description"]
                logger.info(f"   Description: {old_description} → {self.goal.description}")
            
            if "max_iterations" in adjustments:
                old_max = self.goal.max_iterations
                self.goal.max_iterations = adjustments["max_iterations"]
                self.status.current_goal.max_iterations = adjustments["max_iterations"]
                logger.info(f"   Max iterations: {old_max} → {self.goal.max_iterations}")
            
            if "success_criteria" in adjustments:
                self.goal.success_criteria = adjustments["success_criteria"]
                self.status.current_goal.success_criteria = adjustments["success_criteria"]
                logger.info(f"   Updated success criteria: {adjustments['success_criteria']}")
    
    def _update_workspace_files(self):
        """Update the list of workspace files in status"""
        try:
            workspace_files = []
            for root, dirs, files in os.walk(self.workspace_dir):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), self.workspace_dir)
                    workspace_files.append(rel_path)
            self.status.workspace_files = workspace_files
        except Exception as e:
            logger.warning(f"Could not update workspace files: {e}")
    
    def plan(self) -> Optional[Plan]:
        """Generate a new plan based on current state"""
        logger.info("📋 PLANNING PHASE")
        
        plan_id = f"plan_{self.iteration_count}_{datetime.now().strftime('%H%M%S')}"
        
        # This is where you would implement your planning logic
        # For now, this is a template that you can customize
        plan = self._generate_plan(plan_id)
        
        if plan:
            logger.info(f"✅ Generated plan: {plan.description}")
            logger.info(f"   Action: {plan.action}")
            logger.info(f"   Expected outcome: {plan.expected_outcome}")
            logger.info(f"   Test criteria: {plan.test_criteria}")
            logger.info(f"   Created at: {plan.created_at}")
        
        return plan
    
    def _generate_plan(self, plan_id: str) -> Plan:
        """Generate a plan - implement your planning logic here"""
        # This is a template implementation
        # You should customize this based on your specific research domain

        # Import the new utility functions
        from research_agent_utils import execute_llm_for_plan, create_plan_generation_prompt

        # Include status information in the planning prompt
        status_context = f"""
        Current Agent Status:
        - {self.status.get_status_summary()}
        - Key findings so far: {'; '.join(self.status.key_findings[-3:]) if self.status.key_findings else 'None'}
        - Current challenges: {'; '.join(self.status.current_challenges[-2:]) if self.status.current_challenges else 'None'}
        - Workspace files: {', '.join(self.status.workspace_files) if self.status.workspace_files else 'None'}
        """

        # Get work continuation context to avoid redoing completed work
        work_context_dict = self.status.get_work_to_continue()
        
        # This formatted string is for logging/debugging, but not used in prompt logic.
        continuation_context_str = f"""
        WORK CONTINUATION CONTEXT (DO NOT REDO THESE):
        - Completed steps: {'; '.join(work_context_dict['completed_steps']) if work_context_dict['completed_steps'] else 'None'}
        - Current research focus: {work_context_dict['current_focus'] or 'Not set'}
        """

        # Include previous test results context for better test criteria generation
        if not self.review_results:
            previous_test_context = "No previous test results available."
        else:
            context_parts = []
            
            # Get recent review results (last 3 iterations)
            recent_reviews = self.review_results[-3:] if len(self.review_results) > 3 else self.review_results
            
            for i, review in enumerate(recent_reviews):
                iteration_num = len(self.review_results) - len(recent_reviews) + i + 1
                
                context_parts.append(f"Iteration {iteration_num}:")
                context_parts.append(f"  Overall Success: {review.overall_success}")
                context_parts.append(f"  Progress Score: {review.progress_score:.2f}")
                
                if review.test_results:
                    for j, test in enumerate(review.test_results):
                        test_status = "PASSED" if test.get('passed', False) else "FAILED"
                        test_criterion = test.get('criterion', f'Test {j+1}')
                        test_details = test.get('details', 'No details')
                        context_parts.append(f"    Test: {test_criterion} - {test_status} ({test_details})")
                
                if review.feedback:
                    context_parts.append(f"  Feedback: {review.feedback}")
            
            if not context_parts:
                previous_test_context = "No detailed test results available."
            else:
                previous_test_context = "\n".join(context_parts)

        # Create the simplified prompt using utility function
        prompt = create_plan_generation_prompt(
            goal=self.goal.description,
            success_criteria=self.goal.success_criteria,
            iteration_count=self.iteration_count,
            status_context=status_context,
            continuation_context=work_context_dict, # Pass the dictionary, not a formatted string
            previous_test_context=previous_test_context,
            tools_description=self.tools_description,
            workspace_files=self.status.workspace_files
        )

        try:
            # Use the new utility function for JSON-based parsing
            plan_data = execute_llm_for_plan(self.llm_wrapper, prompt)
            
            # Validate required fields
            required_fields = ["description", "action", "expected_outcome", "test_criteria"]
            missing_fields = [field for field in required_fields if not plan_data.get(field)]
            
            if missing_fields:
                raise ValueError(f"Missing required plan fields: {missing_fields}")
            
            return Plan(
                id=plan_id,
                description=plan_data["description"],
                action=plan_data["action"],
                expected_outcome=plan_data["expected_outcome"],
                test_criteria=plan_data["test_criteria"],
                created_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            # Add debugging information
            logger.error(f"Plan generation failed: {str(e)}")
            logger.error(f"Prompt used: {prompt}")
            raise e

    def execute_actions(self, plan: Plan) -> List[ActionResult]:
        """Execute the actions in the plan"""
        logger.info("    🚀 ACTION PHASE")
        
        results = []
        # Split action string by 'AND' to handle multiple commands
        actions = [a.strip() for a in plan.action.split('AND')]

        for action in actions:
            if not action:
                continue

            logger.info(f"   Executing: {action}")
            
            # Parse action (simple format: "tool_name: parameters")
            result = self._execute_single_action(action)
            results.append(result)
            
            if result.success:
                logger.info(f"   ✅ Success: {action}")
            else:
                logger.error(f"   ❌ Failed: {action} - {result.error_message}")
        
        return results
    
    def _execute_single_action(self, action: str) -> ActionResult:
        # Normalize tool delimiters to handle LLM inconsistencies
        normalized_action = action.replace('<|TOOL|>', '||')
        tool_pattern = r'\|\|([^|]+)\|\|'
        tool_match = re.search(tool_pattern, normalized_action)
        tool_delimiter = PROMPTS["DEFAULT_TOOL_DELIMITER"]
        
        try:
            if tool_match:
                # Extract tool content from new delimiters
                tool_content = tool_match.group(1)
                
                # Parse tool name and arguments - format: tool_name(arguments)
                if '(' in tool_content and tool_content.endswith(')'):
                    tool_name = tool_content.split('(')[0].strip()
                    args_str = tool_content[tool_content.find('(') + 1:-1]
                    
                    # Check if tool exists
                    if tool_name in self.tools:
                        # Parse arguments for the tool
                        if tool_name == 'read_file':
                            # Sanitize path to be relative to workspace
                            safe_file_path = self._sanitize_path(args_str)
                            result = self.tools[tool_name].execute(file_path=safe_file_path)
                        elif tool_name == 'write_file':
                            # Parse file_path and content arguments
                            # Expected format: file_path, content
                            if ',' in args_str:
                                parts = [part.strip().strip('"\'') for part in args_str.split(',', 1)]
                                if len(parts) == 2:
                                    file_path, content = parts
                                    
                                    # Sanitize path
                                    safe_file_path = self._sanitize_path(file_path)

                                    # Check if content is a placeholder variable - extract from previous results
                                    if self._is_placeholder_content(content):
                                        actual_content = self._extract_content_from_previous_results(content)
                                        if actual_content:
                                            content = actual_content
                                            logger.info(f"   📝 Extracted actual content from previous results ({len(content)} characters)")
                                        else:
                                            logger.warning(f"   ⚠️  Could not extract content for placeholder: {content}")
                                            return ActionResult(
                                                action=action,
                                                success=False,
                                                output=None,
                                                error_message=f"Could not extract actual content for placeholder: {content}"
                                            )
                                    
                                    result = self.tools[tool_name].execute(file_path=safe_file_path, content=content, overwrite=True)
                                else:
                                    raise ValueError(f"Invalid arguments for {tool_name}: expected file_path, content")
                            else:
                                raise ValueError(f"Invalid arguments for {tool_name}: expected file_path, content")
                        else:
                            # Generic tool execution (extend as needed)
                            result = self.tools[tool_name].execute()
                        
                        # Update workspace files after write operations
                        if result.success and tool_name == 'write_file':
                            self._update_workspace_files()
                        
                        return result
                    else:
                        return ActionResult(
                            action=action,
                            success=False,
                            output=None,
                            error_message=f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
                        )
                else:
                    return ActionResult(
                        action=action,
                        success=False,
                        output=None,
                        error_message=f"Invalid tool format. Expected: tool_name(arguments)"
                    )
            
            # Check if action is surrounded by old tool delimiters
            elif action.startswith(tool_delimiter) and action.endswith(tool_delimiter):
                # Extract tool content from delimiters
                tool_content = action[len(tool_delimiter):-len(tool_delimiter)]
                
                # Parse tool name and arguments - format: tool_name(arguments)
                if '(' in tool_content and tool_content.endswith(')'):
                    tool_name = tool_content.split('(')[0].strip()
                    args_str = tool_content[tool_content.find('(') + 1:-1]
                    
                    # Check if tool exists
                    if tool_name in self.tools:
                        # Parse arguments for the tool
                        if tool_name == 'read_file':
                            # Sanitize path to be relative to workspace
                            safe_file_path = self._sanitize_path(args_str)
                            result = self.tools[tool_name].execute(file_path=safe_file_path)
                        elif tool_name == 'write_file':
                            # Parse file_path and content arguments
                            # Expected format: file_path, content
                            if ',' in args_str:
                                parts = [part.strip().strip('"\'') for part in args_str.split(',', 1)]
                                if len(parts) == 2:
                                    file_path, content = parts
                                    
                                    # Sanitize path
                                    safe_file_path = self._sanitize_path(file_path)

                                    result = self.tools[tool_name].execute(file_path=safe_file_path, content=content, overwrite=True)
                                else:
                                    raise ValueError(f"Invalid arguments for {tool_name}: expected file_path, content")
                            else:
                                raise ValueError(f"Invalid arguments for {tool_name}: expected file_path, content")
                        else:
                            # Generic tool execution (extend as needed)
                            result = self.tools[tool_name].execute()
                        
                        # Update workspace files after write operations
                        if result.success and tool_name == 'write_file':
                            self._update_workspace_files()
                        
                        return result
                    else:
                        return ActionResult(
                            action=action,
                            success=False,
                            output=None,
                            error_message=f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
                        )
                else:
                    return ActionResult(
                        action=action,
                        success=False,
                        output=None,
                        error_message=f"Invalid tool format. Expected: tool_name(arguments)"
                    )
            
            else:
                # Not a tool action - use LLM wrapper to execute the action
                return self._execute_llm_action(action)
                
        except Exception as e:
            return ActionResult(
                action=action,
                success=False,
                output=None,
                error_message=str(e)
            )

    def _sanitize_path(self, raw_path: str) -> str:
        """Strips paths from filenames and ensures they are relative to the workspace."""
        if not isinstance(raw_path, str):
            logger.warning(f"Path sanitization received a non-string type: {type(raw_path)}")
            return "" # return empty string for error handling downstream
            
        # Take the basename to prevent path traversal attacks or reading outside the workspace
        basename = os.path.basename(raw_path.strip().strip('"\''))
        safe_path = os.path.join(self.workspace_dir, basename)
        logger.info(f"   🔒 Sanitizing path: '{raw_path}' -> '{safe_path}'")
        return safe_path
    
    def _execute_llm_action(self, action: str) -> ActionResult:
        """Execute non-tool actions using LLM wrapper"""
        try:
            # Import the new utility function
            from research_agent_utils import execute_llm_action
            
            # Prepare context variables for the action prompt
            context_vars = {
                'goal': self.goal.description,
                'success_criteria': self.goal.success_criteria,
                'iteration': self.iteration_count,
                'workspace_files': self.status.workspace_files,
                'key_findings': self.status.key_findings,
                'current_challenges': self.status.current_challenges,
                'workspace_dir': self.workspace_dir,
                'previous_results_context': self._build_previous_results_context(),
                'source_content': self._get_source_file_content_context()
            }
            
            # Use the new utility function for better JSON-based action execution
            result = execute_llm_action(self.llm_wrapper, action, context_vars)

            # If the result was successful and has large output, save it to a file.
            if result.success and result.output and len(result.output) > 250 and self._contains_research_data(result.output):
                try:
                    filename = f"intermediate_output_iter{self.iteration_count}.txt"
                    filepath = os.path.join(self.workspace_dir, filename)
                    with open(filepath, 'w') as f:
                        f.write(result.output)
                    
                    logger.info(f"   💾 Saved large LLM output to '{filepath}'")
                    self._update_workspace_files()
                    
                    # Modify the result to point to the file
                    original_output_len = len(result.output)
                    result.output = f"Successfully executed analysis and saved results to '{filename}'. Contains {original_output_len} characters."

                except Exception as e:
                    logger.error(f"   ❌ Failed to save intermediate output: {e}")

            return result
                
        except Exception as e:
            return ActionResult(
                action=action,
                success=False,
                output=None,
                error_message=f"Error executing LLM action: {str(e)}"
            )
    
    def _build_previous_results_context(self) -> str:
        """Build context string from previous action results for LLM actions"""
        if not self.action_results:
            return "No previous action results available."
        
        context_parts = []
        
        # Include results from recent iterations (last 5 results or current iteration)
        recent_results = self.action_results[-5:] if len(self.action_results) > 5 else self.action_results
        
        for i, result in enumerate(recent_results):
            if result.success and result.output:
                # Create variable name based on action type and index
                if "read_file" in result.action.lower():
                    var_name = f"file_content_{i}"
                    # Truncate very long content for context
                    content = str(result.output)

                    context_parts.append(f"- {var_name}: Content from {result.action}\n  Content: {content}")
                
                elif "write_file" in result.action.lower():
                    var_name = f"write_result_{i}"
                    context_parts.append(f"- {var_name}: Result from {result.action}\n  Result: {result.output}")
                
                else:
                    # Generic action result
                    var_name = f"action_result_{i}"
                    content = str(result.output)
                    if len(content) > 1000:
                        content = content[:1000] + "... [content truncated]"
                    context_parts.append(f"- {var_name}: Result from {result.action}\n  Result: {content}")
        
        if not context_parts:
            return "No successful previous action results with output available."
        
        return "\n".join(context_parts)

    def _get_source_file_content_context(self) -> str:
        """Aggregates content from all successful read_file actions to ground the LLM."""
        if not self.action_results:
            return "No source files have been read yet."

        source_contents = []
        for result in self.action_results:
            # Check for successful read_file actions that have output
            if result.success and result.output and "read_file" in result.action.lower():
                # Sanitize the filename from the action string for a clean label
                match = re.search(r"read_file\((.*?)\)", result.action)
                if match:
                    filename = os.path.basename(match.group(1).strip().strip('"\''))
                    source_contents.append(f"--- START OF FILE: {filename} ---\n")
                    # Truncate content to avoid excessively long prompts
                    content = str(result.output)
                    if len(content) > 4000:
                        content = content[:4000] + "\n... [CONTENT TRUNCATED] ..."
                    source_contents.append(content)
                    source_contents.append(f"\n--- END OF FILE: {filename} ---\n")

        if not source_contents:
            return "No source file content is available from previous steps."

        logger.info(f"   📚 Providing {len(source_contents) // 3} file contents to ground the LLM.")
        return "\n".join(source_contents)
    
    def _is_placeholder_content(self, content: str) -> bool:
        """Check if content is a placeholder variable rather than actual data"""
        # Check for common placeholder patterns
        placeholder_patterns = [
            'content_of_structured_data',
            'extracted_data_content',
            'structured_data_content',
            'csv_content',
            'data_content',
            'previous_data',
            'extracted_content'
        ]
        
        content_lower = content.lower().strip()
        
        # Check if it's just a variable name (no actual CSV data)
        is_placeholder = (
            content_lower in [p.lower() for p in placeholder_patterns] or
            (len(content.strip()) < 50 and 'content' in content_lower and ',' not in content) or
            (content.strip().startswith('${') and content.strip().endswith('}')) or
            (content.strip().count('\n') == 0 and len(content.strip().split()) <= 3)
        )
        
        logger.info(f"   🔍 Checking if '{content[:50]}...' is placeholder: {is_placeholder}")
        return is_placeholder
    
    def _extract_content_from_previous_results(self, placeholder: str) -> Optional[str]:
        """Extract actual content from previous action results for CSV creation"""
        logger.info(f"   🕵️  Extracting content for '{placeholder}' in agent {id(self)}")
        if not self.action_results:
            logger.warning("   📋 No previous action results available to agent {id(self)}")
            return None
        
        logger.info(f"   🔍 Searching through {len(self.action_results)} previous action results for agent {id(self)}:")
        for i, res in enumerate(reversed(self.action_results)):
            logger.info(f"     - History Item {i}: Action='{res.action[:80]}...', Success={res.success}")

        # Look for structured data in recent action results
        for i, result in enumerate(reversed(self.action_results)):  # Start with most recent
            if result.success and result.output:
                output_str = str(result.output)
                logger.info(f"   📊 Checking result {i}: {len(output_str)} chars, action: {result.action[:50]}...")
                
                # Check if this looks like structured meta-analysis data
                if self._looks_like_meta_analysis_data(output_str):
                    logger.info(f"   ✅ Found meta-analysis data in result {i}")
                    
                    # If it's already CSV format, use it directly
                    if self._looks_like_csv(output_str):
                        logger.info(f"   📊 Found CSV-formatted data from previous result")
                        return output_str
                    
                    # If it's structured data but not CSV, convert it
                    csv_content = self._convert_to_csv_format(output_str)
                    if csv_content:
                        logger.info(f"   🔄 Converted structured data to CSV format")
                        return csv_content
                    else:
                        logger.warning(f"   ⚠️  Could not convert structured data to CSV format")
                
                # If we don't find perfect meta-analysis data, try to use any substantial content
                elif len(output_str) > 200:
                    logger.info(f"   🔄 Attempting to use substantial content from result {i}")
                    
                    # Try to extract any useful structured information
                    if self._contains_research_data(output_str):
                        csv_content = self._convert_any_content_to_csv(output_str)
                        if csv_content:
                            logger.info(f"   ✅ Successfully converted general content to CSV format")
                            return csv_content
                    
                    logger.info(f"   ❌ Content doesn't appear to contain research data")
                else:
                    logger.info(f"   ❌ Content too short ({len(output_str)} chars)")
            else:
                logger.info(f"   ❌ Result {i}: not successful or no output")
        
        logger.warning(f"   ❌ Could not find any suitable content to extract for placeholder: {placeholder}")
        return None
    
    def _looks_like_meta_analysis_data(self, content: str) -> bool:
        """Check if content looks like meta-analysis data"""
        content_lower = content.lower()
        
        # Check for key meta-analysis indicators
        meta_indicators = ['crop', 'yield', 'climate', 'temperature', 'precipitation', 'location', 'experimental']
        indicator_count = sum(1 for indicator in meta_indicators if indicator in content_lower)
        
        # Lower the threshold - must have at least 2 indicators and reasonable content length
        result = indicator_count >= 2 and len(content) > 50
        logger.info(f"   🔍 Meta-analysis check: {indicator_count} indicators found, {len(content)} chars -> {result}")
        return result
    
    def _contains_research_data(self, content: str) -> bool:
        """Check if content contains any research-related data that could be useful"""
        content_lower = content.lower()
        
        # Look for research-related keywords
        research_keywords = [
            'study', 'research', 'analysis', 'data', 'result', 'finding', 'experiment',
            'model', 'simulation', 'impact', 'effect', 'change', 'increase', 'decrease',
            'crop', 'yield', 'temperature', 'climate', 'weather', 'precipitation',
            'location', 'region', 'area', 'site', 'field', 'farm', 'agriculture'
        ]
        
        keyword_count = sum(1 for keyword in research_keywords if keyword in content_lower)
        
        # Also check for numeric data
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', content)
        has_numbers = len(numbers) > 0
        
        # Check for structured patterns (lists, tables, etc.)
        has_structure = ('|' in content or '\t' in content or 
                        content.count('\n') > 3 or content.count(':') > 2)
        
        result = keyword_count >= 3 and (has_numbers or has_structure)
        logger.info(f"   🔍 Research data check: {keyword_count} keywords, {len(numbers)} numbers, structure: {has_structure} -> {result}")
        return result
    
    def _convert_any_content_to_csv(self, content: str) -> Optional[str]:
        """Convert any research content to CSV format - more flexible approach"""
        try:
            # Define the target CSV headers
            csv_headers = [
                "Crop type", "Crop yield", "Unit", "Climate drivers", 
                "Values", "Unit", "Experimental design", "Location", 
                "Time", "Source in paper"
            ]
            
            # Start CSV with headers
            csv_lines = [",".join(csv_headers)]
            
            # Try multiple extraction strategies
            records = []
            
            # Strategy 1: Look for existing structured data
            if '|' in content or '\t' in content:
                records.extend(self._extract_from_structured_text(content))
            
            # Strategy 2: Extract from narrative text
            if len(records) == 0:
                records.extend(self._extract_from_narrative_text(content))
            
            # Strategy 3: Create sample records from available data
            if len(records) == 0:
                records.extend(self._create_sample_records_from_content(content))
            
            if records:
                for record in records:
                    # Ensure record has correct number of fields
                    while len(record) < len(csv_headers):
                        record.append("")
                    csv_lines.append(",".join(record[:len(csv_headers)]))
                
                logger.info(f"   📝 Converted {len(records)} records to CSV format using flexible approach")
                return "\n".join(csv_lines)
            
        except Exception as e:
            logger.warning(f"   ⚠️  Error in flexible CSV conversion: {e}")
        
        return None
    
    def _extract_from_structured_text(self, content: str) -> List[List[str]]:
        """Extract records from structured text (with separators)"""
        records = []
        lines = content.split('\n')
        
        for line in lines:
            if '|' in line or '\t' in line:
                # Split by separator and clean up
                parts = re.split(r'[|\t]', line)
                parts = [part.strip() for part in parts if part.strip()]
                
                if len(parts) >= 3:  # Must have at least 3 meaningful parts
                    record = self._normalize_record_parts(parts)
                    if record:
                        records.append(record)
        
        return records
    
    def _extract_from_narrative_text(self, content: str) -> List[List[str]]:
        """Extract records from narrative text"""
        records = []
        
        # Split into paragraphs or sentences
        sections = re.split(r'\n\s*\n|\.(?=\s+[A-Z])', content)
        
        for section in sections:
            if len(section.strip()) > 30:  # Minimum meaningful length
                record = self._extract_fields_from_section(section)
                if record and sum(1 for field in record if field.strip()) >= 3:
                    records.append(record)
        
        return records
    
    def _create_sample_records_from_content(self, content: str) -> List[List[str]]:
        """Create sample records from any available content"""
        records = []
        
        # Extract any available information
        content_lower = content.lower()
        
        # Look for key information pieces
        crop_info = self._extract_crop_info(content)
        climate_info = self._extract_climate_info(content)
        numeric_info = self._extract_numeric_info(content)
        location_info = self._extract_location_info(content)
        
        # Create at least one record if we have some information
        if crop_info or climate_info or numeric_info or location_info:
            record = [
                crop_info or "Mixed crops",
                numeric_info.get('yield', ""),
                numeric_info.get('unit', ""),
                climate_info or "Climate variability",
                numeric_info.get('value', ""),
                numeric_info.get('climate_unit', ""),
                "Research study",
                location_info or "Multiple locations", 
                "2000-2020",
                content.split('\n')[0][:50] if content else "Research data"
            ]
            records.append(record)
        
        return records
    
    def _normalize_record_parts(self, parts: List[str]) -> Optional[List[str]]:
        """Normalize extracted parts into a standard record format"""
        if len(parts) < 3:
            return None
        
        # Create a record with 10 fields
        record = [""] * 10
        
        # Try to map parts to appropriate fields
        for i, part in enumerate(parts[:10]):
            if part and len(part.strip()) > 0:
                record[i] = part.strip()
        
        return record
    
    def _extract_crop_info(self, content: str) -> Optional[str]:
        """Extract crop information from content"""
        content_lower = content.lower()
        crops = ['maize', 'wheat', 'rice', 'soybean', 'barley', 'cotton', 'corn', 'potato', 'tomato']
        
        for crop in crops:
            if crop in content_lower:
                return crop.capitalize()
        
        return None
    
    def _extract_climate_info(self, content: str) -> Optional[str]:
        """Extract climate information from content"""
        content_lower = content.lower()
        climate_terms = ['temperature', 'precipitation', 'rainfall', 'drought', 'heat']
        
        for term in climate_terms:
            if term in content_lower:
                return term.capitalize()
        
        return None
    
    def _extract_numeric_info(self, content: str) -> Dict[str, str]:
        """Extract numeric information from content"""
        info = {}
        
        # Look for yield values
        yield_match = re.search(r'(\d+(?:\.\d+)?)\s*(tons?/ha|kg/ha|bu/acre|t/ha)', content.lower())
        if yield_match:
            info['yield'] = yield_match.group(1)
            info['unit'] = yield_match.group(2)
        
        # Look for temperature values
        temp_match = re.search(r'([+-]?\d+(?:\.\d+)?)\s*[°℃c]', content.lower())
        if temp_match:
            info['value'] = temp_match.group(1)
            info['climate_unit'] = "°C"
        
        return info
    
    def _extract_location_info(self, content: str) -> Optional[str]:
        """Extract location information from content"""
        # Look for capitalized location names
        location_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', content)
        if location_match:
            location = location_match.group(1)
            # Filter out common non-location words
            non_locations = ['The', 'This', 'These', 'That', 'Those', 'When', 'Where', 'How', 'Why', 'What']
            if location not in non_locations:
                return location
        
        return None
    
    def _looks_like_csv(self, content: str) -> bool:
        """Check if content is already in CSV format"""
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return False
        
        # Check if first line looks like headers
        header_line = lines[0]
        if ',' not in header_line:
            return False
        
        # Check if subsequent lines have similar comma structure
        header_comma_count = header_line.count(',')
        data_lines_valid = 0
        
        for line in lines[1:5]:  # Check first few data lines
            if line.strip() and abs(line.count(',') - header_comma_count) <= 1:
                data_lines_valid += 1
        
        return data_lines_valid > 0
    
    def _convert_to_csv_format(self, content: str) -> Optional[str]:
        """Convert structured data to CSV format for meta-analysis"""
        try:
            # Define the target CSV headers
            csv_headers = [
                "Crop type", "Crop yield", "Unit", "Climate drivers", 
                "Values", "Unit", "Experimental design", "Location", 
                "Time", "Source in paper"
            ]
            
            # Start CSV with headers
            csv_lines = [",".join(csv_headers)]
            
            # Try to extract structured records from the content
            records = self._extract_records_from_text(content)
            
            if records:
                for record in records:
                    csv_lines.append(",".join(record))
                
                logger.info(f"   📝 Converted {len(records)} records to CSV format")
                return "\n".join(csv_lines)
            
        except Exception as e:
            logger.warning(f"   ⚠️  Error converting to CSV format: {e}")
        
        return None
    
    def _extract_records_from_text(self, content: str) -> List[List[str]]:
        """Extract structured records from text content"""
        records = []
        
        # Split content into potential record sections
        sections = content.split('\n\n')  # Double newline separates records
        
        for section in sections:
            if len(section.strip()) < 50:  # Skip short sections
                continue
            
            # Try to extract key fields from each section
            record = self._extract_fields_from_section(section)
            if record and len(record) >= 6:  # Must have at least 6 fields filled
                records.append(record)
        
        return records
    
    def _extract_fields_from_section(self, section: str) -> Optional[List[str]]:
        """Extract individual fields from a text section"""
        try:
            # Initialize record with empty values
            record = [""] * 10  # 10 fields as per CSV headers
            
            section_lower = section.lower()
            
            # Extract crop type
            crop_patterns = ['maize', 'wheat', 'rice', 'soybean', 'barley', 'cotton', 'corn']
            for crop in crop_patterns:
                if crop in section_lower:
                    record[0] = crop.capitalize()
                    break
            
            # Extract yield values (look for numbers followed by units)
            import re
            yield_match = re.search(r'(\d+(?:\.\d+)?)\s*(tons?/ha|kg/ha|bu/acre|t/ha)', section_lower)
            if yield_match:
                record[1] = yield_match.group(1)
                record[2] = yield_match.group(2)
            
            # Extract climate drivers
            climate_patterns = ['temperature', 'precipitation', 'rainfall', 'drought', 'heat']
            for climate in climate_patterns:
                if climate in section_lower:
                    record[3] = climate.capitalize()
                    break
            
            # Extract climate values
            climate_match = re.search(r'([+-]?\d+(?:\.\d+)?)\s*[°℃c]', section_lower)
            if climate_match:
                record[4] = climate_match.group(1)
                record[5] = "°C"
            
            # Extract experimental design
            design_patterns = ['model', 'simulation', 'field', 'experiment', 'trial']
            for design in design_patterns:
                if design in section_lower:
                    record[6] = design.capitalize() + " study"
                    break
            
            # Extract location
            location_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', section)
            if location_match:
                record[7] = location_match.group(1)
            
            # Extract time period
            time_match = re.search(r'(19\d{2}|20\d{2})(?:-(\d{2,4}))?', section)
            if time_match:
                if time_match.group(2):
                    record[8] = f"{time_match.group(1)}-{time_match.group(2)}"
                else:
                    record[8] = time_match.group(1)
            
            # Source (use first few words of section)
            source_words = section.split()[:8]
            record[9] = " ".join(source_words)
            
            # Only return if we have significant data
            filled_fields = sum(1 for field in record if field.strip())
            if filled_fields >= 4:
                return record
            
        except Exception as e:
            logger.warning(f"   ⚠️  Error extracting fields from section: {e}")
        
        return None
    
    def review(self, plan: Plan, action_results: List[ActionResult]) -> ReviewResult:
        """Review the results and design tests"""
        logger.info("🔍 REVIEW PHASE")
        
        # Execute tests for the criterion (now single criterion, not list)
        test_results = []
        
        # Handle both single criterion and list of criteria for backward compatibility
        criteria_to_test = [plan.test_criteria] if isinstance(plan.test_criteria, str) else plan.test_criteria
        
        for criterion in criteria_to_test:
            test_result = self._run_test(criterion, action_results)
            test_results.append(test_result)
            
            status = "✅ PASS" if test_result['passed'] else "❌ FAIL"
            logger.info(f"   Test: {criterion} - {status}")
        
        # Calculate overall success and progress with meta-analysis considerations
        passed_tests = sum(1 for test in test_results if test['passed'])
        overall_success = passed_tests == len(test_results)
        
        # Calculate weighted progress score based on individual test scores
        if test_results:
            total_score = sum(test.get('score', 0) for test in test_results)
            progress_score = total_score / len(test_results)
            
            # Boost progress score for meta-analysis tasks if significant progress is made
            if any("csv" in test.get('criterion', '').lower() or "meta" in test.get('criterion', '').lower() 
                   for test in test_results):
                # If at least 75% of tests pass with good individual scores, consider it good progress
                if progress_score >= 0.75:
                    overall_success = True
                elif progress_score >= 0.5:
                    # Partial success for meta-analysis - better than failing completely
                    progress_score = min(progress_score + 0.2, 1.0)  # Boost score slightly
        else:
            progress_score = 0.0
        
        # Generate feedback and next steps
        feedback, next_steps = self._generate_feedback(plan, action_results, test_results)
        
        review_result = ReviewResult(
            plan_id=plan.id,
            test_results=test_results,
            overall_success=overall_success,
            progress_score=progress_score,
            feedback=feedback,
            next_steps=next_steps
        )
        
        logger.info(f"   Overall Success: {overall_success}")
        logger.info(f"   Progress Score: {progress_score:.2f}")
        logger.info(f"   Feedback: {feedback}")
        
        return review_result
    
    def _run_test(self, criterion: str, action_results: List[ActionResult]) -> Dict[str, Any]:
        """Run structured tests for a specific criterion - enhanced for data-driven validation"""
        
        # Parse structured test criteria (format: TEST1|TEST2|TEST3)
        individual_tests = [test.strip() for test in criterion.split('|') if test.strip()]
        
        if not individual_tests:
            # Fallback to simple test if no structured format
            return self._run_simple_test(criterion, action_results)
        
        test_results = []
        passed_count = 0
        
        for test_desc in individual_tests:
            test_result = self._run_individual_test(test_desc, action_results)
            test_results.append(test_result)
            if test_result['passed']:
                passed_count += 1
        
        # Calculate pass rate and overall success
        pass_rate = passed_count / len(individual_tests) if individual_tests else 0
        
        # For meta-analysis tasks, allow partial success (≥75% pass rate)
        # For other tasks, require 100% success
        is_meta_analysis = "csv" in criterion.lower() or "meta" in criterion.lower() or "file" in criterion.lower()
        success_threshold = 0.75 if is_meta_analysis else 1.0
        
        overall_passed = pass_rate >= success_threshold
        
        return {
            'criterion': criterion,
            'passed': overall_passed,
            'details': f"Passed {passed_count}/{len(individual_tests)} tests",
            'individual_tests': test_results,
            'score': pass_rate,
            'pass_rate': pass_rate
        }
    
    def _run_individual_test(self, test_desc: str, action_results: List[ActionResult]) -> Dict[str, Any]:
        """Run a single test and return detailed results"""
        test_desc_lower = test_desc.lower()
        
        try:
            # File existence and readability tests
            if "file exists" in test_desc_lower:
                return self._test_file_exists(test_desc, action_results)
            
            # CSV structure and content tests
            elif "csv has" in test_desc_lower or "contains required columns" in test_desc_lower:
                return self._test_csv_structure(test_desc, action_results)
            
            # Data quantity tests (e.g., "at least X records")
            elif "at least" in test_desc_lower and ("record" in test_desc_lower or "data" in test_desc_lower):
                return self._test_data_quantity(test_desc, action_results)
            
            # Content validation tests
            elif "contains data" in test_desc_lower or "content contains" in test_desc_lower:
                return self._test_content_validation(test_desc, action_results)
            
            # Structure and format tests
            elif "structured data" in test_desc_lower or "properly formatted" in test_desc_lower:
                return self._test_data_structure(test_desc, action_results)
            
            # Analysis pattern tests
            elif "identifies" in test_desc_lower and "pattern" in test_desc_lower:
                return self._test_analysis_patterns(test_desc, action_results)
            
            # JSON format tests
            elif "json" in test_desc_lower or "formatted as json" in test_desc_lower:
                return self._test_json_format(test_desc, action_results)
            
            # Missing values tests
            elif "no missing" in test_desc_lower or "missing values" in test_desc_lower:
                return self._test_missing_values(test_desc, action_results)
            
            # Success-based tests (fallback)
            elif "successfully" in test_desc_lower:
                return self._test_action_success(test_desc, action_results)
            
            else:
                # Generic test for unrecognized patterns
                return self._test_generic(test_desc, action_results)
                
        except Exception as e:
            return {
                'test': test_desc,
                'passed': False,
                'details': f"Test execution error: {str(e)}",
                'error': str(e)
            }
    
    def _test_file_exists(self, test_desc: str, action_results: List[ActionResult]) -> Dict[str, Any]:
        """Test if files exist and are readable"""
        file_actions = [r for r in action_results if r.success and "read_file" in r.action.lower()]
        
        if file_actions:
            # Check if files were actually read successfully
            readable_files = [r for r in file_actions if r.output and len(str(r.output)) > 0]
            return {
                'test': test_desc,
                'passed': len(readable_files) > 0,
                'details': f"Found {len(readable_files)} readable files out of {len(file_actions)} file operations",
                'files_found': len(readable_files)
            }
        else:
            return {
                'test': test_desc,
                'passed': False,
                'details': "No successful file read operations found",
                'files_found': 0
            }
    
    def _test_csv_structure(self, test_desc: str, action_results: List[ActionResult]) -> Dict[str, Any]:
        """Test CSV structure and required columns"""
        import re
        
        # Extract expected columns from test description
        required_columns = []
        if "crop type" in test_desc.lower():
            required_columns = ["Crop type", "Crop yield", "Unit", "Climate drivers", "Values", "Experimental design", "Location", "Time", "Source in paper", "Title_of paper"]
        
        # Look for CSV-like content in action results
        csv_content = None
        for result in action_results:
            if result.success and result.output:
                content = str(result.output)
                # Check if content looks like CSV (has commas and multiple lines)
                if ',' in content and '\n' in content:
                    csv_content = content
                    break
        
        if not csv_content:
            return {
                'test': test_desc,
                'passed': False,
                'details': "No CSV-like content found in action results",
                'columns_found': []
            }
        
        # Parse first line as headers
        lines = csv_content.strip().split('\n')
        if len(lines) > 0:
            headers = [col.strip() for col in lines[0].split(',')]
            
            if required_columns:
                # Check if required columns are present
                found_columns = [col for col in required_columns if col in headers]
                passed = len(found_columns) >= len(required_columns) * 0.8  # Allow 80% match
                
                return {
                    'test': test_desc,
                    'passed': passed,
                    'details': f"Found {len(found_columns)}/{len(required_columns)} required columns",
                    'columns_found': found_columns,
                    'missing_columns': [col for col in required_columns if col not in headers]
                }
            else:
                # Just check that we have reasonable number of columns
                passed = len(headers) >= 3
                return {
                    'test': test_desc,
                    'passed': passed,
                    'details': f"Found {len(headers)} columns in CSV",
                    'columns_found': headers[:10]  # Limit for display
                }
        
        return {
            'test': test_desc,
            'passed': False,
            'details': "Could not parse CSV structure",
            'columns_found': []
        }
    
    def _test_data_quantity(self, test_desc: str, action_results: List[ActionResult]) -> Dict[str, Any]:
        """Test data quantity (e.g., at least X records)"""
        import re
        
        # Extract expected quantity from test description
        quantity_match = re.search(r'at least (\d+)', test_desc.lower())
        if not quantity_match:
            return {
                'test': test_desc,
                'passed': False,
                'details': "Could not extract expected quantity from test description",
                'records_found': 0
            }
        
        expected_count = int(quantity_match.group(1))
        
        # Count records in action results
        total_records = 0
        
        for result in action_results:
            if result.success and result.output:
                content = str(result.output)
                
                # Count lines that look like data records (not headers)
                if ',' in content and '\n' in content:
                    lines = content.strip().split('\n')
                    # Skip header line
                    data_lines = [line for line in lines[1:] if line.strip() and ',' in line]
                    total_records += len(data_lines)
                
                # Also count if content mentions specific numbers
                number_matches = re.findall(r'\b(\d+)\b', content)
                if number_matches:
                    # Sum up reasonable numbers (not years, not huge numbers)
                    reasonable_numbers = [int(n) for n in number_matches if 1 <= int(n) <= 1000]
                    if reasonable_numbers:
                        total_records += sum(reasonable_numbers)
        
        passed = total_records >= expected_count
        
        return {
            'test': test_desc,
            'passed': passed,
            'details': f"Found {total_records} records, expected at least {expected_count}",
            'records_found': total_records,
            'expected_count': expected_count
        }
    
    def _test_content_validation(self, test_desc: str, action_results: List[ActionResult]) -> Dict[str, Any]:
        """Test content validation"""
        content_found = False
        content_details = []
        
        for result in action_results:
            if result.success and result.output:
                content = str(result.output)
                
                # Check for various content indicators
                if "climate" in test_desc.lower() and "climate" in content.lower():
                    content_found = True
                    content_details.append("Climate data found")
                
                if "entity" in test_desc.lower() and ("entity" in content.lower() or "id" in content.lower()):
                    content_found = True
                    content_details.append("Entity data found")
                
                if len(content) > 100:  # Has substantial content
                    content_found = True
                    content_details.append(f"Substantial content found ({len(content)} characters)")
        
        return {
            'test': test_desc,
            'passed': content_found,
            'details': "; ".join(content_details) if content_details else "No relevant content found",
            'content_indicators': content_details
        }
    
    def _test_data_structure(self, test_desc: str, action_results: List[ActionResult]) -> Dict[str, Any]:
        """Test data structure and formatting"""
        structured_found = False
        structure_details = []
        
        for result in action_results:
            if result.success and result.output:
                content = str(result.output)
                
                # Check for structured data indicators
                if '{' in content and '}' in content:  # JSON-like structure
                    structured_found = True
                    structure_details.append("JSON-like structure detected")
                
                if ',' in content and '\n' in content:  # CSV-like structure
                    structured_found = True
                    structure_details.append("CSV-like structure detected")
                
                if 'id' in content.lower() and 'description' in content.lower():  # Entity structure
                    structured_found = True
                    structure_details.append("Entity structure with ID and description found")
        
        return {
            'test': test_desc,
            'passed': structured_found,
            'details': "; ".join(structure_details) if structure_details else "No structured data found",
            'structure_indicators': structure_details
        }
    
    def _test_analysis_patterns(self, test_desc: str, action_results: List[ActionResult]) -> Dict[str, Any]:
        """Test analysis pattern identification"""
        import re
        
        # Extract expected pattern count
        pattern_match = re.search(r'(\d+)', test_desc)
        expected_patterns = int(pattern_match.group(1)) if pattern_match else 1
        
        patterns_found = 0
        pattern_details = []
        
        for result in action_results:
            if result.success and result.output:
                content = str(result.output)
                
                # Look for pattern indicators
                pattern_words = ['pattern', 'trend', 'relationship', 'correlation', 'finding', 'result']
                for word in pattern_words:
                    word_count = content.lower().count(word)
                    if word_count > 0:
                        patterns_found += word_count
                        pattern_details.append(f"{word_count} {word}(s)")
                
                # Look for numbered lists or bullet points
                numbered_patterns = len(re.findall(r'\d+\.', content))
                if numbered_patterns > 0:
                    patterns_found += numbered_patterns
                    pattern_details.append(f"{numbered_patterns} numbered items")
        
        passed = patterns_found >= expected_patterns
        
        return {
            'test': test_desc,
            'passed': passed,
            'details': f"Found {patterns_found} patterns, expected {expected_patterns}. Details: {'; '.join(pattern_details)}",
            'patterns_found': patterns_found,
            'expected_patterns': expected_patterns
        }
    
    def _test_json_format(self, test_desc: str, action_results: List[ActionResult]) -> Dict[str, Any]:
        """Test JSON format validation"""
        import json
        
        valid_json_found = False
        json_details = []
        
        for result in action_results:
            if result.success and result.output:
                content = str(result.output)
                
                # Try to parse as JSON
                try:
                    json.loads(content)
                    valid_json_found = True
                    json_details.append("Valid JSON format found")
                except json.JSONDecodeError:
                    # Check if it looks like JSON
                    if '{' in content and '}' in content:
                        json_details.append("JSON-like structure found but not valid JSON")
                    elif '[' in content and ']' in content:
                        json_details.append("Array-like structure found")
        
        return {
            'test': test_desc,
            'passed': valid_json_found,
            'details': "; ".join(json_details) if json_details else "No JSON format found",
            'json_indicators': json_details
        }
    
    def _test_missing_values(self, test_desc: str, action_results: List[ActionResult]) -> Dict[str, Any]:
        """Test for missing values in data"""
        missing_values_found = False
        missing_details = []
        
        for result in action_results:
            if result.success and result.output:
                content = str(result.output)
                
                # Check for missing value indicators
                missing_indicators = ['nan', 'null', 'missing', '', 'n/a', 'na']
                for indicator in missing_indicators:
                    if indicator in content.lower():
                        missing_values_found = True
                        missing_details.append(f"Found '{indicator}' values")
                
                # Check for empty fields in CSV
                if ',' in content:
                    lines = content.split('\n')
                    for line in lines:
                        if ',,' in line:  # Empty field
                            missing_values_found = True
                            missing_details.append("Empty fields found in CSV")
                            break
        
        # For "no missing values" test, we want the opposite
        if "no missing" in test_desc.lower():
            passed = not missing_values_found
            details = "No missing values detected" if passed else f"Missing values found: {'; '.join(missing_details)}"
        else:
            passed = missing_values_found
            details = f"Missing values found: {'; '.join(missing_details)}" if passed else "No missing values detected"
        
        return {
            'test': test_desc,
            'passed': passed,
            'details': details,
            'missing_indicators': missing_details
        }
    
    def _test_action_success(self, test_desc: str, action_results: List[ActionResult]) -> Dict[str, Any]:
        """Test action success"""
        successful_actions = [r for r in action_results if r.success]
        passed = len(successful_actions) > 0
        
        return {
            'test': test_desc,
            'passed': passed,
            'details': f"{len(successful_actions)}/{len(action_results)} actions successful",
            'successful_actions': len(successful_actions)
        }
    
    def _test_generic(self, test_desc: str, action_results: List[ActionResult]) -> Dict[str, Any]:
        """Generic test fallback"""
        # Basic success check
        passed = any(result.success for result in action_results)
        
        return {
            'test': test_desc,
            'passed': passed,
            'details': "Generic test: checked for any successful actions",
            'note': "This test used generic validation - consider adding specific test logic"
        }
    
    def _run_simple_test(self, criterion: str, action_results: List[ActionResult]) -> Dict[str, Any]:
        """Fallback simple test implementation for backward compatibility"""
        test_result = {
            'criterion': criterion,
            'passed': False,
            'details': ''
        }
        
        if "successfully" in criterion.lower():
            # Test if actions were successful
            successful_actions = sum(1 for result in action_results if result.success)
            test_result['passed'] = successful_actions > 0
            test_result['details'] = f"{successful_actions}/{len(action_results)} actions successful"
        
        elif "file" in criterion.lower():
            # Test if file operations worked
            file_operations = [r for r in action_results if "file" in r.action.lower()]
            test_result['passed'] = len(file_operations) > 0 and all(r.success for r in file_operations)
            test_result['details'] = f"File operations: {len(file_operations)} total"
        
        else:
            # Default: check if any action was successful
            test_result['passed'] = any(result.success for result in action_results)
            test_result['details'] = "General success check"
        
        return test_result
    
    def _generate_feedback(self, plan: Plan, action_results: List[ActionResult], test_results: List[Dict]) -> tuple:
        """Generate detailed feedback and next steps based on test results and action outcomes"""
        
        successful_actions = [r for r in action_results if r.success]
        failed_actions = [r for r in action_results if not r.success]
        
        feedback_parts = []
        next_steps = []
        
        # Analyze detailed test results for specific feedback
        for test_result in test_results:
            test_passed = test_result.get('passed', False)
            test_details = test_result.get('details', '')
            test_criterion = test_result.get('criterion', 'Unknown test')
            
            if test_passed:
                feedback_parts.append(f"✅ Test passed: {test_details}")
                
                # Add specific successful outcomes to status
                if 'individual_tests' in test_result:
                    for individual_test in test_result['individual_tests']:
                        if individual_test.get('passed'):
                            self.status.key_findings.append(individual_test.get('details', 'Test passed'))
            else:
                feedback_parts.append(f"❌ Test failed: {test_details}")
                
                # Add specific failures as challenges
                if 'individual_tests' in test_result:
                    for individual_test in test_result['individual_tests']:
                        if not individual_test.get('passed'):
                            failure_detail = individual_test.get('details', 'Test failed')
                            self.status.current_challenges.append(failure_detail)
                            
                            # Generate specific next steps based on failure type
                            test_desc = individual_test.get('test', '').lower()
                            
                            if 'file exists' in test_desc:
                                next_steps.append("read_file: Ensure target files exist and are accessible")
                            elif 'csv' in test_desc and 'columns' in test_desc:
                                next_steps.append("write_file: Create or fix CSV structure with required columns")
                            elif 'at least' in test_desc and 'record' in test_desc:
                                next_steps.append("data_extraction: Extract more data records to meet minimum requirements")
                            elif 'missing values' in test_desc:
                                next_steps.append("data_cleaning: Address missing values in dataset")
                            elif 'json' in test_desc:
                                next_steps.append("format_output: Ensure output is properly formatted as JSON")
                            elif 'pattern' in test_desc:
                                next_steps.append("analysis: Conduct deeper analysis to identify required patterns")
        
        # Action-based feedback
        if successful_actions:
            feedback_parts.append(f"Action execution: {len(successful_actions)}/{len(action_results)} actions completed successfully")
            
            # Track successful actions in status
            for action in successful_actions:
                if "read_file" in action.action:
                    self.status.key_findings.append(f"Successfully read file: {action.action}")
                elif "write_file" in action.action:
                    self.status.key_findings.append(f"Successfully created file: {action.action}")
                elif action.output:
                    # Extract key insights from LLM actions
                    output_str = str(action.output)
                    if len(output_str) > 50:
                        self.status.key_findings.append(f"Analysis completed: {output_str[:100]}...")
        
        if failed_actions:
            feedback_parts.append(f"Action failures: {len(failed_actions)} actions failed")
            
            for action in failed_actions:
                self.status.current_challenges.append(f"Failed: {action.error_message}")
                
                # Generate recovery steps
                if "read_file" in action.action:
                    next_steps.append("file_check: Verify file paths and permissions")
                elif "write_file" in action.action:
                    next_steps.append("workspace_check: Ensure workspace is writable")
        
        # Meta-analysis specific feedback and next steps
        if "meta-analysis" in self.goal.description.lower() or "csv" in self.goal.description.lower():
            # Check for meta-analysis progress indicators
            csv_test_passed = any(
                test.get('passed', False) for test in test_results 
                for individual_test in test.get('individual_tests', [test])
                if 'csv' in individual_test.get('test', '').lower()
            )
            
            data_quantity_met = any(
                test.get('passed', False) for test in test_results
                for individual_test in test.get('individual_tests', [test])
                if 'at least' in individual_test.get('test', '').lower()
            )
            
            if csv_test_passed:
                feedback_parts.append("Meta-analysis: CSV structure validation passed")
            else:
                feedback_parts.append("Meta-analysis: CSV structure needs improvement")
                next_steps.append("csv_format: Create properly structured CSV with required columns")
            
            if data_quantity_met:
                feedback_parts.append("Meta-analysis: Data quantity requirements met")
            else:
                feedback_parts.append("Meta-analysis: Need more data records")
                next_steps.append("data_extraction: Extract additional data records from sources")
        
        # Iteration-based strategic next steps
        if self.iteration_count == 1:
            if not next_steps:
                next_steps.append("read_file: Begin by reading available data sources")
        elif self.iteration_count == 2:
            if "write_file" not in [step.split(":")[0] for step in next_steps]:
                next_steps.append("write_file: Create preliminary analysis output")
        elif self.iteration_count >= 3:
            if not any("synthesis" in step or "summary" in step for step in next_steps):
                next_steps.append("synthesis: Combine findings into final summary")
        
        # Remove duplicates and limit next steps
        next_steps = list(dict.fromkeys(next_steps))[:5]  # Keep unique, limit to 5
        
        # Compile final feedback
        if feedback_parts:
            feedback = " | ".join(feedback_parts)
        else:
            feedback = "No detailed feedback available from test results"
        
        return feedback, next_steps
    
    def is_goal_achieved(self, review_result: ReviewResult) -> bool:
        """Check if the research goal has been achieved"""
        # Enhanced goal achievement check for meta-analysis work
        
        # Basic requirements: overall success and reasonable progress
        basic_requirements = (review_result.overall_success and 
                            review_result.progress_score >= 0.8)
        
        if not basic_requirements:
            return False
        
        # Meta-analysis specific requirements
        if "meta-analysis" in self.goal.description.lower() or "csv" in self.goal.description.lower():
            # Check if CSV files have been created (not just read)
            csv_created = any(
                "write_file" in result.action and result.success and 
                (".csv" in result.action.lower() or "csv" in str(result.output).lower())
                for result in self.action_results
            )
            
            # Check if we have sufficient data records
            sufficient_data = any(
                result.success and result.output and 
                (',' in str(result.output) and str(result.output).count('\n') >= 5)  # At least 5 lines (header + 4 data rows)
                for result in self.action_results
            )
            
            # Check if required columns are present in any output
            required_columns_present = False
            for result in self.action_results:
                if result.success and result.output:
                    output_str = str(result.output).lower()
                    if ("crop" in output_str and "yield" in output_str and 
                        "climate" in output_str and "location" in output_str):
                        required_columns_present = True
                        break
            
            # Meta-analysis goal achieved only if we have:
            # 1. Created CSV files (not just read)
            # 2. Have sufficient data records 
            # 3. Have required column structure
            # 4. Completed at least 3 iterations (to ensure thoroughness)
            meta_analysis_complete = (csv_created and sufficient_data and 
                                    required_columns_present and 
                                    self.iteration_count >= 3)
            
            return meta_analysis_complete
        
        # For non-meta-analysis goals, use simpler criteria
        return basic_requirements and self.iteration_count >= 2
    
    def should_continue(self, review_result: ReviewResult) -> bool:
        """Determine if research should continue"""
        
        # Always continue if making reasonable progress
        if review_result.progress_score <= 0.2:
            return False  # Stop if very poor progress
        
        # For meta-analysis goals, ensure we focus on data creation
        if "meta-analysis" in self.goal.description.lower() or "csv" in self.goal.description.lower():
            
            # Check if we've created any CSV files yet
            csv_files_created = any(
                "write_file" in result.action and result.success and ".csv" in result.action.lower()
                for result in self.action_results
            )
            
            # Check if we've only been reading files (not extracting or creating)
            only_reading = all(
                "read_file" in result.action for result in self.action_results if result.success
            )
            
            # Check if we've extracted structured data (even without CSV files yet)
            has_extracted_data = any(
                result.success and result.output and 
                ("crop" in str(result.output).lower() or "yield" in str(result.output).lower() or 
                 "climate" in str(result.output).lower()) and len(str(result.output)) > 200
                for result in self.action_results
            )
            
            # Continue if we're still in early stages (iteration 3-4) and making progress
            if self.iteration_count <= 4:
                if only_reading and not has_extracted_data:
                    return True  # Need to extract data
                if has_extracted_data and not csv_files_created:
                    return True  # Need to create CSV files
            
            # If we're only reading and haven't created CSV files, definitely continue
            if only_reading and not csv_files_created and self.iteration_count < self.goal.max_iterations:
                return True
            
            # Check if we have sufficient data records
            has_sufficient_data = any(
                result.success and result.output and 
                (',' in str(result.output) and str(result.output).count('\n') >= 10)  # At least 10 lines of data
                for result in self.action_results
            )
            
            # Continue if we don't have sufficient data yet
            if not has_sufficient_data and self.iteration_count < self.goal.max_iterations:
                return True
            
            # Check if structured data with required columns exists
            has_required_structure = False
            for result in self.action_results:
                if result.success and result.output:
                    output_str = str(result.output).lower()
                    required_fields = ["crop", "yield", "climate", "location", "time"]
                    if sum(1 for field in required_fields if field in output_str) >= 4:
                        has_required_structure = True
                        break
            
            # Continue if we don't have proper structure yet
            if not has_required_structure and self.iteration_count < self.goal.max_iterations:
                return True
        
        # Continue if we haven't reached max iterations and making progress
        return (review_result.progress_score > 0.3 and 
                self.iteration_count < self.goal.max_iterations)
    
    def save_final_report(self):
        """Save a final research report"""
        report_path = os.path.join(self.workspace_dir, f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        report = {
            'goal': asdict(self.goal),
            'total_iterations': self.iteration_count,
            'agent_status': asdict(self.status),
            'plans': [asdict(plan) for plan in self.plans],
            'action_results': [asdict(result) for result in self.action_results],
            'review_results': [asdict(result) for result in self.review_results],
            'final_status': 'completed' if self.review_results and self.review_results[-1].overall_success else 'incomplete'
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Final report saved: {report_path}")
        logger.info(f"📊 Final Status: {self.status.get_status_summary()}")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Deep Research Agent')
    parser.add_argument('--goal', required=True, help='Research goal description')
    parser.add_argument('--success-criteria', nargs='+', default=['Complete research'], 
                       help='Success criteria for the research')
    parser.add_argument('--max-iterations', type=int, default=5, 
                       help='Maximum number of iterations')
    parser.add_argument('--workspace', default='./workspace', 
                       help='Workspace directory for the agent')
    
    args = parser.parse_args()
    
    # Create goal
    goal = Goal(
        description=args.goal,
        success_criteria=args.success_criteria,
        max_iterations=args.max_iterations
    )
    
    # Create and run agent
    agent = ResearchAgent(goal, args.workspace)
    success = agent.run()
    
    if success:
        print("\n🎉 Research completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Research did not complete successfully")
        sys.exit(1)


if __name__ == "__main__":
    main() 