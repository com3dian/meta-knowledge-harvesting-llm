#!/usr/bin/env python3
"""
Utilities for the research agent
"""

import json
import re
from typing import Dict, Any, List, Optional, Callable
from research_agent_data_types import Plan, ActionResult
from datetime import datetime

def execute_llm_for_plan(llm_wrapper: Callable, prompt: str) -> Dict[str, Any]:
    """Execute LLM for plan generation and parse JSON response"""
    try:
        response = llm_wrapper(prompt)
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        
        # Fallback: try to parse the entire response as JSON
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract structured data
            return extract_plan_from_text(response)
            
    except Exception as e:
        raise ValueError(f"Failed to parse LLM response: {str(e)}\nResponse: {response}")

def extract_plan_from_text(response: str) -> Dict[str, Any]:
    """Fallback method to extract plan data from unstructured text"""
    # Try to find plan components in the text
    plan_data = {
        "description": "",
        "action": "",
        "expected_outcome": "",
        "test_criteria": ""
    }
    
    lines = response.strip().split('\n')
    current_field = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for field indicators
        if 'description:' in line.lower():
            current_field = 'description'
            plan_data[current_field] = line.split(':', 1)[1].strip()
        elif 'action:' in line.lower():
            current_field = 'action'
            plan_data[current_field] = line.split(':', 1)[1].strip()
        elif 'expected:' in line.lower() or 'outcome:' in line.lower():
            current_field = 'expected_outcome'
            plan_data[current_field] = line.split(':', 1)[1].strip()
        elif 'test:' in line.lower() or 'criteria:' in line.lower():
            current_field = 'test_criteria'
            plan_data[current_field] = line.split(':', 1)[1].strip()
        elif current_field and line:
            # Continue adding to current field
            plan_data[current_field] += " " + line
    
    # Validate that we have the essential fields
    if not plan_data["action"]:
        raise ValueError("Could not extract action from LLM response")
    
    return plan_data

def execute_llm_action(llm_wrapper: Callable, action: str, context: Dict[str, Any]) -> ActionResult:
    """Execute non-tool actions using LLM wrapper with JSON response parsing"""
    try:
        # Create action prompt
        action_prompt = f"""
        You are a research agent executing an action. 
        
        Current Context:
        - Goal: {context.get('goal', 'Not specified')}
        - Success Criteria: {context.get('success_criteria', 'Not specified')}
        - Current Iteration: {context.get('iteration', 'Unknown')}
        - Workspace Files: {', '.join(context.get('workspace_files', [])) if context.get('workspace_files') else 'None'}
        - Key Findings: {'; '.join(context.get('key_findings', [])[-3:]) if context.get('key_findings') else 'None'}
        
        Previous Action Results:
        {context.get('previous_results_context', 'No previous results available')}
        
        Action to Execute: {action}
        
        You can reference data from previous actions using the context above.
        
        Please execute this action and respond with ONLY a JSON object in this exact format:
        {{
            "success": true/false,
            "output": "detailed description of what was accomplished or results produced",
            "error_message": "error details if failed, or empty string if successful"
        }}
        
        Examples:
        {{"success": true, "output": "Successfully analyzed climate data and identified 15 key patterns in temperature trends", "error_message": ""}}
        {{"success": false, "output": "", "error_message": "Missing required data fields in the input"}}
        
        JSON Response:
        """
        
        # Get LLM response
        response = llm_wrapper(action_prompt)
        
        # Parse JSON response
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group(0))
            else:
                result_data = json.loads(response.strip())
            
            return ActionResult(
                action=action,
                success=result_data.get('success', False),
                output=result_data.get('output') if result_data.get('success') else None,
                error_message=result_data.get('error_message') if not result_data.get('success') else None
            )
            
        except json.JSONDecodeError as e:
            # Fallback parsing for non-JSON responses
            return parse_action_result_fallback(action, response)
                
    except Exception as e:
        return ActionResult(
            action=action,
            success=False,
            output=None,
            error_message=f"Error executing LLM action: {str(e)}"
        )

def parse_action_result_fallback(action: str, response: str) -> ActionResult:
    """Fallback parsing for action results when JSON parsing fails"""
    # Try to determine success from response content
    response_lower = response.lower()
    
    # Look for success indicators
    success_indicators = ['success', 'completed', 'accomplished', 'found', 'extracted', 'created']
    failure_indicators = ['error', 'failed', 'could not', 'unable to', 'missing']
    
    success_score = sum(1 for indicator in success_indicators if indicator in response_lower)
    failure_score = sum(1 for indicator in failure_indicators if indicator in response_lower)
    
    # Determine success based on indicators
    success = success_score > failure_score and len(response.strip()) > 10
    
    return ActionResult(
        action=action,
        success=success,
        output=response if success else None,
        error_message=response if not success else None
    )

def run_function_criterion(func, action_results, *args, **kwargs):
    """
    Run a function-based criterion for testing
    """
    try:
        return func(action_results, *args, **kwargs)
    except Exception as e:
        return False, f"Function criterion failed: {str(e)}"

def run_prompt_criterion(llm_wrapper, prompt_template, action_results, context=None):
    """
    Run a prompt-based criterion using LLM
    """
    try:
        # Format prompt with action results
        formatted_prompt = prompt_template.format(
            action_results=action_results,
            context=context or {}
        )
        
        response = llm_wrapper(formatted_prompt)
        
        # Simple parsing for success/failure
        response_lower = response.lower()
        success = ('success' in response_lower or 'true' in response_lower or 
                  'pass' in response_lower) and 'fail' not in response_lower
        
        return success, response
        
    except Exception as e:
        return False, f"Prompt criterion failed: {str(e)}"

def create_plan_generation_prompt(
    goal: str,
    success_criteria: str,
    iteration_count: int,
    status_context: str,
    continuation_context: Dict[str, Any],
    previous_test_context: str,
    tools_description: Dict[str, str],
    workspace_files: List[str]
) -> str:
    """Create a simplified, focused prompt for plan generation"""
    
    # Check for intermediate file from previous iteration to set the priority
    previous_iter_file = f"intermediate_output_iter{iteration_count - 1}.txt"
    completed_steps = continuation_context.get('completed_steps', [])

    if previous_iter_file in workspace_files:
        priority = f"An intermediate file '{previous_iter_file}' was just created. Your action for this iteration MUST be to read this file using the read_file tool so its content can be processed."
        iteration_guidance = "Focus: Read the intermediate file to prepare for final CSV creation."
    
    # If no intermediate file, check if the last action was reading a file, which now needs to be written.
    elif completed_steps and "read_file" in completed_steps[-1]:
        priority = "The last step was reading a file. Now, write its contents to a final CSV file using the write_file tool."
        iteration_guidance = "Focus: Create the final CSV file from the loaded content."

    else:
        # Existing iteration-based guidance
        if iteration_count <= 2:
            iteration_guidance = "Focus: Read and analyze source data files to understand what's available."
            priority = "Read key source files."
        elif iteration_count == 3:
            iteration_guidance = "Focus: Extract and structure meta-analysis data. The system will save your output to an intermediate file."
            priority = "Extract specific data records from loaded files and format them. Do not try to write a file yet."
        else: # Iteration 4+
            if any(f.startswith("intermediate_output") for f in workspace_files):
                 priority = "An intermediate file exists but wasn't read. Read it now."
                 iteration_guidance = "Focus: Read existing intermediate file."
            else:
                 priority = "Validate, refine, and complete meta-analysis. If data has been analyzed, create the CSV file."
                 iteration_guidance = "Focus: Create final CSV or refine existing data."

    prompt = f"""
    Create a research plan for iteration {iteration_count}.

    GOAL: {goal}
    
    CURRENT STATUS:
    {status_context}
    
    GUIDANCE:
    - Current Focus: {iteration_guidance}
    - Highest Priority: {priority}
    
    AVAILABLE TOOLS:
    {json.dumps(tools_description, indent=2)}
    
    IMPORTANT WORKFLOW:
    1. First, ANALYZE data. The system will automatically save large outputs to an `intermediate_output_iterX.txt` file.
    2. Next, you MUST READ that intermediate file using `||read_file(intermediate_output_iterX.txt)||`.
    3. Finally, you MUST WRITE the data to a final CSV file using `||write_file(final_report.csv, placeholder_from_read_step)||`.
    
    NEVER skip the READ step. Always READ before you WRITE.
    
    RESPOND WITH ONLY THIS JSON FORMAT:
    {{
        "description": "Brief plan description",
        "action": "Specific action to take",
        "expected_outcome": "What you expect to accomplish",
        "test_criteria": "TEST1: specific test|TEST2: another test"
    }}
    
    JSON Response:
    """
    
    return prompt