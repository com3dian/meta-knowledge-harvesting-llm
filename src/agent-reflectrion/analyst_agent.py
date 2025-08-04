import os
import json
import datetime
import sys
from typing import List, Dict, Any, Optional, Tuple, TypedDict, Annotated
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    print(f"LangGraph not available: {e}")
    print("Please install with: pip install langgraph")
    LANGGRAPH_AVAILABLE = False

try:
    from llm_impl.geminillm import gemini_complete_if_cache
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"LLM imports not available: {e}")
    LLM_AVAILABLE = False
    
    # Fallback implementations for testing
    def gemini_complete_if_cache(*args, **kwargs):
        # Mock implementation for testing
        return "Mock response: This is a test response for development purposes."

LLM_MODEL_NAME = "gemini-2.5-flash-preview-05-20"

# Simple cache implementation for LLM responses
class SimpleCache:
    def __init__(self):
        self.cache = {}
    
    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, value):
        self.cache[key] = value
    
    @property
    def global_config(self):
        return {
            "enable_llm_cache": True,
            "working_dir": ".",
            "llm_model_name": LLM_MODEL_NAME
        }

# Initialize cache storage
llm_cache = SimpleCache()

def llm_wrapper(prompt, history_messages=None, max_tokens=None, **kwargs):
    """Wrapper function for LLM calls with fallback"""
    if history_messages is None:
        history_messages = []

    # Use Google GenAI or mock response
    if LLM_AVAILABLE and os.environ.get("GEMINI_API_KEY"):
        return gemini_complete_if_cache(
            model=LLM_MODEL_NAME,
            prompt=prompt,
            history_messages=history_messages,
            hashing_kv=llm_cache,
            temperature=0.2,
            max_tokens=max_tokens or 1024,
        )
    else:
        raise ValueError("LLM is not available")

# State definition for LangGraph
class AgentState(TypedDict):
    """State for the meta-analysis agent workflow"""
    # Input data
    input_file: str
    entities_data: Optional[str]
    links_data: Optional[str]
    output_dir: str
    
    # Workflow state
    iteration: int
    max_iterations: int
    current_status: str
    workflow_complete: bool
    
    # Plan-Action-Review cycle
    current_plan: Optional[Dict[str, Any]]
    action_result: Optional[Dict[str, Any]]
    review_result: Optional[Dict[str, Any]]
    previous_results: List[Dict[str, Any]]
    
    # Output
    extracted_records: List[Dict[str, Any]]
    intermediate_files: List[str]
    final_output_file: Optional[str]
    errors: List[str]
    
    # Target format
    target_format: Dict[str, Any]

@dataclass
class Plan:
    """Represents a plan for the next action"""
    action_description: str
    test_criteria: str
    expected_output: str
    iteration: int

@dataclass
class ActionResult:
    """Represents the result of an action"""
    success: bool
    output: Any
    intermediate_files: List[str]
    errors: List[str]
    iteration: int

@dataclass
class ReviewResult:
    """Represents the result of a review"""
    meets_criteria: bool
    feedback: str
    next_action_needed: bool
    suggestions: List[str]

class AnalystAgent:
    """LangGraph-based agent for performing meta-analysis using plan-action-review cycle"""
    
    def __init__(self,
                 input_file: str,
                 output_dir: str = "src/outputs",
                 max_iterations: int = 10,
                 target_format: Dict[str, Any] = None):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph is required but not available. Please install with: pip install langgraph")
        
        self.input_file = input_file
        self.output_dir = output_dir
        self.max_iterations = max_iterations
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Target CSV format
        self.target_format = target_format
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("action", self._action_node)
        workflow.add_node("review", self._review_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Add edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "plan")
        workflow.add_edge("plan", "action")
        workflow.add_edge("action", "review")
        
        # Conditional edge from review
        workflow.add_conditional_edges(
            "review",
            self._should_continue,
            {
                "continue": "plan",
                "finalize": "finalize",
                "end": END
            }
        )
        
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _initialize_node(self, state: AgentState) -> AgentState:
        """Initialize the agent state and load input data"""
        print("Initializing agent...")
        
        try:
            # Load the single JSON input file
            with open(state["input_file"], 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # Extract entities and relationships
            entities = json_data.get("entities", [])
            relationships = json_data.get("relationships", [])

            # Store them as strings in the state
            state["entities_data"] = json.dumps(entities, indent=2)
            state["links_data"] = json.dumps(relationships, indent=2)
            state["current_status"] = "Starting meta-analysis. Need to extract structured records from entities and relationships data."
            
            print("Input data loaded successfully from JSON file")
            
        except Exception as e:
            error_msg = f"Error loading input data from JSON file: {e}"
            print(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _plan_node(self, state: AgentState) -> AgentState:
        """Create a plan for the next action"""
        print(f"\n--- Planning (Iteration {state['iteration'] + 1}) ---")
        
        state["iteration"] += 1
        
        # Create plan prompt
        plan_prompt = f"""
        You are a meta-analysis agent. Your goal is to extract structured records from scientific literature data.
        
        Current Status: {state["current_status"]}
        Iteration: {state["iteration"]}
        
        Previous Results: {json.dumps(state["previous_results"], indent=2)}
        
        MANDATORY Target Format (CSV ONLY):
        Headers: {state["target_format"]["headers"]}
        Example: {state["target_format"]["example"]}
        FOCUS COLUMNS (MUST NOT BE EMPTY): {state["target_format"].get("focus_columns", "None")}
        
        FIELD EXPLANATIONS (CRITICAL - FOLLOW EXACTLY):
        {chr(10).join([f"- {header}: {explanation}" for header, explanation in state["target_format"]["field_explanations"].items()])}
        
        Available Data:
        - Input file: {state["input_file"]}
        
        Current extracted records count: {len(state["extracted_records"])}
        
        Based on the current status, create a plan for the next action. The plan should:
        1. Describe what specific action to take
        2. Define clear test criteria to validate the action (MUST expect CSV format output)
        3. Specify expected output format (MUST be CSV format with the exact headers above)
        
        IMPORTANT CONSTRAINTS:
        - The output MUST always be in CSV format with the specified headers
        - Test criteria MUST validate CSV format compliance AND data completeness
        - Expected output MUST specify CSV format
        - Avoiding hallucination (no records not in input files)
        - PRIORITIZE data linking and cross-referencing when N/As are detected
        
        Return your plan in this format:
        ACTION_DESCRIPTION: [specific action to take]
        TEST_CRITERIA: [how to validate the action was successful - must include CSV format validation]
        EXPECTED_OUTPUT: CSV format with headers: {', '.join(state["target_format"]["headers"])}
        """
        
        response = llm_wrapper(plan_prompt, max_tokens=50000)
        
        # Parse response
        action_description = self._extract_section(response, "ACTION_DESCRIPTION")
        test_criteria = self._extract_section(response, "TEST_CRITERIA")
        expected_output = self._extract_section(response, "EXPECTED_OUTPUT")
        
        # Force CSV format if not specified (safety net)
        if "CSV" not in expected_output.upper() and "csv" not in expected_output:
            expected_output = f"CSV format with headers: {', '.join(state['target_format']['headers'])}"
        
        # Ensure test criteria mentions CSV format validation
        if "CSV" not in test_criteria.upper() and "csv" not in test_criteria:
            test_criteria += " Must validate that output is in proper CSV format with correct headers."
        
        state["current_plan"] = {
            "action_description": action_description,
            "test_criteria": test_criteria,
            "expected_output": expected_output,
            "iteration": state["iteration"]
        }
        
        print(f"Plan created: {action_description}")
        
        return state
    
    def _action_node(self, state: AgentState) -> AgentState:
        """Execute the planned action"""
        print("Executing action...")
        
        plan = state["current_plan"]
        
        try:
            # Create action prompt
            action_prompt = f"""
            Execute the following action:
            {plan["action_description"]}
            
            Available Data:
            
            ENTITIES DATA:
            {state["entities_data"]}...
            
            LINKS DATA:
            {state["links_data"]}...
            
            Target Format:
            Headers: {', '.join(state["target_format"]["headers"])}
            Example: {state["target_format"]["example"]}
            FOCUS COLUMNS (MUST NOT BE EMPTY): {state["target_format"].get("focus_columns", "None")}
            
            FIELD EXPLANATIONS (CRITICAL - FOLLOW EXACTLY):
            {chr(10).join([f"- {header}: {explanation}" for header, explanation in state["target_format"]["field_explanations"].items()])}
            
            Current extracted records: {len(state["extracted_records"])}
            
            ENHANCED EXTRACTION INSTRUCTIONS:
            1. CROSS-REFERENCE entities: Look for related entities that can provide missing information
            2. USE LINKS DATA: Examine relationships between entities to build complete records
            3. COMBINE PARTIAL DATA: If Entity A has crop info but no yield, and Entity B has yield data with similar context, combine them
            4. TRACE SOURCES: Use source/reference information to connect related data points
            5. FILL GAPS STRATEGICALLY: Don't leave fields as N/A if related entities contain the missing information
            6. VALIDATE CONNECTIONS: Ensure combined data makes logical sense (same study, location, timeframe)
            
            DATA LINKING STRATEGY:
            - Step 1: Extract direct, complete records first
            - Step 2: Identify partial records with missing critical fields
            - Step 3: Search for complementary entities that can fill the gaps
            - Step 4: Cross-reference source information to validate connections
            - Step 5: Combine data only when confident of the relationship
            
            QUALITY REQUIREMENTS:
            - ALL records MUST have a value for the FOCUS COLUMNS: {state["target_format"].get("focus_columns", "None")}. Records missing these values will be rejected.
            - Minimize N/A fields through strategic data linking
            - Ensure numerical fields contain only numbers (positive/negative)
            - Each record must be traceable to specific entities
            - Avoid hallucination - only use data present in inputs
            
            Return your results in this format:
            EXTRACTED_RECORDS:
            [CSV formatted records]
            
            VALIDATION_INFO:
            [Information about sources and validation]
            """
            
            response = llm_wrapper(action_prompt, max_tokens=2000)
            
            # Parse response
            extracted_records = self._extract_section(response, "EXTRACTED_RECORDS")
            validation_info = self._extract_section(response, "VALIDATION_INFO")
            
            # Save intermediate results
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            intermediate_file = os.path.join(state["output_dir"], f"intermediate_iter_{state['iteration']}_{timestamp}.csv")
            
            with open(intermediate_file, 'w', newline='', encoding='utf-8') as f:
                f.write(extracted_records)
            
            validation_file = os.path.join(state["output_dir"], f"validation_iter_{state['iteration']}_{timestamp}.txt")
            with open(validation_file, 'w', encoding='utf-8') as f:
                f.write(validation_info)
            
            state["action_result"] = {
                "success": True,
                "output": extracted_records,
                "intermediate_files": [intermediate_file, validation_file],
                "errors": [],
                "iteration": state["iteration"]
            }
            
            state["intermediate_files"].extend([intermediate_file, validation_file])
            
            print(f"Action completed successfully. Files: {[intermediate_file, validation_file]}")
            
        except Exception as e:
            error_msg = f"Error in action execution: {e}"
            print(error_msg)
            state["action_result"] = {
                "success": False,
                "output": None,
                "intermediate_files": [],
                "errors": [error_msg],
                "iteration": state["iteration"]
            }
            state["errors"].append(error_msg)
        
        return state
    
    def _review_node(self, state: AgentState) -> AgentState:
        """Review the action result against the plan criteria"""
        print("Reviewing action results...")
        
        plan = state["current_plan"]
        action_result = state["action_result"]
        
        # Analyze the output for data quality issues
        # data_quality_analysis = self._analyze_output_quality(action_result.get("output", ""))
        
        review_prompt = f"""
        Review the following action result against the plan criteria:
        
        PLAN:
        Action Description: {plan["action_description"]}
        Test Criteria: {plan["test_criteria"]}
        Expected Output: {plan["expected_output"]}
        
        ACTION RESULT:
        Success: {action_result["success"]}
        Output: {action_result["output"] if action_result["output"] else "None"}...
        Errors: {action_result["errors"]}
        
        VALIDATION REQUIREMENTS:
        1. FOCUS COLUMNS CHECK: Every single record MUST have a non-empty value for these columns: {state["target_format"].get("focus_columns", [])}. This is a critical failure point.
        2. No hallucination: All records must be traceable to input data
        3. Data completeness: REJECT if >30% of fields are N/A, empty, or missing in non-focus columns.
        4. Format compliance: Must match the target CSV format exactly
        5. Numerical accuracy: Crop Yield and Climate Drivers Value must be pure numbers
        6. Data linking evidence: If many N/As detected, must show evidence of cross-referencing attempts
        7. Source traceability: Each record must reference specific entities/chunks from input
        
        REJECTION CRITERIA (automatically fail if any apply):
        - ANY record is missing a value in one of the FOCUS COLUMNS: {state["target_format"].get("focus_columns", [])}.
        - More than 30% N/A, empty, or missing fields across all non-focus records
        - Identical output to previous iteration (no improvement)
        - Format violations (non-numeric values in numeric fields)
        - Untraceable data (hallucination suspected)
        
        Original Data Sample:
        {state["entities_data"]}...
        
        Evaluate:
        1. Does the result meet the test criteria?
        2. Are there any hallucination issues?
        3. Is the format correct?
        4. What improvements are needed?
        
        Return your review in this format:
        MEETS_CRITERIA: [Yes/No]
        FEEDBACK: [detailed feedback]
        NEXT_ACTION_NEEDED: [Yes/No]
        SUGGESTIONS: [list of specific suggestions]
        """
        
        response = llm_wrapper(review_prompt, max_tokens=800)
        
        # Parse response
        meets_criteria = self._extract_section(response, "MEETS_CRITERIA").lower().strip() == "yes"
        feedback = self._extract_section(response, "FEEDBACK")
        next_action_needed = self._extract_section(response, "NEXT_ACTION_NEEDED").lower().strip() == "yes"
        suggestions = self._extract_section(response, "SUGGESTIONS").split('\n')
        
        state["review_result"] = {
            "meets_criteria": meets_criteria,
            "feedback": feedback,
            "next_action_needed": next_action_needed,
            "suggestions": [s.strip() for s in suggestions if s.strip()]
        }
        
        # Add to previous results
        state["previous_results"].append(state["action_result"])
        
        # Update current status for next iteration
        state["current_status"] = f"Previous iteration: {feedback}. Suggestions: {'; '.join(state['review_result']['suggestions'])}"
        
        print(f"Review completed: {'Meets criteria' if meets_criteria else 'Needs improvement'}")
        print(f"Feedback: {feedback}")
        
        return state
    
    def _finalize_node(self, state: AgentState) -> AgentState:
        """Finalize the workflow and save final results"""
        print("Finalizing results...")
        
        # Save final results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output = os.path.join(state["output_dir"], f"final_meta_analysis_{timestamp}.csv")
        
        if state["previous_results"]:
            with open(final_output, 'w', encoding='utf-8') as f:
                f.write(state["previous_results"][-1]["output"] or "No valid output generated")
        
        state["final_output_file"] = final_output
        state["workflow_complete"] = True
        
        print(f"Final results saved to: {final_output}")
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if the workflow should continue or end"""
        review_result = state["review_result"]
        
        # Check if max iterations reached
        if state["iteration"] >= state["max_iterations"]:
            print(f"Max iterations ({state['max_iterations']}) reached")
            return "finalize"
        
        # Check if criteria met and no next action needed
        if review_result["meets_criteria"] and not review_result["next_action_needed"]:
            print("Criteria met and no further action needed")
            return "finalize"
        
        # Continue with next iteration
        print("Continuing to next iteration")
        return "continue"
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a specific section from LLM response"""
        try:
            lines = text.split('\n')
            section_content = []
            section_found = False
            section_keywords = ['ACTION_DESCRIPTION', 'TEST_CRITERIA', 'EXPECTED_OUTPUT', 'MEETS_CRITERIA', 'FEEDBACK', 'NEXT_ACTION_NEEDED', 'SUGGESTIONS', 'EXTRACTED_RECORDS', 'VALIDATION_INFO']
            
            for i, line in enumerate(lines):
                # Check if this line contains our target section
                if section_name in line and ':' in line:
                    section_found = True
                    # Extract content from the same line if it exists after the colon
                    colon_index = line.find(':')
                    if colon_index != -1 and colon_index < len(line) - 1:
                        after_colon = line[colon_index + 1:].strip()
                        if after_colon:
                            section_content.append(after_colon)
                    continue
                
                # If we found our section, collect content until we hit another section
                if section_found:
                    # Check if this line starts a new section
                    if any(keyword in line and ':' in line for keyword in section_keywords):
                        break
                    section_content.append(line)
            
            result = '\n'.join(section_content).strip()
            return result
        except Exception as e:
            return ""
    
    def run_meta_analysis(self) -> str:
        """Run the complete meta-analysis using LangGraph workflow"""
        
        # Initialize state
        initial_state = AgentState(
            input_file=self.input_file,
            entities_data=None,
            links_data=None,
            output_dir=self.output_dir,
            iteration=0,
            max_iterations=self.max_iterations,
            current_status="",
            workflow_complete=False,
            current_plan=None,
            action_result=None,
            review_result=None,
            previous_results=[],
            extracted_records=[],
            intermediate_files=[],
            final_output_file=None,
            errors=[],
            target_format=self.target_format
        )
        
        # Run the workflow
        try:
            # Set a higher recursion limit. Each iteration (plan-action-review)
            # consists of multiple steps in the graph. The default limit (25)
            # can be too low for the configured number of iterations.
            # We set it dynamically based on max_iterations.
            config = {"recursion_limit": self.max_iterations * 4 + 5}
            result = self.workflow.invoke(initial_state, config=config)
            
            if result["workflow_complete"]:
                print("\nMeta-analysis completed successfully!")
                return result["final_output_file"]
            else:
                print("\nMeta-analysis terminated without completion")
                return None
                
        except Exception as e:
            print(f"Error running meta-analysis workflow: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function to run the LangGraph meta-analysis agent"""
    
    # Define project root to construct absolute paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print(f"LangGraph Available: {LANGGRAPH_AVAILABLE}")
    print(f"LLM Available: {LLM_AVAILABLE}")
    
    if not LANGGRAPH_AVAILABLE:
        print("ERROR: LangGraph is required but not available.")
        print("Please install with: pip install langgraph")
        return
    
    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("WARNING: GEMINI_API_KEY not found. Running in test mode with mock responses.")
        print("To use real LLM, create a .env file with: GEMINI_API_KEY=your_api_key_here")
    
    # Initialize agent with input files
    try:
        input_file_path = os.path.join(PROJECT_ROOT, "src", "agent-reflectrion", "test_output", "final_entities_20250731_113917.json")
        output_dir_path = os.path.join(PROJECT_ROOT, "src", "agent-reflectrion", "test_output")
        
        # Target CSV format
        target_format = {
            "headers": ["Crop Type", "Crop Yield", "Crop Yield Unit", "Climate Drivers", "Climate Drivers Value", "Climate Drivers Unit", "Experimental Design", "Location", "Time", "Source in paper"],
            "focus_columns": ["Crop Type", "Crop Yield"],
            "example": "maize,2,tons/ha,Atmospheric temperature,+1,℃,InfoCrop-MAIZE model simulations,UIGP,baseline period-1970 to 1995,\"Temperature impact study details\"",
            "field_explanations": {
                "Crop Type": "Name of the crop (e.g., maize, wheat, rice, soybean)",
                "Crop Yield": "NUMERICAL VALUE ONLY. Use positive numbers for yield increases, negative numbers for yield decreases. No text or units.",
                "Crop Yield Unit": "Unit of measurement for crop yield (e.g., tons/ha, kg/ha, Mg/ha, bushels/acre)",
                "Climate Drivers": "Climate variable affecting the crop (e.g., temperature, precipitation, CO2, drought)",
                "Climate Drivers Value": "NUMERICAL VALUE ONLY. Use positive numbers for increases (+1, +2.5), negative numbers for decreases (-1, -0.5). No text or units.",
                "Climate Drivers Unit": "Unit of measurement for climate driver (e.g., °C, mm, ppm, %)",
                "Experimental Design": "Type of study or model used (e.g., field experiment, crop model simulation, greenhouse study)",
                "Location": "Geographic location or region (e.g., country, state, coordinates, study site name)",
                "Time": "Time period or duration of study (e.g., 1990-2000, baseline period, future projection)",
                "Source in paper": "Original text description from the entities or links file that contains the specific data point or evidence"
            }
        }

        agent = AnalystAgent(
            input_file=input_file_path,
            output_dir=output_dir_path,
            max_iterations=10,
            target_format=target_format
        )
        
        # Run meta-analysis
        result_file = agent.run_meta_analysis()
        if result_file:
            print(f"\nMeta-analysis completed. Results saved to: {result_file}")
        else:
            print("\nMeta-analysis failed or was incomplete.")
            
    except Exception as e:
        print(f"Error initializing or running agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 