import os
import json
import datetime
import sys
import csv
import io
from typing import List, Dict, Any, Optional, Tuple, TypedDict, Annotated
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Also add project root to import vendored `external` package
PROJECT_ROOT_TOP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT_TOP)
# Add vendored LightRAG repo path
LIGHTRAG_PATH = os.path.join(PROJECT_ROOT_TOP, "external", "LightRAG")
if LIGHTRAG_PATH not in sys.path:
    sys.path.append(LIGHTRAG_PATH)

from langgraph.graph import StateGraph, END
from llm.llm_wrapper import llm_complete


LLM_MODEL_NAME = "gemini-2.5-flash"

# Use vendored LightRAG's JsonKVStorage for caching
from lightrag.kg.json_kv_impl import JsonKVStorage
from llm.llm_cache import bootstrap_llm_cache

# Determine project root for persistent cache path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(PROJECT_ROOT, "run_artifacts")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "kv_store_llm_response_cache.json")

# Initialize cache storage using LightRAG's JsonKVStorage signature
from lightrag.utils import EmbeddingFunc
import numpy as np

_llm_cache_global_config = {
    "working_dir": CACHE_DIR,
}
# Use namespace so file becomes kv_store_llm_response_cache.json under working_dir
llm_cache = JsonKVStorage(
    namespace="llm_response_cache",
    workspace="",
    global_config=_llm_cache_global_config,
    embedding_func=EmbeddingFunc(embedding_dim=1, func=lambda texts: np.zeros((len(texts), 1))),
)

# Ensure storage is initialized (LightRAG storages are async-initialized)
try:
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        # If a loop is running, schedule initialize and wait
        loop.create_task(llm_cache.initialize())
    except RuntimeError:
        asyncio.run(llm_cache.initialize())
except Exception:
    pass

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
                 target_format: Dict[str, Any] = None,
                 output_type: str = "csv"):  # Add output_type parameter
        
        self.input_file = input_file
        self.output_dir = output_dir
        self.max_iterations = max_iterations
        self.output_type = output_type
        
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
            state["current_status"] = "Starting meta-analysis. Need to extract all reasonable structured records from entities and relationships data."
            
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
        You are a meta-analysis agent. Your goal is to extract ALL REASONABLE structured records from scientific literature data.
        
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
        
        response = llm_complete(plan_prompt, max_tokens=50000)
        
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
    
    def _validate_and_format_csv(self, state: AgentState, csv_text: str, headers: List[str]) -> Tuple[bool, str, List[str]]:
        """
        Validate and format CSV data to ensure it's properly formatted.
        Returns: (is_valid, formatted_csv, error_messages)
        """
        errors = []
        
        # Remove any markdown formatting if present
        csv_text = csv_text.replace('```csv', '').replace('```', '').strip()
        
        # Try to parse the CSV
        try:
            # Read CSV from string
            csv_reader = csv.reader(io.StringIO(csv_text))
            rows = list(csv_reader)
            
            # Validate headers
            if not rows:
                return False, "", ["Empty CSV data"]
                
            file_headers = [h.strip() for h in rows[0]]
            if file_headers != headers:
                errors.append(f"Header mismatch. Expected: {headers}, Got: {file_headers}")
                # Try to fix headers
                rows[0] = headers
            
            # Validate and format each row
            formatted_rows = [headers]  # Start with correct headers
            for i, row in enumerate(rows[1:], 1):
                # Skip completely empty rows
                if not any(cell.strip() for cell in row):
                    continue
                    
                # Pad or truncate row to match header length
                formatted_row = []
                for j, header in enumerate(headers):
                    # Get value or N/A if index out of range
                    value = row[j].strip() if j < len(row) else "N/A"
                    
                    # Convert empty values to N/A
                    if not value:
                        value = "N/A"
                        if header in state["target_format"].get("focus_columns", []):
                            errors.append(f"Focus column {header} cannot be N/A in row {i}")
                    
                    formatted_row.append(value)
                
                formatted_rows.append(formatted_row)
            
            # Write formatted CSV to string
            output = io.StringIO()
            csv_writer = csv.writer(output, lineterminator='\n')
            csv_writer.writerows(formatted_rows)
            formatted_csv = output.getvalue()
            
            return len(errors) == 0, formatted_csv, errors
            
        except Exception as e:
            return False, "", [f"CSV parsing error: {str(e)}"]

    def _action_node(self, state: AgentState) -> AgentState:
        """Execute the planned action"""
        print("Executing action...")
        
        plan = state["current_plan"]
        
        try:
            # Create action prompt based on output type
            if state["target_format"]["output_type"] == "csv":
                action_prompt = self._create_csv_action_prompt(state, plan)
            else:
                action_prompt = self._create_kg_action_prompt(state, plan)
            
            response = llm_complete(action_prompt, max_tokens=4000)
            print(f"LLM Response length: {len(response)}")
            print(f"LLM Response preview: {response[:500]}...")  # First 500 chars
            
            # Parse response based on output type
            if state["target_format"]["output_type"] == "csv":
                extracted_records = self._extract_section(response, "EXTRACTED_RECORDS")
                validation_info = self._extract_section(response, "VALIDATION_INFO")
                
                if not extracted_records.strip():
                    error_msg = "No data extracted from EXTRACTED_RECORDS section"
                    state["action_result"] = {
                        "success": False,
                        "output": None,
                        "intermediate_files": [],
                        "errors": [error_msg],
                        "iteration": state["iteration"]
                    }
                    state["errors"].append(error_msg)
                    return state

                # Validate and format CSV
                is_valid, formatted_csv, validation_errors = self._validate_and_format_csv(state, extracted_records, state["target_format"]["headers"])
                
                if validation_errors:
                    validation_info += "\n\nCSV Validation Errors:\n" + "\n".join(validation_errors)
                
                extracted_records = formatted_csv  # Use the formatted version
                
            else:
                extracted_records = self._extract_section(response, "KNOWLEDGE_GRAPH")
                validation_info = self._extract_section(response, "VALIDATION_INFO")
            
            # Save intermediate results
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_suffix = "csv" if state["target_format"]["output_type"] == "csv" else "json"
            intermediate_file = os.path.join(state["output_dir"], f"intermediate_iter_{state['iteration']}_{timestamp}.{file_suffix}")
            
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

    def _create_csv_action_prompt(self, state: AgentState, plan: Dict[str, Any]) -> str:
        """Create prompt for CSV output format"""
        
        # Calculate how much data to show based on total length
        entities_data = state["entities_data"]
        links_data = state["links_data"]
        
        # Show more data - up to 15000 chars each instead of 5000
        entities_preview = entities_data[:15000] if len(entities_data) > 15000 else entities_data
        links_preview = links_data[:15000] if len(links_data) > 15000 else links_data
        
        # Add truncation warning if data is cut off
        entities_truncated = "... [DATA TRUNCATED - CONTINUE SCANNING SYSTEMATICALLY]" if len(entities_data) > 15000 else ""
        links_truncated = "... [DATA TRUNCATED - CONTINUE SCANNING SYSTEMATICALLY]" if len(links_data) > 15000 else ""
        
        return f"""
        Execute the following action:
        {plan["action_description"]}
        
        Available Data:
        
        ENTITIES DATA (Total length: {len(entities_data)} chars):
        {entities_preview}{entities_truncated}
        
        LINKS/RELATIONSHIPS DATA (Total length: {len(links_data)} chars):  
        {links_preview}{links_truncated}
        
        Target Format:
        Headers: {', '.join(state["target_format"]["headers"])}
        Example: {state["target_format"]["example"]}
        
        CRITICAL: You MUST output all reasonable CSV data rows under EXTRACTED_RECORDS.
        Do NOT use ellipses (...) or placeholders. Output real data rows.
        
        Return your results in this format:
        EXTRACTED_RECORDS:
        {','.join(state["target_format"]["headers"])}
        [Extract ALL possible CSV data rows here - aim for maximum quantity]
        
        VALIDATION_INFO:
        [Information about sources and validation]
        """

    def _create_kg_action_prompt(self, state: AgentState, plan: Dict[str, Any]) -> str:
        """Create prompt for Knowledge Graph output format"""
        kg_template = {
            "nodes": [
                {
                    "id": "unique_id",
                    "type": "node_type",
                    "properties": {}
                }
            ],
            "relationships": [
                {
                    "source": "source_id",
                    "target": "target_id",
                    "type": "relationship_type",
                    "properties": {}
                }
            ]
        }
        
        return f"""
        Execute the following action:
        {plan["action_description"]}
        
        Available Data:
        
        ENTITIES DATA:
        {state["entities_data"]}...
        
        LINKS DATA:
        {state["links_data"]}...
        
        Knowledge Graph Schema:
        Node Types: {json.dumps(state["target_format"]["node_types"], indent=2)}
        Relationships: {json.dumps(state["target_format"]["relationships"], indent=2)}
        Field Mappings: {json.dumps(state["target_format"]["field_mappings"], indent=2)}
        
        FIELD EXPLANATIONS (CRITICAL - FOLLOW EXACTLY):
        {chr(10).join([f"- {field}: {explanation}" for field, explanation in state["target_format"]["field_explanations"].items()])}
        
        Current extracted records: {len(state["extracted_records"])}
        
        ENHANCED EXTRACTION INSTRUCTIONS:
        1. CREATE NODES: Create nodes for each entity type (Crop, Climate, Study)
        2. ESTABLISH RELATIONSHIPS: Connect nodes using defined relationships (AFFECTS, OBSERVED_IN)
        3. VALIDATE PROPERTIES: Ensure all required properties are present for each node
        4. MAINTAIN CONSISTENCY: Use consistent node IDs for references
        5. PRESERVE CONTEXT: Include all relevant properties from the source data
        6. VALIDATE CONNECTIONS: Ensure relationships make logical sense
        
        Return your results in this format:
        KNOWLEDGE_GRAPH:
        {json.dumps(kg_template, indent=2)}
        
        VALIDATION_INFO:
        [Information about sources and validation]
        """

    def _review_node(self, state: AgentState) -> AgentState:
        """Review the action result against the plan criteria"""
        print("Reviewing action results...")
        
        plan = state["current_plan"]
        action_result = state["action_result"]
        
        # Create review prompt based on output type
        if state["target_format"]["output_type"] == "csv":
            review_prompt = self._create_csv_review_prompt(state, plan, action_result)
        else:
            review_prompt = self._create_kg_review_prompt(state, plan, action_result)
        
        response = llm_complete(review_prompt, max_tokens=800)
        
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

    def _create_csv_review_prompt(self, state: AgentState, plan: Dict[str, Any], action_result: Dict[str, Any]) -> str:
        """Create review prompt for CSV output"""
        return f"""
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
        1. FOCUS COLUMNS CHECK: Every single record MUST have a non-empty value for these columns: {state["target_format"].get("focus_columns", [])}
        2. No hallucination: All records must be traceable to input data
        3. Data completeness: REJECT if >30% of fields are N/A, empty, or missing in non-focus columns
        4. Format compliance: Must match the target CSV format exactly
        5. Numerical accuracy: Crop Yield and Climate Drivers Value must be pure numbers
        6. Data linking evidence: If many N/As detected, must show evidence of cross-referencing attempts
        7. Source traceability: Each record must reference specific entities/chunks from input
        
        Return your review in this format:
        MEETS_CRITERIA: [Yes/No]
        FEEDBACK: [detailed feedback]
        NEXT_ACTION_NEEDED: [Yes/No]
        SUGGESTIONS: [list of specific suggestions]
        """

    def _create_kg_review_prompt(self, state: AgentState, plan: Dict[str, Any], action_result: Dict[str, Any]) -> str:
        """Create review prompt for Knowledge Graph output"""
        schema_info = {
            "node_types": state["target_format"]["node_types"],
            "relationships": state["target_format"]["relationships"]
        }
        
        return f"""
        Review the following action result against the plan criteria:
        
        PLAN:
        Action Description: {plan['action_description']}
        Test Criteria: {plan['test_criteria']}
        Expected Output: {plan['expected_output']}
        
        ACTION RESULT:
        Success: {action_result['success']}
        Output: {action_result['output'] if action_result['output'] else 'None'}...
        Errors: {action_result['errors']}
        
        VALIDATION REQUIREMENTS:
        1. NODE COMPLETENESS: All required node types (Crop, Climate, Study) must be present
        2. REQUIRED PROPERTIES: Each node must have all required properties specified in the schema
        3. RELATIONSHIP VALIDITY: Relationships must connect valid node pairs as defined in the schema
        4. No hallucination: All nodes and relationships must be traceable to input data
        5. Data completeness: REJECT if >30% of optional properties are missing
        6. Property type compliance: Numerical properties must contain valid numbers
        7. Source traceability: Each node and relationship must reference specific entities/chunks from input
        
        Knowledge Graph Schema:
        {json.dumps(schema_info, indent=2)}
        
        Return your review in this format:
        MEETS_CRITERIA: [Yes/No]
        FEEDBACK: [detailed feedback]
        NEXT_ACTION_NEEDED: [Yes/No]
        SUGGESTIONS: [list of specific suggestions]
        """
    
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
        """Extract a specific section from LLM response.
        
        This method parses structured LLM responses that contain labeled sections
        separated by colons. It handles both inline content (after the colon on
        the same line) and multi-line content (on subsequent lines until the
        next section starts).
        
        Args:
            text (str): The full LLM response text to parse
            section_name (str): The name of the section to extract (e.g., 'ACTION_DESCRIPTION')
            
        Returns:
            str: The extracted section content, stripped of leading/trailing whitespace.
                 Returns empty string if section not found or on parsing error.
                 
        Examples:
            >>> text = '''
            ... ACTION_DESCRIPTION: Analyze the climate data
            ... This involves processing temperature records
            ... and precipitation patterns.
            ... 
            ... TEST_CRITERIA: Validate data completeness
            ... EXPECTED_OUTPUT: CSV format with headers
            ... '''
            >>> agent._extract_section(text, "ACTION_DESCRIPTION")
            'Analyze the climate data\\nThis involves processing temperature records\\nand precipitation patterns.'
            
            >>> text = "MEETS_CRITERIA: Yes\\nFEEDBACK: Good results"
            >>> agent._extract_section(text, "MEETS_CRITERIA")
            'Yes'
            
            >>> # For EXTRACTED_RECORDS, also handles CSV in markdown code blocks
            >>> text = '''
            ... EXTRACTED_RECORDS:
            ... ```csv
            ... Crop Type,Yield
            ... maize,2.5
            ... wheat,1.8
            ... ```
            ... '''
            >>> agent._extract_section(text, "EXTRACTED_RECORDS")
            'Crop Type,Yield\\nmaize,2.5\\nwheat,1.8'
        """
        try:
            # Try original method first
            lines = text.split('\n')
            section_content = []
            section_found = False
            section_keywords = [
                'ACTION_DESCRIPTION', 'TEST_CRITERIA', 'EXPECTED_OUTPUT',
                'MEETS_CRITERIA', 'FEEDBACK', 'NEXT_ACTION_NEEDED',
                'SUGGESTIONS', 'EXTRACTED_RECORDS', 'VALIDATION_INFO',
                'KNOWLEDGE_GRAPH'
            ]
            
            for i, line in enumerate(lines):
                if section_name in line and ':' in line:
                    section_found = True
                    colon_index = line.find(':')
                    if colon_index != -1 and colon_index < len(line) - 1:
                        after_colon = line[colon_index + 1:].strip()
                        if after_colon:
                            section_content.append(after_colon)
                    continue
                
                if section_found:
                    if any(keyword in line and ':' in line for keyword in section_keywords):
                        break
                    section_content.append(line)
            
            result = '\n'.join(section_content).strip()
            
            # If empty and looking for EXTRACTED_RECORDS, try fallback
            if not result and section_name == "EXTRACTED_RECORDS":
                # Look for CSV blocks in markdown fences
                import re
                csv_blocks = re.findall(r'```(?:csv)?\n(.*?)\n```', text, re.DOTALL)
                if csv_blocks:
                    result = csv_blocks[0].strip()
            
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
    bootstrap_llm_cache()

    # Define project root to construct absolute paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("WARNING: GEMINI_API_KEY not found. Running in test mode with mock responses.")
        print("To use real LLM, create a .env file with: GEMINI_API_KEY=your_api_key_here")
    
    # Initialize agent with input files
    try:
        # get the date and time as output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        input_file_path = os.path.join(PROJECT_ROOT, "src", "agent_reflection", "test_output", "final_entities_20250731_113917.json")
        output_dir_path = os.path.join(PROJECT_ROOT, "run_artifacts", "analyst_agent", timestamp)
        
        # Target format schema
        target_format = {
            "headers": ["Crop Type", "Crop Yield", "Crop Yield Unit", "Climate Drivers", "Climate Drivers Value", "Climate Drivers Unit", "Experimental Design", "Location", "Time", "Source in paper"],
            "focus_columns": ["Crop Type", "Crop Yield", "Source in paper"],
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
            },
            "output_type": "csv"
        }

        # Run analysis with CSV output
        print("\nRunning analysis with CSV output...")
        agent_csv = AnalystAgent(
            input_file=input_file_path,
            output_dir=output_dir_path,
            max_iterations=10,
            target_format=target_format,
            output_type="csv"
        )
        result_file_csv = agent_csv.run_meta_analysis()
        
        # Run analysis with Knowledge Graph output
        print("\nRunning analysis with Knowledge Graph output...")
        # agent_kg = AnalystAgent(
        #     input_file=input_file_path,
        #     output_dir=output_dir_path,
        #     max_iterations=10,
        #     target_format=target_format,
        #     output_type="knowledge_graph"
        # )
        # result_file_kg = agent_kg.run_meta_analysis()
        
        # # Report results
        # if result_file_csv:
        #     print(f"\nCSV meta-analysis completed. Results saved to: {result_file_csv}")
        # else:
        #     print("\nCSV meta-analysis failed or was incomplete.")

        # if result_file_kg:
        #     print(f"\nKnowledge Graph meta-analysis completed. Results saved to: {result_file_kg}")
        # else:
        #     print("\nKnowledge Graph meta-analysis failed or was incomplete.")
    except Exception as e:
        print(f"Error initializing or running agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


"""
now I want you to make another agent, the manager, which put together the entity recognition agent and analyst agent, which takes a question as input, with a  extra `source` of evidence (which could be empty, but if empty then use web search, foe web search create a dummy method only, I will choose between methods). and output a answer for the question, firstly call the entity recognition agent with target schema found by manager, and call the analyst agent with corresponding format 

"""