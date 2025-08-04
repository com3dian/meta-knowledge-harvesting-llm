import os
import json
import csv
import datetime
import sys
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
# LLM_MODEL_NAME = "gemini-2.5-flash-lite-preview-06-17"

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
        # Mock response for testing
        if "ACTION_DESCRIPTION" in prompt:
            return """ACTION_DESCRIPTION: Extract crop yield records from the entities data focusing on maize crops with numerical values
TEST_CRITERIA: Records must contain crop type, numerical yield values (positive for increases, negative for decreases), climate drivers with numerical values, and locations that can be traced back to the input data. CSV format validation required.
EXPECTED_OUTPUT: CSV format with headers: Crop type,Crop yield,Crop Yield Unit,Climate drivers,Climate Drivers Value,Climate Drivers Unit,Experimental design,Location,Time,Source in paper"""
        elif "EXTRACTED_RECORDS" in prompt:
            return """EXTRACTED_RECORDS:
Crop type,Crop yield,Crop Yield Unit,Climate drivers,Climate Drivers Value,Climate Drivers Unit,Experimental design,Location,Time,Source in paper
maize,2,Mg/ha,Atmospheric temperature,+1,°C,InfoCrop-MAIZE model simulations,UIGP,baseline period,chunk-110b2a3996a446fc39fb457d4214d315
wheat,-0.5,tons/ha,precipitation,-20,mm,field experiment,Nebraska,2010-2015,Table 3

VALIDATION_INFO:
Records extracted from entities data with proper numerical format. Positive values indicate increases, negative values indicate decreases in both crop yield and climate drivers."""
        elif "MEETS_CRITERIA" in prompt:
            return """MEETS_CRITERIA: Yes
FEEDBACK: Successfully extracted valid records from the input data with proper format and traceability
NEXT_ACTION_NEEDED: No
SUGGESTIONS: Consider extracting more records if available in the data"""
        else:
            return "Mock response: This is a test response for development purposes."

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

class MetaAnalysisAgent:
    """Agent for performing meta-analysis using plan-action-review cycle"""
    
    def __init__(self, entities_file: str, links_file: str, output_dir: str = "src/outputs"):
        self.entities_file = entities_file
        self.links_file = links_file
        self.output_dir = output_dir
        self.entities_data = None
        self.links_data = None
        self.extracted_records = []
        self.iteration = 0
        self.max_iterations = 10
        
        # Target CSV format
        self.target_format = {
            "headers": ["Crop Type", "Crop Yield", "Crop Yield Unit", "Climate Drivers", "Climate Drivers Value", "Climate Drivers Unit", "Experimental Design", "Location", "Time", "Source in paper"],
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
                "Source in paper": "Reference to specific section, figure, table, or page in the source document"
            }
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def load_input_data(self) -> bool:
        """Load entities and links data from files"""
        try:
            # Load entities
            with open(self.entities_file, 'r', encoding='utf-8') as f:
                entities_content = f.read()
            
            # Load links
            with open(self.links_file, 'r', encoding='utf-8') as f:
                links_content = f.read()
            
            self.entities_data = entities_content
            self.links_data = links_content
            return True
            
        except Exception as e:
            print(f"Error loading input data: {e}")
            return False
    
    def plan(self, current_status: str, previous_results: Optional[List[ActionResult]] = None) -> Plan:
        """Create a plan for the next action"""
        self.iteration += 1
        
        # Analyze current status and create plan
        plan_prompt = f"""
        You are a meta-analysis agent. Your goal is to extract structured records from scientific literature data.
        
        Current Status: {current_status}
        Iteration: {self.iteration}
        
        Previous Results: {json.dumps([r.__dict__ for r in previous_results] if previous_results else [], indent=2)}
        
        MANDATORY Target Format (CSV ONLY):
        Headers: {self.target_format["headers"]}
        Example: {self.target_format["example"]}
        
        FIELD EXPLANATIONS (CRITICAL - FOLLOW EXACTLY):
        {chr(10).join([f"- {header}: {explanation}" for header, explanation in self.target_format["field_explanations"].items()])}
        
        Available Data:
        - Entities file: {self.entities_file}
        - Links file: {self.links_file}
        
        Current extracted records count: {len(self.extracted_records)}
        
        Based on the current status, create a plan for the next action. The plan should:
        1. Describe what specific action to take
        2. Define clear test criteria to validate the action (MUST expect CSV format output)
        3. Specify expected output format (MUST be CSV format with the exact headers above)
        
        IMPORTANT CONSTRAINTS:
        - The output MUST always be in CSV format with the specified headers
        - Test criteria MUST validate CSV format compliance
        - Expected output MUST specify CSV format
        - Avoiding hallucination (no records not in input files)
        - Extracting as many factual records as possible
        
        Return your plan in this format:
        ACTION_DESCRIPTION: [specific action to take]
        TEST_CRITERIA: [how to validate the action was successful - must include CSV format validation]
        EXPECTED_OUTPUT: CSV format with headers: {', '.join(self.target_format["headers"])}
        """
        
        response = llm_wrapper(plan_prompt, max_tokens=500)
        
        # Parse response
        action_description = self._extract_section(response, "ACTION_DESCRIPTION")
        test_criteria = self._extract_section(response, "TEST_CRITERIA")
        expected_output = self._extract_section(response, "EXPECTED_OUTPUT")
        
        # Force CSV format if not specified (safety net)
        if "CSV" not in expected_output.upper() and "csv" not in expected_output:
            expected_output = f"CSV format with headers: {', '.join(self.target_format['headers'])}"
        
        # Ensure test criteria mentions CSV format validation
        if "CSV" not in test_criteria.upper() and "csv" not in test_criteria:
            test_criteria += " Must validate that output is in proper CSV format with correct headers."
        
        return Plan(
            action_description=action_description,
            test_criteria=test_criteria,
            expected_output=expected_output,
            iteration=self.iteration
        )
    
    def action(self, plan: Plan) -> ActionResult:
        """Execute the planned action"""
        try:
            # Create action prompt
            action_prompt = f"""
            Execute the following action:
            {plan.action_description}
            
            Available Data:
            
            ENTITIES DATA:
            {self.entities_data}...
            
            LINKS DATA:
            {self.links_data}...
            
            Target Format:
            Headers: {', '.join(self.target_format["headers"])}
            Example: {self.target_format["example"]}
            
            FIELD EXPLANATIONS (CRITICAL - FOLLOW EXACTLY):
            {chr(10).join([f"- {header}: {explanation}" for header, explanation in self.target_format["field_explanations"].items()])}
            
            Current extracted records: {len(self.extracted_records)}
            
            Instructions:
            1. Extract records ONLY from the provided data
            2. Do not hallucinate or create data not present in the input
            3. Format exactly as specified
            4. Return results in CSV format
            5. Include source information for validation
            
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
            intermediate_file = os.path.join(self.output_dir, f"intermediate_iter_{self.iteration}_{timestamp}.csv")
            
            with open(intermediate_file, 'w', newline='', encoding='utf-8') as f:
                f.write(extracted_records)
            
            validation_file = os.path.join(self.output_dir, f"validation_iter_{self.iteration}_{timestamp}.txt")
            with open(validation_file, 'w', encoding='utf-8') as f:
                f.write(validation_info)
            
            return ActionResult(
                success=True,
                output=extracted_records,
                intermediate_files=[intermediate_file, validation_file],
                errors=[],
                iteration=self.iteration
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                output=None,
                intermediate_files=[],
                errors=[str(e)],
                iteration=self.iteration
            )
    
    def review(self, plan: Plan, action_result: ActionResult) -> ReviewResult:
        """Review the action result against the plan criteria"""
        
        review_prompt = f"""
        Review the following action result against the plan criteria:
        
        PLAN:
        Action Description: {plan.action_description}
        Test Criteria: {plan.test_criteria}
        Expected Output: {plan.expected_output}
        
        ACTION RESULT:
        Success: {action_result.success}
        Output: {action_result.output if action_result.output else "None"}...
        Errors: {action_result.errors}
        
        VALIDATION REQUIREMENTS:
        1. No hallucination: All records must be traceable to input data
        2. Completeness: Extract as many factual records as possible
        3. Format compliance: Must match the target CSV format
        
        Original Data Sample:
        {self.entities_data}...
        
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
        
        return ReviewResult(
            meets_criteria=meets_criteria,
            feedback=feedback,
            next_action_needed=next_action_needed,
            suggestions=[s.strip() for s in suggestions if s.strip()]
        )
    
    def run_meta_analysis(self) -> str:
        """Run the complete meta-analysis using plan-action-review cycle"""
        
        # Load input data
        if not self.load_input_data():
            return "Failed to load input data"
        
        current_status = "Starting meta-analysis. Need to extract structured records from entities and links data."
        previous_results = []
        
        while self.iteration < self.max_iterations:
            print(f"\n--- Iteration {self.iteration + 1} ---")
            
            # PLAN
            plan = self.plan(current_status, previous_results)
            print(f"Plan: {plan.action_description}")
            
            # ACTION
            action_result = self.action(plan)
            print(f"Action result: {'Success' if action_result.success else 'Failed'}")
            if action_result.intermediate_files:
                print(f"Intermediate files: {action_result.intermediate_files}")
            
            # REVIEW
            review_result = self.review(plan, action_result)
            print(f"Review: {'Meets criteria' if review_result.meets_criteria else 'Needs improvement'}")
            print(f"Feedback: {review_result.feedback}")
            
            # Update for next iteration
            previous_results.append(action_result)
            
            if review_result.meets_criteria and not review_result.next_action_needed:
                print("Meta-analysis completed successfully!")
                break
            
            # Update current status for next iteration
            current_status = f"Previous iteration: {review_result.feedback}. Suggestions: {'; '.join(review_result.suggestions)}"
        
        # Save final results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output = os.path.join(self.output_dir, f"final_meta_analysis_{timestamp}.csv")
        
        if previous_results:
            with open(final_output, 'w', encoding='utf-8') as f:
                f.write(previous_results[-1].output or "No valid output generated")
        
        return final_output
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a specific section from LLM response"""
        try:
            lines = text.split('\n')
            section_start = None
            section_content = []
            
            for i, line in enumerate(lines):
                if section_name in line:
                    section_start = i
                    break
            
            if section_start is not None:
                for line in lines[section_start + 1:]:
                    if line.strip() and not line.startswith(section_name) and ':' in line and any(keyword in line for keyword in ['ACTION_DESCRIPTION', 'TEST_CRITERIA', 'EXPECTED_OUTPUT', 'MEETS_CRITERIA', 'FEEDBACK', 'NEXT_ACTION_NEEDED', 'SUGGESTIONS', 'EXTRACTED_RECORDS', 'VALIDATION_INFO']):
                        break
                    section_content.append(line)
            
            return '\n'.join(section_content).strip()
        except:
            return ""

def main():
    """Main function to run the meta-analysis agent
    
    To use with real LLM:
    1. Create a .env file with: GEMINI_API_KEY=your_api_key_here
    2. Get your key from: https://ai.google.dev/
    """
    
    global LLM_AVAILABLE
    
    print(f"LLM Available: {LLM_AVAILABLE}")
    
    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("WARNING: GEMINI_API_KEY not found. Running in test mode with mock responses.")
        print("To use real LLM, create a .env file with: GEMINI_API_KEY=your_api_key_here")
        # Switch to mock mode
        LLM_AVAILABLE = False
    
    # Initialize agent with input files
    agent = MetaAnalysisAgent(
        entities_file="/home/com3dian/Github/meta-knowledge-harvesting-llm/src/outputs/_result_20250713_093148_entities.txt",
        links_file="/home/com3dian/Github/meta-knowledge-harvesting-llm/src/outputs/_result_20250713_093148_links.txt",
        output_dir="../outputs"
    )
    
    # Run meta-analysis
    try:
        result_file = agent.run_meta_analysis()
        print(f"\nMeta-analysis completed. Results saved to: {result_file}")
    except Exception as e:
        print(f"Error running meta-analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()