import os
import json
import datetime
import sys
from typing import List, Dict, Any, Optional, Tuple, TypedDict, Annotated
from dataclasses import dataclass
from dotenv import load_dotenv
import networkx as nx

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

# State definition for the graph
class EntityAgentState(TypedDict):
    """State for the entity recognition agent."""
    input_text: str
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
    extracted_entities: List[Dict[str, Any]]
    extracted_relationships: List[Dict[str, Any]]
    intermediate_files: List[str]
    final_output_file: Optional[str]
    errors: List[str]
    target_schema: Optional[Dict[str, Any]]

@dataclass
class Plan:
    """Represents a plan for the next action."""
    action_description: str
    test_criteria: str
    expected_output: str
    iteration: int

@dataclass
class ActionResult:
    """Represents the result of an action."""
    success: bool
    output: Any
    intermediate_files: List[str]
    errors: List[str]
    iteration: int

@dataclass
class ReviewResult:
    """Represents the result of a review."""
    meets_criteria: bool
    feedback: str
    next_action_needed: bool
    suggestions: List[str]


class EntityRecognitionAgent:
    def __init__(self, output_dir: str = "src/outputs", max_iterations: int = 5, ontology_graph: Optional[nx.Graph] = None, target_schema: Optional[Dict[str, Any]] = None):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph is required but not available.")
        
        self.output_dir = output_dir
        self.max_iterations = max_iterations
        self.ontology_graph = ontology_graph
        self.target_schema = target_schema
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Builds the LangGraph workflow for entity recognition."""
        workflow = StateGraph(EntityAgentState)
        
        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("action", self._action_node)
        workflow.add_node("review", self._review_node)
        workflow.add_node("finalize", self._finalize_node)
        
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "plan")
        workflow.add_edge("plan", "action")
        workflow.add_edge("action", "review")
        
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

    def _initialize_node(self, state: EntityAgentState) -> EntityAgentState:
        print("Initializing Entity Recognition Agent...")
        state["current_status"] = "Starting entity recognition. Input text is ready for processing."
        state["iteration"] = 0
        state["previous_results"] = []
        state["errors"] = []
        return state

    def _plan_node(self, state: EntityAgentState) -> EntityAgentState:
        print(f"\n--- Planning (Iteration {state['iteration'] + 1}) ---")
        state["iteration"] += 1
        
        previous_summary = []
        if state["previous_results"]:
            last_result = state["previous_results"][-1]
            if last_result and last_result.get("output"):
                num_entities = len(last_result["output"].get("entities", []))
                num_rels = len(last_result["output"].get("relationships", []))
                previous_summary.append(f"In the last iteration ({last_result['iteration']}), you extracted {num_entities} entities and {num_rels} relationships.")
            else:
                previous_summary.append(f"The last iteration ({last_result['iteration']}) failed or produced no output.")
        
        summary_text = "\n".join(previous_summary)

        plan_prompt = f"""
        You are an agent that extracts structured data from text based on a defined schema.

        Your Goal: To extract all relevant information from the provided text that matches the entity types defined in the target schema.
        If this is not the first iteration, your goal is to refine the previous extraction by adding missing information or correcting errors based on the review feedback.

        Target Schema (for entities):
        {json.dumps(state.get('target_schema', {'info': 'No schema provided, perform open extraction.'}), indent=2)}
        
        Relationship Extraction Goal:
        Even though the schema does not define specific relationship types, you MUST extract relationships between entities. A relationship should connect two entities that are discussed together in the text and have a clear connection. For the relationship's "description", use the sentence from the text that shows their connection.

        Current Status: {state["current_status"]}
        Iteration: {state["iteration"]}
        
        Summary of Last Iteration:
        {summary_text}
        
        Plan for the next action:
        **CRITICAL CONSTRAINT**: The output format MUST ALWAYS be a JSON object with two keys: "entities" and "relationships". Do NOT propose to restructure the output into a different format. Your plan must focus ONLY on adding, removing, or correcting items within the "entities" and "relationships" lists.

        1.  **Action Description**: If iteration 1, describe a plan for a full extraction of both entities (based on the schema) and relationships (based on contextual links). If iteration > 1, describe a plan to refine the previous extraction by adding missing items or correcting errors in the existing entities and relationships.
        2.  **Test Criteria**: How will you validate that the extracted entities and relationships are correct according to the text and the target schema?
        3.  **Expected Output**: Confirm that the output will be a single JSON object with two keys: "entities" and "relationships".
        
        Return your plan in this format:
        ACTION_DESCRIPTION: [specific action to take]
        TEST_CRITERIA: [how to validate the action]
        EXPECTED_OUTPUT: [description of the expected output format, confirming it will be a JSON with "entities" and "relationships" keys]
        """
        
        response = llm_wrapper(plan_prompt, max_tokens=1000)
        
        action_description = self._extract_section(response, "ACTION_DESCRIPTION")
        test_criteria = self._extract_section(response, "TEST_CRITERIA")
        expected_output = self._extract_section(response, "EXPECTED_OUTPUT")
        
        state["current_plan"] = {
            "action_description": action_description,
            "test_criteria": test_criteria,
            "expected_output": expected_output,
            "iteration": state["iteration"]
        }
        
        print(f"Plan created: {action_description}")
        return state

    def _action_node(self, state: EntityAgentState) -> EntityAgentState:
        print("Executing action...")
        plan = state["current_plan"]
        
        try:
            action_prompt = f"""
            Based on the following plan, extract entities and relationships from the input text.
            
            Action: {plan['action_description']}

            You must extract entities matching the following types: {json.dumps(list(state.get('target_schema', {}).keys()))}
            Here are the descriptions for each entity type:
            {json.dumps(state.get('target_schema', {}), indent=2)}
            
            Input Text:
            ---
            {state['input_text']}
            ---
            
            **CRITICAL INSTRUCTION FOR DESCRIPTIONS**:
            For each extracted entity and relationship, the "description" field MUST be one or two **full sentences** copied **exactly as they appear** from the input text.
            These sentences should best describe the entity's attributes or the relationship.
            **Do not summarize, paraphrase, or alter the original text in any way. Copy it verbatim.**

            **STRICT OUTPUT FORMAT**:
            Your output MUST be a single JSON object.
            The object must have two keys: "entities" and "relationships".
            - Each object in the "entities" list must have the keys: "name", "type", and "description".
            - Each object in the "relationships" list must have the keys: "source", "target", and "description".

            Please extract the entities and relationships in a structured JSON format.
            
            Example of the expected JSON format:
            {{
              "entities": [
                {{
                  "name": "Entity1",
                  "type": "Type1",
                  "description": "This is a description of Entity1 with its key characteristics."
                }},
                {{
                  "name": "Entity2",
                  "type": "Type2",
                  "description": "Entity2 is related to some measurements and has specific attributes."
                }},
                {{
                  "name": "Entity3",
                  "type": "Type3",
                  "description": "Entity3 represents a specific measurement or value in the context."
                }}
              ],
              "relationships": [
                {{
                  "source": "Entity1",
                  "target": "Entity2",
                  "description": "Entity1 and Entity2 are connected through this specific relationship described in the text."
                }}
              ]
            }}
            
            Return ONLY the JSON object.
            """
            
            response = llm_wrapper(action_prompt, max_tokens=4000)
            
            # Clean the response to ensure it's valid JSON
            json_response = self._clean_json_response(response)
            initial_extraction = json.loads(json_response)

            # Defensive check: if LLM returns a list, wrap it in a dict
            if isinstance(initial_extraction, list):
                initial_extraction = {"entities": initial_extraction, "relationships": []}
            
            extracted_entities = initial_extraction.get("entities", [])
            extracted_relationships = initial_extraction.get("relationships", [])

            # --- Ontology-based inference starts here ---
            if self.ontology_graph:
                if state["iteration"] > 1:
                    print("Skipping ontology-based inference for iteration > 1")
                else:
                    print("Performing ontology-based inference...")
                    inferred_placeholders = self._infer_from_ontology(extracted_entities)
                
                    if inferred_placeholders.get("entities") or inferred_placeholders.get("relationships"):
                        filled_inferred = self._fill_inferred_with_llm(inferred_placeholders, state['input_text'])
                        
                        extracted_entities.extend(filled_inferred.get("entities", []))
                        extracted_relationships.extend(filled_inferred.get("relationships", []))
            # --- Ontology-based inference ends ---

            final_output = {
                "entities": extracted_entities,
                "relationships": extracted_relationships
            }
            
            # Save intermediate results
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            intermediate_file = os.path.join(self.output_dir, f"entities_iter_{state['iteration']}_{timestamp}.json")
            
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, indent=2)
            
            state["action_result"] = {
                "success": True,
                "output": final_output,
                "intermediate_files": [intermediate_file],
                "errors": [],
                "iteration": state["iteration"]
            }
            
            state["intermediate_files"].append(intermediate_file)
            print(f"Action completed. Intermediate file created: {intermediate_file}")

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

    def _review_node(self, state: EntityAgentState) -> EntityAgentState:
        print("Reviewing action results...")
        plan = state["current_plan"]
        action_result = state["action_result"]

        review_prompt = f"""
        Review the following action result. Your goal is to stop iterating as soon as the result is reasonably good. Do not aim for perfection.

        PLAN:
        {plan}

        ACTION RESULT:
        {action_result}

        EVALUATION CRITERIA:
        1. **Error Check**: The `success` flag in the `ACTION RESULT` indicates if a critical error occurred. If `success` is `False`, the action failed and is not good enough.
        2. **Completeness Check**: Has the agent extracted a comprehensive set of entities and relationships? While perfection is not required, the extraction should cover the main points from the text. If there are obvious, high-value entities or relationships still missing, it is not good enough.

        Based on these criteria, provide your review.

        Return your review in this format:
        IS_GOOD_ENOUGH: [Yes/No] - Answer "Yes" only if `success` is `True` and the extracted data is comprehensive.
        FEEDBACK: [Provide brief feedback. If it's good enough, say "The result is comprehensive and sufficient." If not, list a few examples of the most critical missing entities or relationships to guide the next iteration.]
        """
        response = llm_wrapper(review_prompt, max_tokens=1000)

        is_good_enough = self._extract_section(response, "IS_GOOD_ENOUGH").lower().strip() == "yes"
        feedback = self._extract_section(response, "FEEDBACK")

        state["review_result"] = {
            "meets_criteria": is_good_enough,
            "feedback": feedback,
            "next_action_needed": not is_good_enough,
            "suggestions": [] # Keep this simple
        }
        
        state["previous_results"].append(action_result)
        state["current_status"] = f"Previous iteration feedback: {feedback}"
        
        print(f"Review completed: {'Good enough' if is_good_enough else 'Needs improvement'}")
        print(f"Feedback: {feedback}")
        return state

    def _finalize_node(self, state: EntityAgentState) -> EntityAgentState:
        print("Finalizing entity recognition...")
        
        final_result = state["previous_results"][-1]["output"]
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_file = os.path.join(self.output_dir, f"final_entities_{timestamp}.json")
        
        with open(final_output_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2)
            
        state["final_output_file"] = final_output_file
        state["workflow_complete"] = True
        
        print(f"Final results saved to: {final_output_file}")
        return state

    def _should_continue(self, state: EntityAgentState) -> str:
        if state["iteration"] >= state["max_iterations"]:
            print("Max iterations reached.")
            return "finalize"
        
        review = state.get("review_result")
        if review and review["meets_criteria"] and not review["next_action_needed"]:
            print("Criteria met, finalizing.")
            return "finalize"
            
        return "continue"

    def _infer_from_ontology(self, extracted_entities: List[Dict]) -> Dict:
        """Infers potential entities and relationships from an ontology graph."""
        placeholders = {"entities": [], "relationships": []}
        if not self.ontology_graph:
            return placeholders

        for entity in extracted_entities:
            entity_type = entity.get("type")
            entity_name = entity.get("name")
            if entity_type and entity_name and self.ontology_graph.has_node(entity_type):
                for neighbor_type in self.ontology_graph.neighbors(entity_type):
                    # Add placeholder for the new entity
                    placeholders["entities"].append({
                        "name": "<FILL>",
                        "type": neighbor_type,
                        "description": f"<FILL: description for {neighbor_type} related to {entity_name}>"
                    })
                    # Add placeholder for the relationship
                    placeholders["relationships"].append({
                        "source": entity_name,
                        "target": "<FILL>",
                        "description": "<FILL>"
                    })
        return placeholders

    def _fill_inferred_with_llm(self, placeholders: Dict, input_text: str) -> Dict:
        """Uses LLM to fill in the details of inferred entities and relationships."""
        prompt = f"""
        Given the following text:
        ---
        {input_text}
        ---
        
        And the following placeholders for entities and relationships:
        ---
        {json.dumps(placeholders, indent=2)}
        ---

        **STRICT OUTPUT FORMAT**:
        Your output MUST be a single JSON object.
        The object must have two keys: "entities" and "relationships".
        - Each object in the "entities" list must have the keys: "name", "type", and "description".
        - Each object in the "relationships" list must have the keys: "source", "target", and "description".

        **CRITICAL INSTRUCTION FOR DESCRIPTIONS**:
        For each placeholder you fill, the "description" field MUST be one or two **full sentences** copied **exactly as they appear** from the input text.
        These sentences should best describe the entity's attributes or the relationship.
        **Do not summarize, paraphrase, or alter the original text in any way. Copy it verbatim.**
        
        Please fill in the "<FILL>" values based on the text.
        If you cannot find information for a placeholder, omit it from your response.
        Return ONLY a JSON object with the completed entities and relationships.
        """
        
        response = llm_wrapper(prompt, max_tokens=4000)
        json_response = self._clean_json_response(response)
        
        try:
            filled_data = json.loads(json_response)

            # Defensive check: if LLM returns a list, wrap it in a dict
            if isinstance(filled_data, list):
                filled_data = {"entities": filled_data, "relationships": []}

            # Ensure the dict has the expected keys, even if the LLM omits them
            if "entities" not in filled_data:
                filled_data["entities"] = []
            if "relationships" not in filled_data:
                filled_data["relationships"] = []

            return filled_data
        except json.JSONDecodeError:
            return {"entities": [], "relationships": []}

    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a specific section from LLM response"""
        try:
            lines = text.split('\n')
            section_content = []
            section_found = False
            section_keywords = ['ACTION_DESCRIPTION', 'TEST_CRITERIA', 'EXPECTED_OUTPUT', 'MEETS_CRITERIA', 'FEEDBACK', 'NEXT_ACTION_NEEDED', 'SUGGESTIONS', 'EXTRACTED_ENTITIES', 'VALIDATION_INFO', 'IS_GOOD_ENOUGH']
            
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
            return result
        except Exception as e:
            return ""

    def _clean_json_response(self, response: str) -> str:
        """Cleans the LLM response to ensure it's a valid JSON string."""
        # Remove markdown code block fences
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        
        return response.strip()
        
    def run(self, input_text: str, max_iterations: Optional[int] = None) -> Optional[str]:
        current_max_iterations = max_iterations if max_iterations is not None else self.max_iterations
        initial_state = EntityAgentState(
            input_text=input_text,
            output_dir=self.output_dir,
            iteration=0,
            max_iterations=current_max_iterations,
            current_status="",
            workflow_complete=False,
            current_plan=None,
            action_result=None,
            review_result=None,
            previous_results=[],
            extracted_entities=[],
            extracted_relationships=[],
            intermediate_files=[],
            final_output_file=None,
            errors=[],
            target_schema=self.target_schema
        )
        
        try:
            result = self.workflow.invoke(initial_state)
            if result.get("workflow_complete"):
                print("\nEntity recognition completed successfully!")
                return result.get("final_output_file")
            else:
                print("\nEntity recognition terminated without completion.")
                return None
        except Exception as e:
            print(f"Error running entity recognition workflow: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # Example Usage
    sample_text = """
    3.1.1 Impact of fixed rise in temperature, CO₂ and change in rain fall

    Monsoon crop
    Results of simulation analysis indicate that maize yields in monsoon season are adversely affected due to rise in atmospheric temperatures in all three regions (Fig. 2a). Grain yield decreased with each degree rise in atmospheric temperature. However, the rate of decrease varied with location. The mean baseline yield of rainfed maize crop is about 2 Mg ha⁻¹ in UIGP, where the projected yield loss is up to 7, 11, 15, 22, and 33% relative to baseline yields with 1, 2, 3, 4, 5°C degrees rise in atmospheric temperatures. However, a 20% increase in rainfall is projected to offset the yield loss due to 1°C rise in temperature. Similarly, a 30% increase in rainfall is predicted to offset the adverse impact of 2°C rise in temperature. In MIGP region, yield reduction of about 8–35% with 1–5°C rise in atmospheric temperature is projected. In this region, increase in rainfall is likely to offset the temperature rise up to 0.75°C and any increase beyond this temperature will adversely impact the yields, in spite of increase in rainfall. The SP region is also projected to experience adverse impact with -10, -15, -23, -27 and -35% reductions from the baseline yield levels at each 1°C rise in temperature. A 10% increase in rainfall will offset the reduction in yield due to 1°C rise in temperature in this region.

    Even though maize is a C4 plant, increase in carbon dioxide is projected to benefit the crop yield ranging from 0.1 to 3.4% at 450 ppmV and 0.6 to 7.2% at 550 ppmV. The benefits are projected to be high in mild water stress conditions, but they are likely to reduce in severe water stress situations (Table 3). The yield gains due to increase in atmospheric CO₂ concentration are projected to be more in SP regions (low rainfall area) followed by UIGP and MIGP regions.

    Winter crop
    Maize crop during winter is provided with assured irrigation and thus yields about 1.5 times more than that of monsoon crop. Winter maize grain yield reduced with increase in temperatures in SP and MIGP, but in UIGP rise in temperatures up to 2.7°C is likely to improve the maize yields. However, further increase in temperature is projected to reduce grain yields and the reductions are likely to be more than those at MIGP and SP (Fig. 3a). In UIGP, this beneficial effect with rise in temperature is projected to be more up to 2°C rise (13% increase over current yields). In this region, yield will improve with 2°C in spite of reduction in rainfall. In the event of further increase in temperature to about 2.7°C, the reduction in yields can be offset only if rainfall is increased or more irrigation is provided. With temperature rise, the crop experiences conditions closer to optimal temperature during grain development, benefiting grain number. Relatively low temperature during grain filling period required more days to satisfy thermal time requirement. However, in both MIGP and SP, where the average maximum temperatures during winter crop season are relatively higher (Table 2), any increase in temperature can cause reduction in yield.

    Table 3 Influence of atmospheric carbon dioxide concentration on maize yields in rainfall deficit conditions during monsoon season

    In UIGP, rise in temperatures beyond 2.7°C caused reduction in yield mainly due to reduced number of grains. This limited the gains in spite of increase in GFD and individual grain weight. Further increase in temperature resulted in yield reduction from current yields. In UIGP, GFD was found to increase with rise in temperature because of current lower temperature during winter. While the rise in temperature prolonged GFD significantly at UIGP than at MIGP, it actually reduced at SP. In all locations, flowering hastened due to increase in temperature.
    3.1.2 Impact of climate change scenarios on maize yield

    The climate change scenario outputs of HadCM3 model on minimum and maximum temperatures and rainfall; CO₂ concentrations as per Bern CC model for 2020, 2050 and 2080 were coupled to InfoCrop-MAIZE model. This approach was followed because of reported spatio-temporal variations in climate change scenarios (IPCC 2007).

    Monsoon crop
    The analysis indicates that in UIGP region, climate change is projected to insignificantly affect the productivity of monsoon maize crop in 2020, 2050 and 2080 scenarios (Fig. 4a). This is mainly due to projected increase in rainfall during crop season, which will provide scope for improved dry matter production and increase in grain number. This implies that the maize crop may benefit from additional availability of water in spite of increase in temperature and related reduction in crop duration by 3–4 days. On the other hand, in MIGP, maize is likely to suffer yield loss in future scenarios. The loss from current yields is projected to be ~5%, ~13%, ~17% in 2020, 2050 and 2080, respectively. In SP, monsoon season crop is projected to lose grain yield by 21% from current yields due to climate change by 2020 and 35% by 2050 and later. Projected rise in daytime temperature during monsoon is higher in SP and MIGP as compared to UIGP region, even though minimum temperatures are projected to rise almost similarly in these locations. Apart from this, rainfall is projected to increase in UIGP while it is likely to change in MIGP. Thus, the spatio-temporal variation in existing climatic conditions and projected changes in temperature and rainfall would bring about differential impacts on monsoon maize crop in India.

    Winter crop
    As far as maize crop grown in winter is concerned, yield gains are projected to be ~5% over current yield in 2020 scenario at UIGP and this benefit is likely to remain till 2050 (Fig. 4b). However, in 2080 scenario, yields are projected to be reduced by 25% from current yields. Winter maize in MIGP, currently a high yielding zone, is projected to suffer in post-2020 scenario. The reduction in yield is likely to be to the tune of ~50% by 2050 and about 60% by 2080. In SP region, yields are projected to decline by about 13% in 2020, 17% in 2050 and 21% in 2080. In these areas, winter maize is well irrigated and thus variation in winter rainfall, which otherwise is low, is less influential. The projected rise in temperature during winter crop season is more in UIGP in 2020 and 2050 than in MIGP and SP, particularly during later part of crop growth.
    """
    
    # Define the target schema based on the standard from AnalystAgent
    target_schema = {
        "Crop Type": "Name of the crop (e.g., maize, wheat, rice, soybean)",
        "Crop Yield": "NUMERICAL VALUE ONLY. Use positive numbers for yield increases, negative numbers for yield decreases. No text or units.",
        "Crop Yield Unit": "Unit of measurement for crop yield (e.g., tons/ha, kg/ha, Mg/ha, bushels/acre, %)",
        "Climate Drivers": "Climate variable affecting the crop (e.g., temperature, precipitation, CO2, drought)",
        "Climate Drivers Value": "NUMERICAL VALUE ONLY. Use positive numbers for increases (+1, +2.5), negative numbers for decreases (-1, -0.5). No text or units.",
        "Climate Drivers Unit": "Unit of measurement for climate driver (e.g., °C, mm, ppm, %)",
        "Experimental Design": "Type of study or model used (e.g., field experiment, crop model simulation, greenhouse study)",
        "Location": "Geographic location or region (e.g., country, state, coordinates, study site name)",
        "Time": "Time period or duration of study (e.g., 1990-2000, baseline period, future projection)",
        "Source in paper": "Original text description from the entities or links file that contains the specific data point or evidence"
    }

    # Create a sample ontology from the schema keys
    ontology = nx.Graph()
    entity_types = list(target_schema.keys())
    # Example: link location to crop type
    if "Location" in entity_types and "Crop Type" in entity_types:
        ontology.add_edge("Location", "Crop Type")
    if "Climate Drivers" in entity_types and "Crop Yield" in entity_types:
        ontology.add_edge("Climate Drivers", "Crop Yield")

    
    # Set output directory relative to this script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "test_output")

    agent = EntityRecognitionAgent(ontology_graph=ontology, target_schema=target_schema, output_dir=output_dir)
    result_file = agent.run(sample_text, max_iterations=10)
    
    if result_file:
        print(f"\nAgent run finished. See results in: {result_file}")
    else:
        print("\nAgent run failed.") 