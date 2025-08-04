# LangGraph Meta-Analysis Agent

This directory contains two implementations of a meta-analysis agent that extracts structured records from scientific literature data using a plan-action-review cycle:

1. **Original Agent** (`agent_reflection.py`) - A traditional class-based implementation
2. **LangGraph Agent** (`langgraph_agent.py`) - A new implementation using LangGraph's state management and workflow system

## Key Differences

### Original Agent (`agent_reflection.py`)
- **Simple iteration loop**: Uses a basic while loop to manage the plan-action-review cycle
- **Manual state management**: Tracks state using instance variables
- **Linear execution**: Sequential execution of plan → action → review
- **Basic error handling**: Simple try-catch blocks
- **Direct method calls**: Methods are called directly in sequence

### LangGraph Agent (`langgraph_agent.py`)
- **State machine workflow**: Uses LangGraph's StateGraph for workflow management
- **Typed state management**: Uses TypedDict for structured state tracking
- **Graph-based execution**: Workflow defined as nodes and edges
- **Advanced routing**: Conditional edges determine workflow flow
- **Persistent state**: State is maintained across workflow steps
- **Modular design**: Each step is a separate node function

## LangGraph Advantages

1. **Better State Management**: 
   - Structured state with TypedDict
   - Automatic state persistence between nodes
   - Clear state transitions

2. **Workflow Visualization**: 
   - Graph structure makes workflow easy to understand
   - Can be visualized and debugged
   - Clear dependencies between steps

3. **Conditional Routing**: 
   - Dynamic workflow paths based on review results
   - Better handling of success/failure cases
   - More sophisticated control flow

4. **Extensibility**: 
   - Easy to add new nodes or modify workflow
   - Can integrate with other LangGraph tools
   - Better support for complex workflows

5. **Error Recovery**: 
   - Better error handling and recovery mechanisms
   - State preservation during failures
   - Easier debugging

## Workflow Structure

The LangGraph agent implements the following workflow:

```
Initialize → Plan → Action → Review → [Continue/Finalize/End]
    ↑                              ↓
    └──────────── Continue ────────┘
```

### Nodes:
- **Initialize**: Load input data and set up initial state
- **Plan**: Create a plan for the next action using LLM
- **Action**: Execute the planned action to extract records
- **Review**: Review results against plan criteria
- **Finalize**: Save final results and complete workflow

### Conditional Routing:
- **Continue**: Go back to planning for another iteration
- **Finalize**: Complete workflow and save results
- **End**: Terminate workflow

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install langgraph langchain langchain-core python-dotenv
```

## Usage

### Basic Usage

```python
from langgraph_agent import LangGraphMetaAnalysisAgent

# Initialize the agent
agent = LangGraphMetaAnalysisAgent(
    entities_file="path/to/entities.txt",
    links_file="path/to/links.txt",
    output_dir="outputs"  
)

# Run the meta-analysis
result_file = agent.run_meta_analysis()
print(f"Results saved to: {result_file}")
```

### Configuration

Set up your environment variables in a `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

## State Structure

The agent uses a comprehensive state structure:

```python
class AgentState(TypedDict):
    # Input data
    entities_file: str
    links_file: str
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
```

## Output Format

Both agents produce the same CSV output format with these headers:
- Crop Type
- Crop Yield  
- Crop Yield Unit
- Climate Drivers
- Climate Drivers Value
- Climate Drivers Unit
- Experimental Design
- Location
- Time
- Source in paper

## Error Handling

The LangGraph agent provides improved error handling:
- Graceful handling of LLM failures
- State preservation during errors
- Detailed error logging
- Fallback to mock responses for testing

## Testing

Both agents support mock mode for testing without API keys:
- Set `GEMINI_API_KEY` environment variable for real LLM usage
- Agents automatically fall back to mock responses if API key is missing
- Mock responses demonstrate expected output format

## Performance Considerations

- **LangGraph overhead**: Slightly more overhead due to state management
- **Better debugging**: Easier to debug and understand workflow execution
- **Scalability**: Better foundation for complex, multi-step workflows
- **Maintainability**: More modular and easier to extend

## When to Use Which

### Use Original Agent When:
- Simple, straightforward workflows
- Minimal dependencies preferred
- Quick prototyping
- Linear execution is sufficient

### Use LangGraph Agent When:
- Complex workflows with multiple paths
- Need for better state management
- Planning to extend with more features
- Want better debugging and visualization
- Building production systems
- Need conditional workflow routing

## Future Enhancements

The LangGraph implementation provides a foundation for:
- Multi-agent workflows
- Tool integration
- Human-in-the-loop processes
- Parallel processing
- Advanced error recovery
- Workflow persistence and resumption 