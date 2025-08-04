# Deep Research Agent Framework

A flexible framework for building research agents that follow the **Plan → Action → Review → Plan** cycle. The agent autonomously designs tests for each plan to evaluate progress toward research goals.

## Features

- 🔄 **Iterative Research Cycle**: Plan → Action → Review → Plan
- 🧪 **Automated Testing**: Each plan includes test criteria to evaluate success
- 📁 **File Operations**: Built-in tools for reading and writing files
- 📊 **Progress Tracking**: Detailed logging and progress scoring
- 🎯 **Goal-Oriented**: Clear success criteria and goal achievement detection
- 📋 **Comprehensive Reporting**: Automatic generation of final research reports

## Architecture

### Core Components

1. **ResearchAgent**: Main orchestrator that manages the research cycle
2. **Goal**: Defines the research objective and success criteria
3. **Plan**: Represents a research plan with actions and test criteria
4. **Tools**: Extensible tools for performing actions (ReadFileTool, WriteFileTool)
5. **ActionResult**: Contains results of executed actions
6. **ReviewResult**: Contains evaluation results and next steps

### Research Cycle

```
┌─────────┐    ┌─────────┐    ┌─────────┐
│  PLAN   │───▶│ ACTION  │───▶│ REVIEW  │
└─────────┘    └─────────┘    └─────────┘
     ▲                             │
     └─────────────────────────────┘
```

1. **PLAN**: Generate research plan with actions and test criteria
2. **ACTION**: Execute the planned actions using available tools
3. **REVIEW**: Run tests, evaluate progress, and determine next steps

## Quick Start

### Command Line Usage

```bash
# Basic usage
python research_agent.py --goal "Analyze project structure and create documentation"

# With custom parameters
python research_agent.py \
    --goal "Research Python best practices" \
    --success-criteria "Document findings" "Create examples" \
    --max-iterations 5 \
    --workspace ./my_research
```

### Programmatic Usage

```python
from research_agent import ResearchAgent, Goal

# Define your research goal
goal = Goal(
    description="Analyze codebase and generate documentation",
    success_criteria=[
        "Successfully read source files",
        "Generate comprehensive documentation",
        "Create usage examples"
    ],
    max_iterations=5
)

# Create and run the agent
agent = ResearchAgent(goal, workspace_dir="./research_output")
success = agent.run()
```

## Customization

### Extending the Planning Logic

Override the `_generate_plan()` method to implement domain-specific planning:

```python
class CustomResearchAgent(ResearchAgent):
    def _generate_plan(self, plan_id: str) -> Plan:
        # Your custom planning logic here
        if self.iteration_count == 1:
            # First iteration logic
            return Plan(
                id=plan_id,
                description="Custom initial exploration",
                actions=["read_file: config.yaml", "write_file: analysis_start"],
                expected_outcomes=["Configuration understood"],
                test_criteria=["Config file successfully parsed"],
                created_at=datetime.now().isoformat()
            )
        # Additional iterations...
```

### Adding Custom Tools

Extend the `Tool` base class to add new capabilities:

```python
class WebSearchTool(Tool):
    def execute(self, query: str) -> ActionResult:
        # Implement web search functionality
        try:
            results = search_web(query)
            return ActionResult(
                action=f"web_search({query})",
                success=True,
                output=results
            )
        except Exception as e:
            return ActionResult(
                action=f"web_search({query})",
                success=False,
                output=None,
                error_message=str(e)
            )

# Add to agent
agent.tools['web_search'] = WebSearchTool()
```

### Custom Test Logic

Override the `_run_test()` method for domain-specific testing:

```python
def _run_test(self, criterion: str, action_results: List[ActionResult]) -> Dict[str, Any]:
    test_result = {'criterion': criterion, 'passed': False, 'details': ''}
    
    if "documentation" in criterion.lower():
        # Check if documentation was created
        doc_files = [r for r in action_results if r.success and "doc" in str(r.output)]
        test_result['passed'] = len(doc_files) > 0
        test_result['details'] = f"Found {len(doc_files)} documentation files"
    
    return test_result
```

## File Structure

After running the agent, you'll find:

```
workspace/
├── research_log_1.md          # Iteration 1 log
├── research_log_2.md          # Iteration 2 log
├── final_report_YYYYMMDD_HHMMSS.json  # Complete research report
└── research_agent.log         # Detailed execution log
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--goal` | Research goal description | Required |
| `--success-criteria` | List of success criteria | `["Complete research"]` |
| `--max-iterations` | Maximum number of iterations | `5` |
| `--workspace` | Workspace directory | `"./workspace"` |

## Logging

The agent provides comprehensive logging at multiple levels:

- **Console Output**: Real-time progress with emojis and clear phases
- **File Logging**: Detailed logs saved to `research_agent.log`
- **Progress Tracking**: Numerical progress scores (0.0 to 1.0)
- **Final Reports**: Complete JSON reports with all data

## Example Output

```
================================================================================
STARTING RESEARCH AGENT
Goal: Analyze project structure and create documentation
Success Criteria: ['Read source files', 'Generate documentation']
================================================================================

🔄 ITERATION 1
--------------------------------------------------
📋 PLANNING PHASE
✅ Generated plan: Initial exploration and setup
   Actions: ['read_file: Check existing files', 'write_file: Create initial doc']
   Expected outcomes: ['Understanding of current state', 'Foundation document created']
   Test criteria: ['Files successfully read/written', 'Document contains goal statement']

🚀 ACTION PHASE
   Executing: read_file: Check existing files
   ✅ Success: read_file: Check existing files
   Executing: write_file: Create initial doc
   ✅ Success: write_file: Create initial doc

🔍 REVIEW PHASE
   Test: Files successfully read/written - ✅ PASS
   Test: Document contains goal statement - ✅ PASS
   Overall Success: True
   Progress Score: 1.00
   Feedback: Successfully completed 2 actions

🎉 GOAL ACHIEVED!
📄 Final report saved: ./workspace/final_report_20240101_120000.json
```

## Advanced Features

### Progress Scoring
- Each review calculates a progress score (0.0 to 1.0)
- Based on test success rate and goal achievement
- Used to determine if research should continue

### Adaptive Planning
- Plans adapt based on previous iteration results
- Failed actions trigger retry strategies
- Success builds on previous achievements

### Test-Driven Research
- Each plan includes specific test criteria
- Tests are automatically executed during review
- Results guide next iteration planning

## Contributing

To extend the framework:

1. **Add Tools**: Inherit from `Tool` class and implement `execute()` method
2. **Custom Agents**: Inherit from `ResearchAgent` and override specific methods
3. **New Data Types**: Add dataclasses for custom data structures
4. **Enhanced Testing**: Extend `_run_test()` for domain-specific evaluations

## Requirements

- Python 3.7+
- Standard library only (no external dependencies)

## License

This framework is provided as-is for educational and research purposes. 