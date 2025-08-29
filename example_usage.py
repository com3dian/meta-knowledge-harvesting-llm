#!/usr/bin/env python3
"""
Example usage of the Research Agent Framework
"""

from src.research_agent import ResearchAgent, Goal

def example_programmatic_usage():
    """Example of using the agent programmatically"""
    
    # Define a research goal
    goal = Goal(
        description="Analyze the content of README.md and create a summary",
        success_criteria=[
            "Successfully read the README.md file",
            "Create a comprehensive summary",
            "Generate actionable insights"
        ],
        max_iterations=3
    )
    
    # Create and run the agent
    agent = ResearchAgent(goal, workspace_dir="./example_workspace")
    
    print("Starting research agent programmatically...")
    success = agent.run()
    
    if success:
        print("✅ Research completed successfully!")
    else:
        print("❌ Research did not complete successfully")
    
    return success

def custom_research_example():
    """Example with custom research topic"""
    
    goal = Goal(
        description="Research and document Python best practices for file handling",
        success_criteria=[
            "Gather information about file handling",
            "Document best practices",
            "Create examples and guidelines"
        ],
        max_iterations=4
    )
    
    agent = ResearchAgent(goal, workspace_dir="./python_research")
    return agent.run()

if __name__ == "__main__":
    print("=" * 60)
    print("Research Agent Framework - Example Usage")
    print("=" * 60)
    
    # Run the example
    example_programmatic_usage()
    
    print("\n" + "=" * 60)
    print("Check the workspace directory for generated files!")
    print("=" * 60) 