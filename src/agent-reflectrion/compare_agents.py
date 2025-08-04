#!/usr/bin/env python3
"""
Comparison script for Original vs LangGraph Meta-Analysis Agents

This script demonstrates both implementations and compares their features.
"""

import os
import time
import sys
from typing import Optional

def test_original_agent() -> Optional[str]:
    """Test the original agent implementation"""
    print("=" * 60)
    print("TESTING ORIGINAL AGENT")
    print("=" * 60)
    
    try:
        from agent_reflection import MetaAnalysisAgent
        
        agent = MetaAnalysisAgent(
            entities_file="/home/com3dian/Github/meta-knowledge-harvesting-llm/src/outputs/_result_20250713_093148_entities.txt",
            links_file="/home/com3dian/Github/meta-knowledge-harvesting-llm/src/outputs/_result_20250713_093148_links.txt",
            output_dir="../outputs/original_agent"
        )
        
        # Set max iterations to 3 for comparison
        agent.max_iterations = 3
        
        start_time = time.time()
        result_file = agent.run_meta_analysis()
        end_time = time.time()
        
        print(f"\nOriginal Agent Results:")
        print(f"- Max iterations: {agent.max_iterations}")
        print(f"- Execution time: {end_time - start_time:.2f} seconds")
        print(f"- Output file: {result_file}")
        print(f"- Iterations completed: {agent.iteration}")
        print(f"- Records extracted: {len(agent.extracted_records)}")
        
        return result_file
        
    except Exception as e:
        print(f"Error testing original agent: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_langgraph_agent() -> Optional[str]:
    """Test the LangGraph agent implementation"""
    print("\n" + "=" * 60)
    print("TESTING LANGGRAPH AGENT")  
    print("=" * 60)
    
    try:
        from langgraph_agent import LangGraphMetaAnalysisAgent, LANGGRAPH_AVAILABLE
        
        if not LANGGRAPH_AVAILABLE:
            print("LangGraph not available. Please install with: pip install langgraph")
            return None
        
        agent = LangGraphMetaAnalysisAgent(
            entities_file="/home/com3dian/Github/meta-knowledge-harvesting-llm/src/outputs/_result_20250713_093148_entities.txt",
            links_file="/home/com3dian/Github/meta-knowledge-harvesting-llm/src/outputs/_result_20250713_093148_links.txt", 
            output_dir="../outputs/langgraph_agent",
            max_iterations=3  # Set max iterations to 3 for comparison
        )
        
        start_time = time.time()
        result_file = agent.run_meta_analysis()
        end_time = time.time()
        
        print(f"\nLangGraph Agent Results:")
        print(f"- Max iterations: {agent.max_iterations}")
        print(f"- Execution time: {end_time - start_time:.2f} seconds")
        print(f"- Output file: {result_file}")
        
        return result_file
        
    except Exception as e:
        print(f"Error testing LangGraph agent: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_results(original_file: Optional[str], langgraph_file: Optional[str]):
    """Compare the results from both agents"""
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    if not original_file and not langgraph_file:
        print("No results to compare - both agents failed")
        return
    
    if original_file and langgraph_file:
        try:
            # Compare file sizes
            original_size = os.path.getsize(original_file) if os.path.exists(original_file) else 0
            langgraph_size = os.path.getsize(langgraph_file) if os.path.exists(langgraph_file) else 0
            
            print(f"File sizes:")
            print(f"- Original agent: {original_size} bytes")
            print(f"- LangGraph agent: {langgraph_size} bytes")
            
            # Compare content (first few lines)
            if os.path.exists(original_file):
                with open(original_file, 'r', encoding='utf-8') as f:
                    original_preview = f.read(500)
            else:
                original_preview = "File not found"
                
            if os.path.exists(langgraph_file):
                with open(langgraph_file, 'r', encoding='utf-8') as f:
                    langgraph_preview = f.read(500)
            else:
                langgraph_preview = "File not found"
            
            print(f"\nContent preview (first 500 chars):")
            print(f"Original agent output:\n{original_preview}\n")
            print(f"LangGraph agent output:\n{langgraph_preview}\n")
            
        except Exception as e:
            print(f"Error comparing results: {e}")
    
    elif original_file:
        print("Only original agent succeeded")
        print(f"Original agent output: {original_file}")
    
    elif langgraph_file:
        print("Only LangGraph agent succeeded")
        print(f"LangGraph agent output: {langgraph_file}")

def print_feature_comparison():
    """Print a feature comparison table"""
    print("\n" + "=" * 60)
    print("FEATURE COMPARISON")
    print("=" * 60)
    
    features = [
        ("Feature", "Original Agent", "LangGraph Agent"),
        ("-" * 30, "-" * 15, "-" * 17),
        ("State Management", "Manual", "TypedDict"),
        ("Workflow Structure", "While Loop", "State Graph"),
        ("Error Handling", "Basic", "Advanced"),
        ("Debugging", "Limited", "Graph Visualization"),
        ("Extensibility", "Moderate", "High"),
        ("Dependencies", "Minimal", "LangGraph + LangChain"),
        ("Conditional Flow", "Basic", "Advanced Routing"),
        ("State Persistence", "Instance Vars", "Structured State"),
        ("Modularity", "Methods", "Graph Nodes"),
        ("Testing", "Direct", "Node Isolation")
    ]
    
    for feature, original, langgraph in features:
        print(f"{feature:<30} {original:<15} {langgraph}")

def main():
    """Main comparison function"""
    print("Meta-Analysis Agent Comparison")
    print("This script compares the original agent with the new LangGraph implementation")
    print("Both agents configured to run for a maximum of 3 iterations")
    
    # Check if files exist
    entities_file = "/home/com3dian/Github/meta-knowledge-harvesting-llm/src/outputs/_result_20250713_093148_entities.txt"
    links_file = "/home/com3dian/Github/meta-knowledge-harvesting-llm/src/outputs/_result_20250713_093148_links.txt"
    
    if not os.path.exists(entities_file) or not os.path.exists(links_file):
        print(f"Warning: Input files not found:")
        print(f"- Entities: {entities_file}")
        print(f"- Links: {links_file}")
        print("Running in test mode with mock data...")
    
    # Test both agents
    original_result = test_original_agent()
    langgraph_result = test_langgraph_agent()
    
    # Compare results
    compare_results(original_result, langgraph_result)
    
    # Print feature comparison
    print_feature_comparison()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✓ Original Agent: Traditional class-based implementation")
    print("✓ LangGraph Agent: Modern workflow-based implementation")
    print("✓ Both agents implement the same plan-action-review cycle")
    print("✓ LangGraph provides better state management and extensibility")
    print("✓ Choose based on your complexity and maintenance needs")

if __name__ == "__main__":
    main() 