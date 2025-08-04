#!/usr/bin/env python3
"""
Test script for the first step planning functionality
"""

import os
import json
from research_agent import ResearchAgent
from research_agent_data_types import Goal
from dotenv import load_dotenv
from llm_impl.geminillm import gemini_complete_if_cache

LLM_MODEL_NAME = "gemini-2.5-flash-preview-05-20"

load_dotenv()

def mock_llm_wrapper(prompt: str, history_messages=None, max_tokens=None) -> str:
    if history_messages is None:
        history_messages = []


    # Use Google GenAI
    return gemini_complete_if_cache(
        model=LLM_MODEL_NAME,
        prompt=prompt,
        history_messages=history_messages,
        temperature=0.2,
        max_tokens=max_tokens or 1024,
    )


def main():
    print("🧪 TESTING COMPLETE META-ANALYSIS EXTRACTION WORKFLOW")
    print("=" * 60)

    entities_file = "/home/com3dian/Github/meta-knowledge-harvesting-llm/src/outputs/_result_20250708_133329_entities copy.txt"
    links_file = "/home/com3dian/Github/meta-knowledge-harvesting-llm/src/outputs/_result_20250708_133329_links copy.txt"
    
    # Define the exact CSV format we want to extract
    target_csv_format = """
    TARGET CSV FORMAT:
    Crop type,Crop yield,Unit,Climate drivers,Values,Unit,Experimental design,Location,Time,Source in paper
    
    Example row:
    maize,2,tons/ha,Atmospheric temperature,+1,℃,InfoCrop-MAIZE model simulations,UIGP,baseline period-1970 to 1995,"Temperature impact study details"
    """
    
    # Create a comprehensive meta-analysis goal
    goal = Goal(
        description=f"""
        EXTRACT and CREATE comprehensive meta-analysis data from available sources and produce structured CSV output files.
        
        MANDATORY REQUIREMENTS (all must be completed):
        1. READ source files: {entities_file}, {links_file}
        2. EXTRACT climate impact data records from the sources
        3. CREATE CSV file(s) with exact columns: Crop type, Crop yield, Unit, Climate drivers, Values, Unit, Experimental design, Location, Time, Source in paper
        4. ENSURE at least 10-15 complete data records in the CSV output
        5. VALIDATE data quality, completeness, and proper formatting
        6. PRODUCE final meta-analysis CSV file in workspace
        
        CRITICAL: This is not just about reading data - you MUST create new structured CSV files with extracted records.
        
        REFERENCE FORMAT: {target_csv_format}
        
        SUCCESS DEFINITION: Goal is achieved ONLY when structured CSV files have been created in the workspace containing meta-analysis records in the specified format.
        """,
        success_criteria="Successfully extract meta-analysis data, create properly formatted CSV files with required columns, validate data completeness with 10+ records, and save output files to workspace.",
        max_iterations=8  # Increased to ensure completion
    )
    
    # Create workspace
    workspace = "./test_workspace"
    os.makedirs(workspace, exist_ok=True)
    
    # Create research agent
    agent = ResearchAgent(
        goal=goal,
        llm_wrapper=mock_llm_wrapper,
        workspace_dir=workspace
    )
    
    print(f"📋 Goal: {goal.description}")
    print(f"✅ Success Criteria: {goal.success_criteria}")
    print(f"🔄 Max Iterations: {goal.max_iterations}")
    print()
    
    # Show target format
    print("🎯 TARGET CSV FORMAT:")
    print(target_csv_format)
    print()
    
    print("🚀 STARTING MULTI-ITERATION META-ANALYSIS WORKFLOW...")
    print("=" * 60)
    
    # Run multiple iterations to show complete workflow
    all_plans = []
    all_action_results = []
    all_review_results = []
    
    for iteration in range(1, goal.max_iterations + 1):
        print(f"\n🔄 ITERATION {iteration}")
        print("-" * 40)
        
        agent.iteration_count = iteration
        agent.status.iteration_count = iteration
        
        # Show current status
        print(f"📊 Status: {agent.status.get_status_summary()}")
        if agent.status.key_findings:
            print(f"📌 Key Findings: {len(agent.status.key_findings)} total, latest: {agent.status.key_findings[-1][:60]}...")
        if agent.status.current_challenges:
            print(f"⚠️  Current Challenges: {len(agent.status.current_challenges)} total, latest: {agent.status.current_challenges[-1][:60]}...")
        print()
        
        # PHASE 1: PLANNING
        print(f"📋 PHASE 1: PLANNING (Iteration {iteration})")
        print("-" * 30)
        
        try:
            plan = agent.plan()
            
            if plan:
                all_plans.append(plan)
                print("✅ PLAN GENERATED:")
                print(f"   Description: {plan.description}")
                print(f"   Action: {plan.action}")
                print(f"   Expected Outcome: {plan.expected_outcome}")
                print(f"   Test Criteria: {plan.test_criteria}")
            else:
                print("❌ PLAN GENERATION FAILED!")
                break
                
        except Exception as e:
            print(f"❌ ERROR DURING PLANNING: {e}")
            break
        
        # PHASE 2: ACTION EXECUTION
        print(f"\n🚀 PHASE 2: ACTION EXECUTION (Iteration {iteration})")
        print("-" * 30)
        
        try:
            action_results = agent.execute_actions(plan)
            agent.action_results.extend(action_results)
            all_action_results.extend(action_results)
            
            for result in action_results:
                print(f"📊 Action Result:")
                print(f"   Action: {result.action}")
                print(f"   Success: {'✅ Yes' if result.success else '❌ No'}")
                
                if result.success and result.output:
                    output_str = str(result.output)
                    output_length = len(output_str)
                    
                    # Show overview instead of full content
                    if output_length > 200:
                        print(f"   Output Length: {output_length} characters")
                        print(f"   Output Preview: {output_str[:100]}...")
                    else:
                        print(f"   Output: {output_str}")
                        
                    # Check if output looks like CSV
                    if ',' in output_str and '\n' in output_str:
                        lines = output_str.strip().split('\n')
                        print(f"   📊 CSV-like output: {len(lines)} lines")
                        if len(lines) > 1:
                            headers = lines[0]
                            if len(headers) > 80:
                                headers = headers[:80] + "..."
                            print(f"   Headers: {headers}")
                else:
                    print(f"   Error: {result.error_message}")
                    
        except Exception as e:
            print(f"❌ ERROR DURING ACTION EXECUTION: {e}")
            break
        
        # PHASE 3: REVIEW AND TESTING
        print(f"\n🔍 PHASE 3: REVIEW AND TESTING (Iteration {iteration})")
        print("-" * 30)
        
        try:
            review_result = agent.review(plan, action_results)
            all_review_results.append(review_result)
            
            print(f"📊 Review Results:")
            print(f"   Overall Success: {'✅ Yes' if review_result.overall_success else '❌ No'}")
            print(f"   Progress Score: {review_result.progress_score:.2f}")
            print(f"   Feedback: {review_result.feedback}")
            
            # Show detailed test results
            if review_result.test_results:
                for i, test_result in enumerate(review_result.test_results, 1):
                    print(f"\n   Test {i}: {'✅ PASSED' if test_result['passed'] else '❌ FAILED'}")
                    print(f"     Details: {test_result['details']}")
                    
                    if 'individual_tests' in test_result:
                        passed_subtests = sum(1 for t in test_result['individual_tests'] if t.get('passed', False))
                        total_subtests = len(test_result['individual_tests'])
                        print(f"     Subtests: {passed_subtests}/{total_subtests} passed")
                        
                        # Show only failed tests for brevity
                        failed_tests = [t for t in test_result['individual_tests'] if not t.get('passed', False)]
                        if failed_tests and len(failed_tests) <= 3:  # Only show if 3 or fewer failures
                            for j, individual_test in enumerate(failed_tests, 1):
                                test_name = individual_test.get('test', f'Subtest {j}')
                                if len(test_name) > 50:
                                    test_name = test_name[:50] + "..."
                                print(f"       ❌ {test_name}")
                        elif failed_tests:
                            print(f"       ❌ {len(failed_tests)} tests failed (details truncated)")
            
            # Show next steps
            if review_result.next_steps:
                print(f"\n   📋 Next Steps ({len(review_result.next_steps)} total):")
                # Show max 3 next steps for brevity
                steps_to_show = review_result.next_steps[:3]
                for step in steps_to_show:
                    if len(step) > 70:
                        step = step[:70] + "..."
                    print(f"     • {step}")
                if len(review_result.next_steps) > 3:
                    print(f"     ... and {len(review_result.next_steps) - 3} more steps")
            
            # Update agent status
            agent.status.update_from_review(review_result, action_results)
            
        except Exception as e:
            print(f"❌ ERROR DURING REVIEW: {e}")
            break
        
        # PHASE 4: GOAL ACHIEVEMENT CHECK
        print(f"\n🎯 PHASE 4: GOAL ACHIEVEMENT CHECK (Iteration {iteration})")
        print("-" * 30)
        
        goal_achieved = agent.is_goal_achieved(review_result)
        should_continue = agent.should_continue(review_result)
        
        # Detailed diagnostic of goal achievement criteria
        print(f"📊 Goal Achievement Analysis:")
        print(f"   Overall Success: {'✅' if review_result.overall_success else '❌'}")
        print(f"   Progress Score: {review_result.progress_score:.2f} (need ≥0.8)")
        
        # Check specific meta-analysis criteria
        csv_created = any(
            "write_file" in result.action and result.success and 
            (".csv" in result.action.lower() or "csv" in str(result.output).lower())
            for result in all_action_results
        )
        print(f"   CSV Files Created: {'✅' if csv_created else '❌'}")
        
        # Check if data has been extracted (even if not saved to CSV yet)
        data_extracted = any(
            result.success and result.output and 
            ("crop" in str(result.output).lower() or "yield" in str(result.output).lower() or 
             "climate" in str(result.output).lower()) and len(str(result.output)) > 200
            for result in all_action_results
        )
        print(f"   Data Extracted: {'✅' if data_extracted else '❌'}")
        
        sufficient_data = any(
            result.success and result.output and 
            (',' in str(result.output) and str(result.output).count('\n') >= 5)
            for result in all_action_results
        )
        print(f"   Sufficient Data (≥5 lines): {'✅' if sufficient_data else '❌'}")
        
        required_columns_present = False
        for result in all_action_results:
            if result.success and result.output:
                output_str = str(result.output).lower()
                if ("crop" in output_str and "yield" in output_str and 
                    "climate" in output_str and "location" in output_str):
                    required_columns_present = True
                    break
        print(f"   Required Columns Present: {'✅' if required_columns_present else '❌'}")
        print(f"   Minimum Iterations (≥3): {'✅' if iteration >= 3 else '❌'}")
        
        # Continuation analysis
        print(f"\n📋 Continuation Analysis:")
        only_reading = all(
            "read_file" in result.action for result in all_action_results if result.success
        )
        print(f"   Only Reading Files: {'⚠️ Yes (need to extract data)' if only_reading else '✅ No'}")
        print(f"   Progress Score: {review_result.progress_score:.2f} (continue if >0.2)")
        print(f"   Iteration: {iteration}/{goal.max_iterations}")
        
        print(f"\n🎯 Decision:")
        print(f"   Goal Achieved: {'✅ Yes' if goal_achieved else '❌ No'}")
        print(f"   Should Continue: {'✅ Yes' if should_continue else '❌ No'}")
        
        if goal_achieved:
            print(f"\n🎉 GOAL ACHIEVED IN ITERATION {iteration}!")
            break
        elif not should_continue:
            print(f"\n⏹️  STOPPING: Insufficient progress in iteration {iteration}")
            if only_reading and not data_extracted:
                print(f"   🔧 Issue: Agent is only reading files, needs to extract data")
            elif data_extracted and not csv_created:
                print(f"   🔧 Issue: Data extracted but not saved to CSV files")
            if review_result.progress_score <= 0.2:
                print(f"   🔧 Issue: Progress score too low ({review_result.progress_score:.2f})")
            break
        elif iteration < goal.max_iterations:
            print(f"\n➡️  CONTINUING TO ITERATION {iteration + 1}")
            if iteration == 2 and only_reading:
                print(f"   🎯 Next priority: Extract and structure meta-analysis data from loaded files")
            elif iteration == 3 and data_extracted and not csv_created:
                print(f"   🎯 Next priority: Create CSV files with the extracted data")
            elif not sufficient_data:
                print(f"   🎯 Next priority: Extract more data records (need ≥10)")
            elif not required_columns_present:
                print(f"   🎯 Next priority: Ensure proper column structure")
        else:
            print(f"\n⏰ REACHED MAXIMUM ITERATIONS ({goal.max_iterations})")
            break
        
        print("\n" + "=" * 60)
    
    # FINAL ANALYSIS
    print(f"\n📈 COMPLETE WORKFLOW ANALYSIS")
    print("=" * 60)
    
    print(f"📊 Execution Summary:")
    print(f"   Total Iterations: {len(all_plans)}")
    print(f"   Plans Generated: {len(all_plans)}")
    print(f"   Actions Executed: {len(all_action_results)}")
    print(f"   Reviews Completed: {len(all_review_results)}")
    
    successful_actions = [r for r in all_action_results if r.success]
    print(f"   Successful Actions: {len(successful_actions)}/{len(all_action_results)}")
    
    # Analyze CSV extraction progress
    csv_outputs = []
    for result in all_action_results:
        if result.success and result.output:
            output_str = str(result.output)
            if ',' in output_str and '\n' in output_str and 'crop' in output_str.lower():
                csv_outputs.append(output_str)
    
    print(f"\n📊 Meta-Analysis Extraction Results:")
    print(f"   CSV-like outputs found: {len(csv_outputs)}")
    
    if csv_outputs:
        print(f"\n📄 Final CSV Output Analysis:")
        final_csv = csv_outputs[-1]  # Take the last/most recent CSV output
        lines = final_csv.strip().split('\n')
        print(f"   Total lines: {len(lines)}")
        
        if len(lines) > 0:
            headers = lines[0]
            if len(headers) > 100:
                headers = headers[:100] + "..."
            print(f"   Headers: {headers}")
            
            # Check for required columns
            required_cols = ["crop type", "crop yield", "unit", "climate drivers", "values", "experimental design", "location", "time"]
            header_lower = headers.lower()
            found_cols = [col for col in required_cols if col in header_lower]
            print(f"   Required columns found: {len(found_cols)}/{len(required_cols)}")
            
            missing_cols = [col for col in required_cols if col not in header_lower]
            if missing_cols and len(missing_cols) <= 4:
                print(f"   Missing columns: {missing_cols}")
            elif missing_cols:
                print(f"   Missing columns: {len(missing_cols)} total")
            
            if len(lines) > 1:
                data_rows = len(lines) - 1
                print(f"   Data rows: {data_rows}")
    
    # Show final status
    print(f"\n📋 Final Agent Status:")
    print(f"   Status Summary: {agent.status.get_status_summary()}")
    print(f"   Key Findings: {len(agent.status.key_findings)} total")
    if agent.status.key_findings:
        # Show only the most recent 2 key findings
        recent_findings = agent.status.key_findings[-2:]
        for finding in recent_findings:
            if len(finding) > 80:
                finding = finding[:80] + "..."
            print(f"     • {finding}")
        if len(agent.status.key_findings) > 2:
            print(f"     ... and {len(agent.status.key_findings) - 2} more findings")
    
    print(f"   Current Challenges: {len(agent.status.current_challenges)} total")
    if agent.status.current_challenges:
        # Show only the most recent 2 challenges
        recent_challenges = agent.status.current_challenges[-2:]
        for challenge in recent_challenges:
            if len(challenge) > 80:
                challenge = challenge[:80] + "..."
            print(f"     • {challenge}")
        if len(agent.status.current_challenges) > 2:
            print(f"     ... and {len(agent.status.current_challenges) - 2} more challenges")
    
    # Show workspace files created
    print(f"\n📁 Workspace Files Created:")
    workspace_files = agent.status.workspace_files or []
    if workspace_files:
        for file in workspace_files:
            print(f"   📄 {file}")
    else:
        print("   (No files created)")
    
    print(f"\n🎯 META-ANALYSIS EXTRACTION ASSESSMENT:")
    if csv_outputs and len(csv_outputs) > 0:
        print(f"   ✅ CSV extraction: SUCCESS")
        print(f"   ✅ Data format: Structured output achieved")
        print(f"   ✅ Multi-iteration workflow: COMPLETED")
    else:
        print(f"   ❌ CSV extraction: INCOMPLETE")
        print(f"   ⚠️  Consider adjusting goal or adding more specific file sources")
    
    print(f"\n🎉 COMPLETE META-ANALYSIS WORKFLOW TEST FINISHED!")
    print("=" * 60)


if __name__ == "__main__":
    main() 