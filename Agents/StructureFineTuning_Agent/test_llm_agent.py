#!/usr/bin/env python3
"""
Test script for LLM-Powered Structure Fine-Tuning Agent
Demonstrates various capabilities using the OpenAI fine-tuned model.
"""

from llm_agent import LLMStructureFineTuningAgent

def test_llm_agent_capabilities():
    """Test various capabilities of the LLM agent"""
    agent = LLMStructureFineTuningAgent()
    
    # Test cases with natural language inputs
    test_cases = [
        {
            "input": "Here are two monomers: monomer1 = CCNC1OC1Cc1ccccc1CCCCBr and monomer2 = CCCOOCC. remove CN group from monomer 1.",
            "description": "Remove CN group from monomer 1"
        },
        {
            "input": "Here are two monomers: monomer1 = O=C(OCC1CO1)C3CC2OC3CC2C(=O)OCC4CO4 and monomer2 = CCC2OC2COOCC. add [*]C(=O)O group to monomer 2.",
            "description": "Add carboxylic acid group to monomer 2"
        },
        {
            "input": "Here are two monomers: monomer1 = CCNC1OC1Cc1ccccc1CCCCBr and monomer2 = CCC2OC2COOCC. show the solubility properties of the given monomers.",
            "description": "Analyze solubility properties"
        },
        {
            "input": "Analyze the properties of CCO",
            "description": "Analyze molecular properties of ethanol"
        },
        {
            "input": "Make CC(=O)O and CCO compatible for polymerization",
            "description": "Make monomers reaction-compatible"
        },
        {
            "input": "Validate the pair CCO c1ccccc1",
            "description": "Validate a pair of SMILES"
        },
        {
            "input": "Calculate compatibility between CC(=O)O and CCO",
            "description": "Calculate compatibility score"
        },
        {
            "input": "Analyze functional groups in CCO",
            "description": "Analyze functional groups"
        },
        {
            "input": "Optimize groups for solubility in monomer 1",
            "description": "Optimize functional groups for solubility"
        },
        {
            "input": "Generate a comprehensive report for CC(=O)O and CCO",
            "description": "Generate comprehensive report"
        }
    ]
    
    print("Testing LLM-Powered Structure Fine-Tuning Agent")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Input: {test_case['input']}")
        print("-" * 50)
        
        try:
            result = agent.process_user_input(test_case['input'])
            
            if "help" in result:
                print("Help information retrieved successfully")
            elif "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Response: {result['response']}")
                if result.get('tool_calls'):
                    print(f"Tools called: {result['tool_calls']}")
                
        except Exception as e:
            print(f"Exception: {str(e)}")
        
        print()
    
    print("LLM agent testing completed!")

def test_interactive_mode():
    """Test the agent in interactive mode with predefined inputs"""
    agent = LLMStructureFineTuningAgent()
    
    print("\nInteractive Mode Test")
    print("=" * 40)
    
    # Simulate user inputs
    test_inputs = [
        "help",
        "Here are two monomers: monomer1 = CCNC1OC1Cc1ccccc1CCCCBr and monomer2 = CCCOOCC. remove CN group from monomer 1.",
        "Analyze the properties of CCO",
        "Make CC(=O)O and CCO compatible for polymerization",
        "quit"
    ]
    
    for user_input in test_inputs:
        print(f"\nUser input: {user_input}")
        
        if user_input.lower() == "quit":
            print("Exiting...")
            break
        
        try:
            result = agent.process_user_input(user_input)
            
            if "help" in result:
                print("Help information:")
                print(result["help"][:300] + "...")  # Show first 300 chars
            elif "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Response: {result['response']}")
                if result.get('tool_calls'):
                    print(f"Tools called: {result['tool_calls']}")
                    
        except Exception as e:
            print(f"Exception: {str(e)}")

def test_complex_scenarios():
    """Test more complex scenarios"""
    agent = LLMStructureFineTuningAgent()
    
    print("\nComplex Scenarios Test")
    print("=" * 40)
    
    complex_tests = [
        {
            "input": "Here are two monomers: monomer1 = CCNC1OC1Cc1ccccc1CCCCBr and monomer2 = CCC2OC2COOCC. remove O[O] bonds from monomer 2.",
            "description": "Remove specific bonds from monomer 2"
        },
        {
            "input": "Here are two monomers: monomer1 = CCNC1OC1Cc1ccccc1CCCCBr and monomer2 = CCC2OC2COOCC. add [*]Cl group to monomer 1.",
            "description": "Add chlorine group to monomer 1"
        },
        {
            "input": "Here are two monomers: monomer1 = CCNC1OC1Cc1ccccc1CCCCBr and monomer2 = CCC2OC2COOCC. remove c1ccccc1 group from monomer 1.",
            "description": "Remove benzene ring from monomer 1"
        },
        {
            "input": "Here are two monomers: monomer1 = O=C(OCC1CO1)C3CC2OC3CC2C(=O)OCC4CO4 and monomer2 = CCC2OC2COOCC. add [*]c1ccccc1 group to monomer2.",
            "description": "Add benzene ring to monomer 2"
        }
    ]
    
    for i, test_case in enumerate(complex_tests, 1):
        print(f"\nComplex Test {i}: {test_case['description']}")
        print(f"Input: {test_case['input']}")
        print("-" * 50)
        
        try:
            result = agent.process_user_input(test_case['input'])
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Response: {result['response']}")
                if result.get('tool_calls'):
                    print(f"Tools called: {result['tool_calls']}")
                    
        except Exception as e:
            print(f"Exception: {str(e)}")
        
        print()

def test_15_tools_comprehensive():
    """Test 15 different tools/operations for two monomers"""
    agent = LLMStructureFineTuningAgent()
    
    # Define two test monomers
    monomer1 = "CCNC1OC1Cc1ccccc1CCCCBr"  # Complex monomer with benzene ring and bromine
    monomer2 = "CCC2OC2COOCC"  # Simpler monomer with ether and ester groups
   
    
    print("\n" + "="*80)
    print("COMPREHENSIVE 15-TOOL TEST FOR TWO MONOMERS")
    print("="*80)
    print(f"Monomer 1: {monomer1}")
    print(f"Monomer 2: {monomer2}")
    print("="*80)
    
    # Test cases covering 15 different tools/operations
    comprehensive_tests = [
        {
            "input": f"Here are two monomers: monomer1 = {monomer1} and monomer2 = {monomer2}. Analyze the properties of monomer 1.",
            "description": "1. Analyze Molecular Properties",
            "tool": "analyze_properties"
        },
        {
            "input": f"Here are two monomers: monomer1 = {monomer1} and monomer2 = {monomer2}. Validate this pair.",
            "description": "2. Validate SMILES Pair",
            "tool": "validate_pair"
        },
        {
            "input": f"Here are two monomers: monomer1 = {monomer1} and monomer2 = {monomer2}. Calculate compatibility between them.",
            "description": "3. Calculate Compatibility Score",
            "tool": "calculate_compatibility"
        },
        {
            "input": f"Here are two monomers: monomer1 = {monomer1} and monomer2 = {monomer2}. What structural improvements can be made to these monomers?",
            "description": "4. Suggest Structural Improvements",
            "tool": "suggest_improvements"
        },
        {
            "input": f"Here are two monomers: monomer1 = {monomer1} and monomer2 = {monomer2}. Remove CN group from monomer 1.",
            "description": "5. Remove Bond/Group (CN from monomer 1)",
            "tool": "remove_bond"
        },
        {
            "input": f"Here are two monomers: monomer1 = {monomer1} and monomer2 = {monomer2}. Add [*]C(=O)O group to monomer 2.",
            "description": "6. Add Group (Carboxylic acid to monomer 2)",
            "tool": "add_group"
        },
        {
            "input": f"Here are two monomers: monomer1 = {monomer1} and monomer2 = {monomer2}. Modify the benzene ring in monomer 1 to saturate it.",
            "description": "7. Modify Ring System (Saturate benzene)",
            "tool": "modify_ring"
        },
        {
            "input": f"Here are two monomers: monomer1 = {monomer1} and monomer2 = {monomer2}. Create alkyl_variants for monomer 1.",
            "description": "8. Create Structural Variants",
            "tool": "create_variants"
        },
        {
            "input": f"Here are two monomers: monomer1 = {monomer1} and monomer2 = {monomer2}. Optimize functional groups for solubility in monomer 1.",
            "description": "9. Optimize Functional Groups",
            "tool": "optimize_groups"
        },
        {
            "input": f"Here are two monomers: monomer1 = {monomer1} and monomer2 = {monomer2}. Analyze functional groups in monomer 1.",
            "description": "10. Analyze Functional Groups",
            "tool": "analyze_groups"
        },
        {
            "input": f"Here are two monomers: monomer1 = {monomer1} and monomer2 = {monomer2}. Modify stereochemistry by racemizing in monomer 1.",
            "description": "11. Modify Stereochemistry",
            "tool": "modify_stereochemistry"
        },
        {
            "input": f"Here are two monomers: monomer1 = {monomer1} and monomer2 = {monomer2}. Make these monomers reaction perfect.",
            "description": "13. Make Monomers Reaction Compatible",
            "tool": "make_compatible"
        },
        
        {
            "input": f"Here are two monomers: monomer1 = {monomer1} and monomer2 = {monomer2}. Generate a comprehensive report.",
            "description": "15. Generate Comprehensive Report",
            "tool": "generate_report"
        }
    ]
    
    successful_tests = 0
    total_tests = len(comprehensive_tests)
                
    i=11
    print(f"\n{'='*60}")
    print(f"TEST {i+1:2d}/13: {comprehensive_tests[i]['description']}")
    print(f"Tool: {comprehensive_tests[i]['tool']}")
    print(f"{'='*60}")
    print(f"Input: {comprehensive_tests[i]['input']}")
    print("-" * 60)
    
    try:
        result = agent.process_user_input(comprehensive_tests[i]['input'])
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        elif "help" in result:
            print("ℹ️  Help information retrieved")
            successful_tests += 1
        else:
            print("✅ Success!")
            print(f"Response: {result['response']}")  # Show first 300 chars
            if result.get('tool_calls'):
                print(f"🔧 Tools called: {result['tool_calls']}")
            successful_tests += 1
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
    
    # for i, test_case in enumerate(comprehensive_tests, 1):
    #     print(f"\n{'='*60}")
    #     print(f"TEST {i:2d}/15: {test_case['description']}")
    #     print(f"Tool: {test_case['tool']}")
    #     print(f"{'='*60}")
    #     print(f"Input: {test_case['input']}")
    #     print("-" * 60)
        
    #     try:
    #         result = agent.process_user_input(test_case['input'])
            
    #         if "error" in result:
    #             print(f"❌ Error: {result['error']}")
    #         elif "help" in result:
    #             print("ℹ️  Help information retrieved")
    #             successful_tests += 1
    #         else:
    #             print("✅ Success!")
    #             print(f"Response: {result['response'][:300]}...")  # Show first 300 chars
    #             if result.get('tool_calls'):
    #                 print(f"🔧 Tools called: {result['tool_calls']}")
    #             successful_tests += 1
                
    #     except Exception as e:
    #         print(f"❌ Exception: {str(e)}")
        
    #     print()
    
    # Summary
    # print("\n" + "="*80)
    # print("COMPREHENSIVE TEST SUMMARY")
    # print("="*80)
    # print(f"Total tests: {total_tests}")
    # print(f"Successful: {successful_tests}")
    # print(f"Failed: {total_tests - successful_tests}")
    # print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
    # print("="*80)
    
    # # Tool coverage summary
    # print("\nTOOL COVERAGE:")
    # print("-" * 40)
    # tools_covered = [
    #     "1. Analyze Molecular Properties",
    #     "2. Validate SMILES Pair", 
    #     "3. Calculate Compatibility Score",
    #     "4. Suggest Structural Improvements",
    #     "5. Remove Bond/Group",
    #     "6. Add Group",
    #     "7. Modify Ring System",
    #     "8. Create Structural Variants",
    #     "9. Optimize Functional Groups",
    #     "10. Analyze Functional Groups",
    #     "11. Modify Stereochemistry",
    #     "12. Modify Bond Types",
    #     "13. Make Monomers Reaction Compatible",
    #     "14. Generate Comprehensive Report"
    # ]
    
    # for tool in tools_covered:
    #     print(f"✓ {tool}")
    
    # print("\n" + "="*80)

def test_incompatible_monomers_with_suggestions():
    """
    Test case with two monomers that are NOT reaction compatible 
    but the make_compatible_tool will provide suggestions.
    """
    # Monomers that are NOT directly compatible but will get suggestions
    monomer1 = "CCCCBr"  # Butyl bromide (halide)
    monomer2 = "CCCCCC"  # Hexane (no functional groups)
    
    print(f"\n=== Testing Incompatible Monomers with Suggestions ===")
    print(f"Monomer 1: {monomer1} (Butyl bromide - halide)")
    print(f"Monomer 2: {monomer2} (Hexane - no functional groups)")
    print(f"Expected: These are NOT compatible, but should get suggestions")
    
    # Test the make_compatible_tool
    test_input = f"Here are two monomers: monomer1 = {monomer1} and monomer2 = {monomer2}. Make them reaction compatible for polymerization."
    
    agent = LLMStructureFineTuningAgent()
    result = agent.process_user_input(test_input)
    
    print(f"\n📋 **CLEAN READABLE RESULT:**")
    print("-" * 60)
    print(result['response'])
    print("-" * 60)

# #!/usr/bin/env python3
# """
# Test script to demonstrate suggestions for both monomers
# """
# import sys, os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from llm_agent import LLMStructureFineTuningAgent

# def test_both_monomers_suggestions():
#     agent = LLMStructureFineTuningAgent()
    
#     print("=" * 80)
#     print("TEST CASE 1: Monomer 1 has halide, Monomer 2 has no functional groups")
#     print("=" * 80)
#     monomer1 = "CCCCBr"
#     monomer2 = "CCCCCC"
#     print(f"Monomer 1: {monomer1}")
#     print(f"Monomer 2: {monomer2}")
#     print("\nResult:")
#     result = agent.process_user_input(f"Here are two monomers: monomer1 = {monomer1} and monomer2 = {monomer2}. Make them reaction compatible for polymerization.")
#     print(result['response'])
    
#     print("\n" + "=" * 80)
#     print("TEST CASE 2: Monomer 1 has no functional groups, Monomer 2 has halide")
#     print("=" * 80)
#     monomer1 = "CCCCCC"
#     monomer2 = "CCCCBr"
#     print(f"Monomer 1: {monomer1}")
#     print(f"Monomer 2: {monomer2}")
#     print("\nResult:")
#     result = agent.process_user_input(f"Here are two monomers: monomer1 = {monomer1} and monomer2 = {monomer2}. Make them reaction compatible for polymerization.")
#     print(result['response'])
    
#     print("\n" + "=" * 80)
#     print("TEST CASE 3: BOTH monomers have no functional groups (should suggest for both)")
#     print("=" * 80)
#     monomer1 = "CCCCCC"
#     monomer2 = "CCCCCC"
#     print(f"Monomer 1: {monomer1}")
#     print(f"Monomer 2: {monomer2}")
#     print("\nResult:")
#     result = agent.process_user_input(f"Here are two monomers: monomer1 = {monomer1} and monomer2 = {monomer2}. Make them reaction compatible for polymerization.")
#     print(result['response'])
    
#     print("\n" + "=" * 80)
#     print("TEST CASE 4: Both monomers have conflicting groups (should suggest for both)")
#     print("=" * 80)
#     monomer1 = "CCCCBr"
#     monomer2 = "CCCCBr"
#     print(f"Monomer 1: {monomer1}")
#     print(f"Monomer 2: {monomer2}")
#     print("\nResult:")
#     result = agent.process_user_input(f"Here are two monomers: monomer1 = {monomer1} and monomer2 = {monomer2}. Make them reaction compatible for polymerization.")
#     print(result['response'])



if __name__ == "__main__":
    print("LLM-Powered Structure Fine-Tuning Agent Test Suite")
    print("=" * 60)
    
    # Test basic capabilities
    #test_llm_agent_capabilities()
    
    # Test interactive mode
    #test_interactive_mode()
    
    # Test complex scenarios
    #test_complex_scenarios()
    
    # Test comprehensive 15 tools
    test_15_tools_comprehensive()
    
    # Test incompatible monomers with suggestions
   # test_both_monomers_suggestions()
    
    print("\nAll tests completed!") 