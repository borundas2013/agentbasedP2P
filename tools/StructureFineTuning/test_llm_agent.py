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

if __name__ == "__main__":
    print("LLM-Powered Structure Fine-Tuning Agent Test Suite")
    print("=" * 60)
    
    # Test basic capabilities
    test_llm_agent_capabilities()
    
    # Test interactive mode
    test_interactive_mode()
    
    # Test complex scenarios
    test_complex_scenarios()
    
    print("\nAll tests completed!") 