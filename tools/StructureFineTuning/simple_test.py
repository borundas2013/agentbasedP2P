#!/usr/bin/env python3
"""
Simple test script for LLM-Powered Structure Fine-Tuning Agent
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

# Import the agent
from llm_agent_fixed import LLMStructureFineTuningAgent

def test_agent():
    """Test the LLM agent with a simple example"""
    print("Testing LLM-Powered Structure Fine-Tuning Agent")
    print("=" * 50)
    
    try:
        # Create agent instance
        agent = LLMStructureFineTuningAgent()
        print("✓ Agent created successfully")
        
        # Test help functionality
        result = agent.process_user_input("help")
        if "help" in result:
            print("✓ Help functionality works")
            print("Help content preview:")
            print(result["help"][:200] + "...")
        else:
            print("✗ Help functionality failed")
        
        # Test a simple property analysis
        test_input = "Analyze the properties of CCO"
        print(f"\nTesting: {test_input}")
        result = agent.process_user_input(test_input)
        
        if "error" in result:
            print(f"✗ Error: {result['error']}")
        else:
            print("✓ Agent processed input successfully")
            print(f"Response: {result['response'][:100]}...")
            if result.get('tool_calls'):
                print(f"Tools called: {result['tool_calls']}")
        
        print("\nAgent test completed!")
        
    except Exception as e:
        print(f"✗ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent() 