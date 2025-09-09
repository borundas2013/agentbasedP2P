#!/usr/bin/env python3
"""
Demo script to show the clean, readable output format
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_agent import LLMStructureFineTuningAgent

def demo_clean_output():
    """Demonstrate the clean, readable output format."""
    
    # Test monomers that are NOT compatible but will get suggestions
    monomer1 = "CCCCBr"  # Butyl bromide (halide)
    monomer2 = "CCCCCC"  # Hexane (no functional groups)
    
    print("🔬 **DEMO: CLEAN READABLE OUTPUT**")
    print("=" * 60)
    print(f"Monomer 1: {monomer1} (Butyl bromide - halide)")
    print(f"Monomer 2: {monomer2} (Hexane - no functional groups)")
    print("=" * 60)
    
    # Test the make_compatible_tool
    test_input = f"Here are two monomers: monomer1 = {monomer1} and monomer2 = {monomer2}. Make them reaction compatible for polymerization."
    
    agent = LLMStructureFineTuningAgent()
    result = agent.process_user_input(test_input)
    
    print("\n📋 **CLEAN READABLE RESULT:**")
    print("-" * 60)
    print(result['response'])
    print("-" * 60)
    
    print("\n✅ **This is the clean, readable format you wanted!**")
    print("No more raw dictionary output with technical details.")

if __name__ == "__main__":
    demo_clean_output() 