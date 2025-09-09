"""
LLM-Powered Structure Fine-Tuning Agent using LangChain
An intelligent agent that uses OpenAI fine-tuned model for natural language understanding and task execution.
"""

import sys
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import re

# Add the parent directory to the path to import the structure fine-tuning functions
sys.path.append(os.path.dirname(__file__))

# Import all the structure fine-tuning functions
from structurefinetune import (
    analyze_molecular_properties,
    validate_smiles_pair,
    calculate_compatibility_score,
    suggest_structural_improvements,
    remove_bond_by_smarts,
    add_group_by_smarts,
    modify_ring_system,
    create_structural_variants,
    optimize_functional_groups,
    analyze_functional_groups,
    has_functional_group,
    suggest_functional_group_addition,
    suggest_functional_group_removal,
    modify_stereochemistry,
    modify_bond_types,
    make_monomers_reaction_compatible,
    compare_structures,
    generate_structure_report
)

# LangChain imports
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# API Key (you can move this to environment variables)

class TaskType(Enum):
    """Enumeration of available tasks"""
    ANALYZE_PROPERTIES = "analyze_properties"
    VALIDATE_PAIR = "validate_pair"
    CALCULATE_COMPATIBILITY = "calculate_compatibility"
    SUGGEST_IMPROVEMENTS = "suggest_improvements"
    REMOVE_BOND = "remove_bond"
    ADD_GROUP = "add_group"
    MODIFY_RING = "modify_ring"
    CREATE_VARIANTS = "create_variants"
    OPTIMIZE_GROUPS = "optimize_groups"
    ANALYZE_GROUPS = "analyze_groups"
    MODIFY_STEREOCHEMISTRY = "modify_stereochemistry"
    MODIFY_BONDS = "modify_bonds"
    MAKE_COMPATIBLE = "make_compatible"
    COMPARE_STRUCTURES = "compare_structures"
    GENERATE_REPORT = "generate_report"

# Pydantic models for tool inputs
class AnalyzePropertiesTool(BaseModel):
    smiles: str = Field(..., description="SMILES string of the molecule to analyze")

class ValidatePairTool(BaseModel):
    smiles1: str = Field(..., description="First SMILES string")
    smiles2: str = Field(..., description="Second SMILES string")

class CalculateCompatibilityTool(BaseModel):
    smiles1: str = Field(..., description="First monomer SMILES")
    smiles2: str = Field(..., description="Second monomer SMILES")

class SuggestImprovementsTool(BaseModel):
    smiles1: str = Field(..., description="First monomer SMILES")
    smiles2: str = Field(..., description="Second monomer SMILES")

class RemoveBondTool(BaseModel):
    smiles1: str = Field(..., description="First monomer SMILES")
    smiles2: str = Field(..., description="Second monomer SMILES")
    bond_smarts: str = Field(..., description="SMARTS pattern for bond to remove")
    target_monomer: str = Field("1", description="Which monomer to modify ('1' or '2')")

class AddGroupTool(BaseModel):
    smiles1: str = Field(..., description="First monomer SMILES")
    smiles2: str = Field(..., description="Second monomer SMILES")
    group_smarts: str = Field(..., description="SMARTS pattern for group to add (must contain [*])")
    target_monomer: str = Field("1", description="Which monomer to modify ('1' or '2')")
    #attachment_atom_idx: int = Field(0, description="Atom index for attachment")

class ModifyRingTool(BaseModel):
    smiles1: str = Field(..., description="First monomer SMILES")
    smiles2: str = Field(..., description="Second monomer SMILES")
    ring_smarts: str = Field(..., description="SMARTS pattern for ring to modify")
    modification_type: str = Field(..., description="Type of modification: 'aromatize' or 'saturate'")
    target_monomer: str = Field("1", description="Which monomer to modify ('1' or '2')")

class CreateVariantsTool(BaseModel):
    smiles1: str = Field(..., description="First monomer SMILES")
    smiles2: str = Field(..., description="Second monomer SMILES")
    variant_type: str = Field(..., description="Type of variant: 'carbon_chain' or 'alkyl_chain'")
    target_monomer: str = Field("1", description="Which monomer to modify ('1' or '2')")

class OptimizeGroupsTool(BaseModel):
    smiles1: str = Field(..., description="First monomer SMILES")
    smiles2: str = Field(..., description="Second monomer SMILES")
    optimization_type: str = Field(..., description="Optimization type: 'improve_solubility', 'improve_stability', or 'add_polar_groups'")
    target_monomer: str = Field("1", description="Which monomer to modify ('1' or '2')")

class AnalyzeGroupsTool(BaseModel):
    smiles: str = Field(..., description="SMILES string to analyze for functional groups")

class ModifyStereochemistryTool(BaseModel):
    smiles1: str = Field(..., description="First monomer SMILES")
    smiles2: str = Field(..., description="Second monomer SMILES")
    stereochemistry_type: str = Field(..., description="Type of modification: 'invert' or 'racemize'")
    target_monomer: str = Field("1", description="Which monomer to modify ('1' or '2')")

class ModifyBondsTool(BaseModel):
    smiles1: str = Field(..., description="First monomer SMILES")
    smiles2: str = Field(..., description="Second monomer SMILES")
    smarts_pattern: str = Field(..., description="SMARTS pattern for bonds to modify")
    new_bond_type: str = Field(..., description="New bond type: 'single', 'double', 'triple', or 'aromatic'")
    target_monomer: str = Field("1", description="Which monomer to modify ('1' or '2')")

class MakeCompatibleTool(BaseModel):
    smiles1: str = Field(..., description="First monomer SMILES")
    smiles2: str = Field(..., description="Second monomer SMILES")

class CompareStructuresTool(BaseModel):
    smiles1: str = Field(..., description="Original first monomer SMILES")
    smiles2: str = Field(..., description="Original second monomer SMILES")
    smiles1_modified: str = Field(..., description="Modified first monomer SMILES")
    smiles2_modified: str = Field(..., description="Modified second monomer SMILES")

class GenerateReportTool(BaseModel):
    smiles1: str = Field(..., description="First monomer SMILES")
    smiles2: str = Field(..., description="Second monomer SMILES")

# Tool definitions using @tool decorator
@tool
def analyze_properties_tool(input: AnalyzePropertiesTool) -> str:
    """
    [PROPERTIES_ANALYSIS_TOOL] Analyze molecular properties of a single molecule. 
    Use this tool when asked to "analyze properties", "analyze molecular properties", "show properties", or "what are the properties" of a single molecule.
    This tool analyzes ONE molecule at a time, not pairs. 
    DO NOT use this tool for improvement suggestions or structural modifications.
    
    Parameters:
    - smiles (str): SMILES string of the molecule to analyze
    
    Returns:
    - str: Detailed molecular properties analysis
    """
    return analyze_molecular_properties(input.smiles)

@tool
def validate_pair_tool(input: ValidatePairTool) -> str:
    """
    Validate a pair of SMILES strings for compatibility and structure.
    
    Parameters:
    - smiles1 (str): First SMILES string
    - smiles2 (str): Second SMILES string
    
    Returns:
    - str: Validation results and compatibility assessment
    """
    return validate_smiles_pair(input.smiles1, input.smiles2)

@tool
def calculate_compatibility_tool(input: CalculateCompatibilityTool) -> str:
    """
    Calculate compatibility score between two monomers.
    
    Parameters:
    - smiles1 (str): First monomer SMILES
    - smiles2 (str): Second monomer SMILES
    
    Returns:
    - str: Compatibility score and analysis
    """
    from rdkit import Chem
    mol1 = Chem.MolFromSmiles(input.smiles1)
    mol2 = Chem.MolFromSmiles(input.smiles2)
    if mol1 and mol2:
        score = calculate_compatibility_score(mol1, mol2)
        return f"Compatibility score: {score:.3f}"
    else:
        return "Error: Invalid SMILES strings"

@tool
def suggest_improvements_tool(input: SuggestImprovementsTool) -> str:
    """
    [STRUCTURAL_IMPROVEMENTS_TOOL] Suggest structural improvements and modifications for a pair of monomers. 
    Use this tool ONLY when asked to "suggest structural improvements", "suggest improvements", "recommend structural modifications", "what improvements can be made", or "what structural improvements".
    This tool provides specific structural improvement suggestions, NOT property analysis.
    DO NOT use this tool for property analysis requests.
    
    Parameters:
    - smiles1 (str): First monomer SMILES
    - smiles2 (str): Second monomer SMILES
    
    Returns:
    - str: Suggested improvements and modifications
    """
    result = suggest_structural_improvements(input.smiles1, input.smiles2)
    
    if isinstance(result, dict):
        if "error" in result:
            return f"Error: {result['error']}"
        
        suggestions = result.get("suggestions", [])
        monomer1_analysis = result.get("monomer1_analysis", {})
        monomer2_analysis = result.get("monomer2_analysis", {})
        
        response = "Structural Improvement Suggestions:\n\n"
        
        if suggestions:
            response += "Suggested Improvements:\n"
            for i, suggestion in enumerate(suggestions, 1):
                response += f"{i}. {suggestion}\n"
            response += "\n"
        else:
            response += "No specific improvements suggested at this time.\n\n"
        
        response += "Monomer 1 Analysis:\n"
        for key, value in monomer1_analysis.items():
            response += f"  {key}: {value:.2f}\n"
        
        response += "\nMonomer 2 Analysis:\n"
        for key, value in monomer2_analysis.items():
            response += f"  {key}: {value:.2f}\n"
        
        return response
    else:
        return str(result)

@tool
def remove_bond_tool(input: RemoveBondTool) -> str:
    """
    Remove bonds matching a SMARTS pattern from a monomer.
    
    Parameters:
    - smiles1 (str): First monomer SMILES (extract from "monomer1 = SMILES1")
    - smiles2 (str): Second monomer SMILES (extract from "monomer2 = SMILES2") 
    - bond_smarts (str): SMARTS pattern for bonds to remove 
    - target_monomer (str): Which monomer to modify ('1' or '2', extract from "to monomer 2")
    
    Returns:
    - str: Modified monomers with bond removal
    """
    return str(remove_bond_by_smarts(input.smiles1, input.smiles2, input.bond_smarts, input.target_monomer))

@tool
def add_group_tool(input: AddGroupTool) -> str:
    """
    Add a functional group to a monomer.
    Use this tool ONLY when asked to "add group".
    This tool adds a specific functional group to a target monomer.
    
   
    
    Parameters:
    - smiles1 (str): First monomer SMILES (extract from "monomer1 = SMILES1")
    - smiles2 (str): Second monomer SMILES (extract from "monomer2 = SMILES2") 
    - group_smarts (str): SMARTS pattern for group to add (extract from "add [*]C(=O)O group")
    - target_monomer (str): Which monomer to modify ('1' or '2', extract from "to monomer 2")
    
    Returns:
    - str: Modified monomers with group addition
    """
    try:
        result = add_group_by_smarts(input.smiles1, input.smiles2, input.group_smarts, input.target_monomer)
        
        if isinstance(result, dict):
            if "error" in result:
                return f"Error: {result['error']}"
            return str(result)
        else:
            return str(result)
            
    except Exception as e:
        return f"Error adding group: {str(e)}"

@tool
def modify_ring_tool(input: ModifyRingTool) -> str:
    """
    Modify ring systems in a monomer.
    
    Parameters:
    - smiles1 (str): First monomer SMILES
    - smiles2 (str): Second monomer SMILES
    - ring_smarts (str): SMARTS pattern for ring to modify
    - modification_type (str): Type of modification ('aromatize' or 'saturate')
    - target_monomer (str): Which monomer to modify ('1' or '2')
    
    Returns:
    - str: Modified monomers with ring modifications
    """
    return modify_ring_system(input.smiles1, input.smiles2, input.ring_smarts, input.modification_type, input.target_monomer)

@tool
def create_variants_tool(input: CreateVariantsTool) -> str:
    """
    Create structural variants of a monomer.
    
    Parameters:
    - smiles1 (str): First monomer SMILES
    - smiles2 (str): Second monomer SMILES
    - variant_type (str): Type of variant ('carbon_chain' or 'alkyl_chain')
    - target_monomer (str): Which monomer to modify ('1' or '2')
    
    Returns:
    - str: Generated structural variants
    """
    variants = create_structural_variants(input.smiles1, input.smiles2, input.variant_type, input.target_monomer)
    return f"Generated variants: {variants}"

@tool
def optimize_groups_tool(input: OptimizeGroupsTool) -> str:
    """
    Optimize functional groups in a monomer.
    
    Parameters:
    - smiles1 (str): First monomer SMILES
    - smiles2 (str): Second monomer SMILES
    - optimization_type (str): Type of optimization ('improve_solubility', 'improve_stability', 'add_polar_groups')
    - target_monomer (str): Which monomer to modify ('1' or '2')
    
    Returns:
    - str: Optimized monomers with improved functional groups
    """
    return optimize_functional_groups(input.smiles1, input.smiles2, input.optimization_type, input.target_monomer)

@tool
def analyze_groups_tool(input: AnalyzeGroupsTool) -> str:
    """
    [ANALYZE_GROUPS_TOOL] Analyze functional groups in a molecule.
    Use this tool when asked to "analyze functional groups", "analyze groups", or "what functional groups".
    This tool identifies and counts functional groups in a molecule.
    
    Parameters:
    - smiles (str): SMILES string to analyze
    
    Returns:
    - str: Functional group analysis results
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(input.smiles)
    if mol:
        fg_dict = analyze_functional_groups(mol)
        
        # Format the output nicely
        result = f"Functional Group Analysis for {input.smiles}:\n\n"
        
        # Group functional groups by type
        polar_groups = ["hydroxyl", "carboxylic_acid", "amine", "nitrile"]
        unsaturated_groups = ["vinyl", "acrylate"]
        ring_groups = ["epoxide", "cyclic_ether"]
        carbonyl_groups = ["ester", "amide", "aldehyde", "ketone"]
        other_groups = ["halide", "imine"]
        
        # Polar groups
        polar_found = [(name, count) for name, count in fg_dict.items() if name in polar_groups and count > 0]
        if polar_found:
            result += "Polar Groups:\n"
            for name, count in polar_found:
                result += f"  - {name.replace('_', ' ').title()}: {count} instance(s)\n"
            result += "\n"
        
        # Ring groups
        ring_found = [(name, count) for name, count in fg_dict.items() if name in ring_groups and count > 0]
        if ring_found:
            result += "Ring Systems:\n"
            for name, count in ring_found:
                result += f"  - {name.replace('_', ' ').title()}: {count} instance(s)\n"
            result += "\n"
        
        # Carbonyl groups
        carbonyl_found = [(name, count) for name, count in fg_dict.items() if name in carbonyl_groups and count > 0]
        if carbonyl_found:
            result += "Carbonyl Groups:\n"
            for name, count in carbonyl_found:
                result += f"  - {name.replace('_', ' ').title()}: {count} instance(s)\n"
            result += "\n"
        
        # Unsaturated groups
        unsaturated_found = [(name, count) for name, count in fg_dict.items() if name in unsaturated_groups and count > 0]
        if unsaturated_found:
            result += "Unsaturated Groups:\n"
            for name, count in unsaturated_found:
                result += f"  - {name.replace('_', ' ').title()}: {count} instance(s)\n"
            result += "\n"
        
        # Other groups
        other_found = [(name, count) for name, count in fg_dict.items() if name in other_groups and count > 0]
        if other_found:
            result += "Other Groups:\n"
            for name, count in other_found:
                result += f"  - {name.replace('_', ' ').title()}: {count} instance(s)\n"
            result += "\n"
        
        # Check if no functional groups found
        total_groups = sum(count for count in fg_dict.values() if count > 0)
        if total_groups == 0:
            result += "No specific functional groups detected. This appears to be a simple hydrocarbon.\n"
        
        return result
    else:
        return "Error: Invalid SMILES string"

@tool
def modify_stereochemistry_tool(input: ModifyStereochemistryTool) -> str:
    """
    [STEREOCHEMISTRY_TOOL] Modify stereochemistry of chiral centers in a monomer.
    Use this tool ONLY when asked to "modify stereochemistry", "invert stereochemistry", "racemize", "invert chiral centers", or "change stereochemistry".
    This tool specifically handles chiral center modifications (invert/racemize). Do NOT use for general structural modifications.
    
    Parameters:
    - smiles1 (str): First monomer SMILES (extract from "monomer1 = SMILES1")
    - smiles2 (str): Second monomer SMILES (extract from "monomer2 = SMILES2")
    - stereochemistry_type (str): Type of modification ('invert' or 'racemize', extract from "invert stereochemistry", "invert chiral centers", "racemize", "racemizing", "inverting")
    - target_monomer (str): Which monomer to modify ('1' or '2', extract from "to monomer 2", "to monomer 1", "in monomer 1", "in monomer 2", "monomer 1", "monomer 2")
    
    Returns:
    - str: Modified monomers with stereochemistry changes
    """
    try:
        result = modify_stereochemistry(input.smiles1, input.smiles2, input.stereochemistry_type, input.target_monomer)
        
        if isinstance(result, dict):
            if "error" in result:
                return f"Error: {result['error']}"
            return str(result)
        else:
            return str(result)
            
    except Exception as e:
        return f"Error modifying stereochemistry: {str(e)}"

# @tool
# def modify_bonds_tool(input: ModifyBondsTool) -> str:
#     """
#     [BOND_MODIFICATION_TOOL] Modify bond types in a monomer.
#     Use this tool ONLY when asked to "modify bonds", "change bond types", "convert bonds", or "modify bond types".
#     This tool specifically handles bond type modifications (single/double/triple/aromatic). Do NOT use for general structural modifications.
    
#     Parameters:
#     - smiles1 (str): First monomer SMILES (extract from "monomer1 = SMILES1")
#     - smiles2 (str): Second monomer SMILES (extract from "monomer2 = SMILES2")
#     - smarts_pattern (str): SMARTS pattern for bonds to modify (extract from "convert single bonds", "convert C-C bonds", "convert double bonds")
#     - new_bond_type (str): New bond type ('single', 'double', 'triple', 'aromatic', extract from "to double bonds", "to single bonds", "to triple bonds")
#     - target_monomer (str): Which monomer to modify ('1' or '2', extract from "in monomer 2", "in monomer 1", "to monomer 2", "to monomer 1")
    
#     Returns:
#     - str: Modified monomers with bond type changes
#     """
#     try:
#         result = modify_bond_types(input.smiles1, input.smiles2, input.smarts_pattern, input.new_bond_type, input.target_monomer)
        
#         if isinstance(result, dict):
#             if "error" in result:
#                 return f"Error: {result['error']}"
#             return str(result)
#         else:
#             return str(result)
            
#     except Exception as e:
#         return f"Error modifying bonds: {str(e)}"

@tool
def make_compatible_tool(input: MakeCompatibleTool) -> str:
    """
    [COMPATIBILITY_TOOL] Make two monomers reaction-compatible for polymerization. 
    Use this tool when asked to "make compatible", "make reaction compatible", "make compatible for polymerization", "make reaction perfect", or "make them compatible".
    This tool focuses on making monomers compatible for polymerization reactions. Do NOT use for general improvement suggestions.
    
    Parameters:
    - smiles1 (str): First monomer SMILES (extract from "monomer1 = SMILES1")
    - smiles2 (str): Second monomer SMILES (extract from "monomer2 = SMILES2")
    
    Returns:
    - str: Compatibility analysis and suggested modifications
    """
    try:
        result = make_monomers_reaction_compatible(input.smiles1, input.smiles2)
        
        if isinstance(result, dict):
            if "error" in result:
                return f"Error: {result['error']}"
            
            # Format the response in a readable way
            response = "🔬 **COMPATIBILITY ANALYSIS**\n\n"
            
            # Functional groups analysis
            response += "**Functional Groups:**\n"
            fg1 = result.get("functional_groups_1", {})
            fg2 = result.get("functional_groups_2", {})
            
            # Format monomer 1 functional groups
            fg1_list = [f"{group}: {count}" for group, count in fg1.items() if count > 0]
            response += f"• Monomer 1 ({result.get('monomer1', 'N/A')}): {', '.join(fg1_list) if fg1_list else 'No functional groups'}\n"
            
            # Format monomer 2 functional groups
            fg2_list = [f"{group}: {count}" for group, count in fg2.items() if count > 0]
            response += f"• Monomer 2 ({result.get('monomer2', 'N/A')}): {', '.join(fg2_list) if fg2_list else 'No functional groups'}\n\n"
            
            # Compatibility status
            if result.get("reaction_types"):
                response += "✅ **COMPATIBLE FOR POLYMERIZATION**\n"
                response += "**Reaction Types:**\n"
                for reaction_type in result["reaction_types"]:
                    response += f"• {reaction_type}\n"
            else:
                response += "❌ **NOT DIRECTLY COMPATIBLE**\n"
                if result.get("compatibility_issues"):
                    response += "**Issues:**\n"
                    for issue in result["compatibility_issues"]:
                        response += f"• {issue}\n"
            
            # Suggestions
            if result.get("suggestions"):
                response += "\n💡 **SUGGESTIONS TO MAKE COMPATIBLE:**\n"
                for i, suggestion in enumerate(result["suggestions"], 1):
                    response += f"{i}. {suggestion}\n"
            
            return response
        else:
            return str(result)
            
    except Exception as e:
        return f"Error making compatible: {str(e)}"

@tool
def compare_structures_tool(input: CompareStructuresTool) -> str:
    """
    Compare original and modified structures.
    
    Parameters:
    - smiles1 (str): Original first monomer SMILES
    - smiles2 (str): Original second monomer SMILES
    - smiles1_modified (str): Modified first monomer SMILES
    - smiles2_modified (str): Modified second monomer SMILES
    
    Returns:
    - str: Structural comparison analysis
    """
    return compare_structures(input.smiles1, input.smiles2, input.smiles1_modified, input.smiles2_modified)

@tool
def generate_report_tool(input: GenerateReportTool) -> str:
    """
    Generate a comprehensive report for a pair of monomers.
    
    Parameters:
    - smiles1 (str): First monomer SMILES
    - smiles2 (str): Second monomer SMILES
    
    Returns:
    - str: Comprehensive structural and property report
    """
    return generate_structure_report(input.smiles1, input.smiles2)

class LLMStructureFineTuningAgent:
    """
    An LLM-powered agent that uses OpenAI fine-tuned model for structure fine-tuning tasks.
    """
    
    def __init__(self, model_name="ft:gpt-4o-mini-2024-07-18:personal::BKodpSOI"):
        """
        Initialize the agent with OpenAI model.
        
        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key (if None, uses default)
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model=model_name, 
            api_key=self.api_key, 
            temperature=0, 
            max_tokens=1000
        )
        
        # Create agent with all tools
        self.agent = create_react_agent(
            self.llm,
            tools=[
                analyze_properties_tool,
                validate_pair_tool,
                calculate_compatibility_tool,
                suggest_improvements_tool,
                remove_bond_tool,
                add_group_tool,
                modify_ring_tool,
                create_variants_tool,
                optimize_groups_tool,
                analyze_groups_tool,
                modify_stereochemistry_tool,
                make_compatible_tool,
                generate_report_tool
            ]
        )
    
    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input using the LLM agent.
        
        Args:
            user_input: User's natural language input
            
        Returns:
            Dictionary containing agent response
        """
        try:
            # Check for help request
            if "help" in user_input.lower():
                return {"help": self.get_help()}
            
            # Add system message to guide tool selection
            system_message = """You are a molecular structure analysis agent. 
            
            CRITICAL RULES:
            1. Call ONLY ONE tool per request - never multiple tools
            2. Extract parameters carefully from the input text
            3. For "add group" requests → use ONLY add_group_tool
            4. For "remove bond" requests → use ONLY remove_bond_tool  
            5. For "suggest improvements" requests → use ONLY suggest_improvements_tool
            6. For "analyze properties" requests → use ONLY analyze_properties_tool
            7. For "modify stereochemistry" requests → use ONLY modify_stereochemistry_tool
            8. For "analyze groups" requests → use ONLY analyze_groups_tool
            9. For "create variants" requests → use ONLY create_variants_tool
            10. For "make reaction" requests → use ONLY make_compatible_tool
            11. For "generate report" requests → use ONLY generate_report_tool
            
                         PARAMETER EXTRACTION:
             - Extract SMILES from "monomer1 = SMILES1" and "monomer2 = SMILES2"
             - Extract target monomer from "to monomer 2", "to monomer 1", "in monomer 1", "in monomer 2", or "monomer 1"/"monomer 2"
             
            TOOL SELECTION GUIDANCE:
            - Look for tool-specific keywords in the request
            - Choose the most specific tool that matches the request
            - Do NOT call multiple tools for a single request
            - If unsure, choose the tool that best matches the specific action requested"""
            
            # Process with LangChain agent
            response = self.agent.invoke({
                "messages": [
                    ("system", system_message),
                    ("human", user_input)
                ]
            })
            
            # Extract tool calls if any
            tool_calls = []
            for msg in response["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for call in msg.tool_calls:
                        tool_calls.append(call["name"])
            
            return {
                "response": response["messages"][-1].content,
                "tool_calls": tool_calls,
                "full_response": response
            }
            
        except Exception as e:
            return {"error": f"Error processing input: {str(e)}"}
    
    def get_help(self) -> str:
        """Get help information"""
        help_text = """
LLM-Powered Structure Fine-Tuning Agent - Available Tasks:

The agent can understand natural language requests and perform the following tasks:

1. **Analyze Properties**: "Analyze the properties of CCO"
2. **Validate Pair**: "Validate the pair CCO c1ccccc1"
3. **Calculate Compatibility**: "Calculate compatibility between CC(=O)O and CCO"
4. **Suggest Improvements**: "Suggest improvements for CC(=O)O and CCO"
5. **Remove Bonds**: "Remove CN bonds from monomer 1" or "Remove c1ccccc1 from monomer 2"
6. **Add Groups**: "Add hydroxyl group to monomer 1" or "Add [*]C(=O)O to monomer 2"
7. **Modify Rings**: "Aromatize the benzene ring in monomer 1" or "Saturate the epoxide in monomer 2"
8. **Create Variants**: "Create carbon chain variants of monomer 1"
9. **Optimize Groups**: "Optimize groups for solubility in monomer 1"
10. **Analyze Groups**: "Analyze functional groups in CCO"
11. **Modify Stereochemistry**: "Invert stereochemistry of monomer 1" or "Racemize monomer 2"
12. **Make Compatible**: "Make CC(=O)O and CCO reaction-compatible"
13. **Compare Structures**: "Compare original and modified structures"
14. **Generate Report**: "Generate a comprehensive report for CC(=O)O and CCO"

Examples:
- "Here are two monomers: monomer1 = CCNC1OC1Cc1ccccc1CCCCBr and monomer2 = CCCOOCC. remove CN group from monomer 1."
- "Analyze the properties of CCO"
- "Make CC(=O)O and CCO compatible for polymerization"
- "Add hydroxyl group to monomer 1"
- "Optimize groups for solubility in monomer 2"
        """
        return help_text

def main():
    """Main function to run the agent interactively"""
    agent = LLMStructureFineTuningAgent()
    
    print("LLM-Powered Structure Fine-Tuning Agent")
    print("=" * 50)
    print("Type 'help' for available commands")
    print("Type 'quit' to exit")
    print()
    
    while True:
        try:
            user_input = input("LLM Agent> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            result = agent.process_user_input(user_input)
            
            if "help" in result:
                print(result["help"])
            elif "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Response: {result['response']}")
                if result.get('tool_calls'):
                    print(f"Tools called: {result['tool_calls']}")
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 