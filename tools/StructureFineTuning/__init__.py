"""
Structure Fine-Tuning Package
A modular package for molecular structure fine-tuning and analysis.
"""

# Import all functions from the main module
from .main import *

# Import the convenience class
from .main import StructureFineTuner

# Package version
__version__ = "1.0.0"

# Package description
__description__ = "A modular package for molecular structure fine-tuning and analysis"

# Author information
__author__ = "Structure Fine-Tuning Team"

# Export the main class and all functions
__all__ = [
    'StructureFineTuner',
    # Molecular properties
    'analyze_molecular_properties',
    'validate_smiles_pair',
    'calculate_compatibility_score',
    'suggest_structural_improvements',
    
    # Structure modifications
    'remove_bond_by_smarts',
    'add_group_by_smarts',
    'modify_ring_system',
    'create_structural_variants',
    
    # Functional groups
    'optimize_functional_groups',
    'analyze_functional_groups',
    'has_functional_group',
    'suggest_functional_group_addition',
    'suggest_functional_group_removal',
    
    # Stereochemistry
    'modify_stereochemistry',
    'modify_bond_types',
    
    # Reaction compatibility
    'make_monomers_reaction_compatible',
    
    # Structure analysis
    'compare_structures',
    'generate_structure_report'
] 