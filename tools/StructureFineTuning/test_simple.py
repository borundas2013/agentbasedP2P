#!/usr/bin/env python3
"""
Simple test script for Structure Fine-Tuning Package
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from agentbasedP2P.tools.StructureFineTuning.main import StructureFineTuner

def test_basic_functionality():
    """Test basic functionality of the StructureFineTuner"""
    print("Testing Structure Fine-Tuning Package...")
    
    # Initialize the fine-tuner
    tuner = StructureFineTuner()
    
    # Test SMILES
    test_smiles = {
        "simple": "CCO",  # Ethanol
        "aromatic": "c1ccccc1",  # Benzene
        "complex": "CC(C)(C)c1ccc(O)cc1",  # 4-tert-butylphenol
        "chiral": "C[C@H](N)C(=O)O",  # L-alanine
        "vinyl": "C=CC",  # Propene
        "epoxide": "C1COC1",  # Ethylene oxide
        "amine": "CCN",  # Diethylamine
        "carboxylic": "CC(=O)O",  # Acetic acid
        "hydroxyl": "CCO",  # Ethanol
        "imine": "CC(=N)C",  # Acetone imine
        "acrylate": "C=CC(=O)OC",  # Methyl acrylate
    }
    
    print("\n1. Testing Molecular Properties:")
    result = tuner.analyze_properties(test_smiles["simple"])
    print(f"  Ethanol properties: {result}")
    
    print("\n2. Testing SMILES Validation:")
    result = tuner.validate_pair(test_smiles["simple"], test_smiles["aromatic"])
    print(f"  Validation result: {result}")
    
    print("\n3. Testing Reaction Compatibility:")
    result = tuner.make_compatible(test_smiles["carboxylic"], test_smiles["hydroxyl"])
    print(f"  Esterification compatibility: {result}")
    
    print("\n4. Testing Structure Report:")
    result = tuner.generate_report(test_smiles["complex"], test_smiles["vinyl"])
    print(f"  Structure report: {result}")
    
    print("\n5. Testing Stereochemistry:")
    result = tuner.modify_stereo(test_smiles["chiral"], test_smiles["simple"], "invert", "1")
    print(f"  Stereochemistry modification: {result}")
    
    print("\n6. Testing Functional Group Analysis:")
    from rdkit import Chem
    mol = Chem.MolFromSmiles(test_smiles["hydroxyl"])
    if mol:
        result = tuner.analyze_groups(mol)
        print(f"  Functional groups: {result}")
    
    print("\n7. Testing Structure Comparison:")
    result = tuner.compare_structures(
        test_smiles["simple"], test_smiles["aromatic"],
        test_smiles["complex"], test_smiles["vinyl"]
    )
    print(f"  Structure comparison: {result}")
    
    print("\nAll basic tests completed successfully!")
    return True

if __name__ == "__main__":
    test_basic_functionality() 