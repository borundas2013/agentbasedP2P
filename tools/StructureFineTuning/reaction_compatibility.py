"""
Reaction Compatibility Module
Contains functions for analyzing monomer reaction compatibility.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from functional_groups import analyze_functional_groups, has_functional_group, suggest_functional_group_addition, suggest_functional_group_removal

def make_monomers_reaction_compatible(smiles1: str, smiles2: str) -> dict:
    """
    Analyze two monomers and suggest modifications to make them reaction-compatible.
    
    Args:
        smiles1: SMILES string of first monomer
        smiles2: SMILES string of second monomer
        
    Returns:
        dict: Analysis and suggestions for making monomers compatible
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return {"error": "Invalid SMILES provided"}
        
        # Analyze functional groups in both monomers
        fg1 = analyze_functional_groups(mol1)
        fg2 = analyze_functional_groups(mol2)
        
        # Check for common polymerization reactions
        compatibility_analysis = {
            "monomer1": smiles1,
            "monomer2": smiles2,
            "functional_groups_1": fg1,
            "functional_groups_2": fg2,
            "compatibility_issues": [],
            "suggestions": [],
            "reaction_types": []
        }
        
        # Check for condensation polymerization (COOH + OH, NH2 + COOH, etc.)
        if has_functional_group(fg1, "carboxylic_acid") and has_functional_group(fg2, "hydroxyl"):
            compatibility_analysis["reaction_types"].append("Esterification")
            compatibility_analysis["suggestions"].append("Monomers are compatible for esterification polymerization")
        elif has_functional_group(fg1, "hydroxyl") and has_functional_group(fg2, "carboxylic_acid"):
            compatibility_analysis["reaction_types"].append("Esterification")
            compatibility_analysis["suggestions"].append("Monomers are compatible for esterification polymerization")
        elif has_functional_group(fg1, "carboxylic_acid") and has_functional_group(fg2, "amine"):
            compatibility_analysis["reaction_types"].append("Amidation")
            compatibility_analysis["suggestions"].append("Monomers are compatible for amidation polymerization")
        elif has_functional_group(fg1, "amine") and has_functional_group(fg2, "carboxylic_acid"):
            compatibility_analysis["reaction_types"].append("Amidation")
            compatibility_analysis["suggestions"].append("Monomers are compatible for amidation polymerization")
        
        # Check for addition polymerization (vinyl groups)
        elif has_functional_group(fg1, "vinyl") and has_functional_group(fg2, "vinyl"):
            compatibility_analysis["reaction_types"].append("Vinyl Addition")
            compatibility_analysis["suggestions"].append("Monomers are compatible for vinyl addition polymerization")
        
        # Check for ring-opening polymerization with multiplicity requirements
        elif fg1["epoxide"] >= 2 and fg2["imine"] >= 2:
            compatibility_analysis["reaction_types"].append("Epoxide-Imine Ring-opening")
            compatibility_analysis["suggestions"].append("Monomers are compatible for epoxide-imine ring-opening polymerization (at least 2 epoxide + 2 imine groups)")
        elif fg2["epoxide"] >= 2 and fg1["imine"] >= 2:
            compatibility_analysis["reaction_types"].append("Epoxide-Imine Ring-opening")
            compatibility_analysis["suggestions"].append("Monomers are compatible for epoxide-imine ring-opening polymerization (at least 2 epoxide + 2 imine groups)")
        elif has_functional_group(fg1, "epoxide") and has_functional_group(fg2, "hydroxyl"):
            compatibility_analysis["reaction_types"].append("Ring-opening")
            compatibility_analysis["suggestions"].append("Monomers are compatible for ring-opening polymerization")
        elif has_functional_group(fg1, "hydroxyl") and has_functional_group(fg2, "epoxide"):
            compatibility_analysis["reaction_types"].append("Ring-opening")
            compatibility_analysis["suggestions"].append("Monomers are compatible for ring-opening polymerization")
        
        # Check for vinyl + acrylate polymerization
        elif has_functional_group(fg1, "vinyl") and has_functional_group(fg2, "acrylate"):
            compatibility_analysis["reaction_types"].append("Vinyl-Acrylate Addition")
            compatibility_analysis["suggestions"].append("Monomers are compatible for vinyl-acrylate addition polymerization")
        elif has_functional_group(fg1, "acrylate") and has_functional_group(fg2, "vinyl"):
            compatibility_analysis["reaction_types"].append("Vinyl-Acrylate Addition")
            compatibility_analysis["suggestions"].append("Monomers are compatible for vinyl-acrylate addition polymerization")
        
        # Check for vinyl + hydroxyl polymerization (radical or catalytic)
        elif has_functional_group(fg1, "vinyl") and has_functional_group(fg2, "hydroxyl"):
            compatibility_analysis["reaction_types"].append("Vinyl-Hydroxyl Addition")
            compatibility_analysis["suggestions"].append("Monomers are compatible for vinyl-hydroxyl addition polymerization (requires catalyst)")
        elif has_functional_group(fg1, "hydroxyl") and has_functional_group(fg2, "vinyl"):
            compatibility_analysis["reaction_types"].append("Vinyl-Hydroxyl Addition")
            compatibility_analysis["suggestions"].append("Monomers are compatible for vinyl-hydroxyl addition polymerization (requires catalyst)")
        
        # If no direct compatibility, suggest modifications
        if not compatibility_analysis["reaction_types"]:
            compatibility_analysis["compatibility_issues"].append("No direct reaction compatibility found")
            
            # Suggest adding functional groups to monomer1
            suggestions_monomer1 = suggest_functional_group_addition(fg1, fg2)
            if suggestions_monomer1:
                compatibility_analysis["suggestions"].extend([f"Add to monomer 1: {s}" for s in suggestions_monomer1])
            
            # Suggest adding functional groups to monomer2
            suggestions_monomer2 = suggest_functional_group_addition(fg2, fg1)
            if suggestions_monomer2:
                compatibility_analysis["suggestions"].extend([f"Add to monomer 2: {s}" for s in suggestions_monomer2])
            
            # Suggest removing incompatible groups
            removal_suggestions = suggest_functional_group_removal(fg1, fg2)
            if removal_suggestions:
                compatibility_analysis["suggestions"].extend(removal_suggestions)
        
        return compatibility_analysis
        
    except Exception as e:
        return {"error": f"Error analyzing monomers: {str(e)}"} 