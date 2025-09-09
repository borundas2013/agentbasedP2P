"""
Molecular Properties Module
Contains functions for analyzing molecular properties and validating SMILES pairs.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np
from rdkit.Chem import DataStructs

def analyze_molecular_properties(smiles: str) -> dict:
    """
    Analyze molecular properties for structure optimization guidance
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "Invalid SMILES"}
    
    return {
        "molecular_weight": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "aromatic_rings": Descriptors.NumAromaticRings(mol),
        "tpsa": Descriptors.TPSA(mol),
        "clogp": Descriptors.MolLogP(mol),
        "polar_surface_area": Descriptors.TPSA(mol)
    }

def validate_smiles_pair(smiles1: str, smiles2: str) -> dict:
    """
    Validate SMILES pair and provide structural insights
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return {"error": "Invalid SMILES provided"}
    
    return {
        "monomer1_valid": mol1 is not None,
        "monomer2_valid": mol2 is not None,
        "monomer1_properties": analyze_molecular_properties(smiles1),
        "monomer2_properties": analyze_molecular_properties(smiles2),
        "compatibility_score": calculate_compatibility_score(mol1, mol2)
    }

def calculate_compatibility_score(mol1, mol2) -> float:
    """
    Calculate compatibility score between two monomers
    """
    # Simple compatibility based on molecular weight difference and polarity
    mw1, mw2 = Descriptors.MolWt(mol1), Descriptors.MolWt(mol2)
    tpsa1, tpsa2 = Descriptors.TPSA(mol1), Descriptors.TPSA(mol2)
    
    mw_diff = abs(mw1 - mw2) / max(mw1, mw2)
    tpsa_diff = abs(tpsa1 - tpsa2) / max(tpsa1, tpsa2)
    
    return 1.0 - (mw_diff + tpsa_diff) / 2.0

def suggest_structural_improvements(smiles1: str, smiles2: str) -> dict:
    """
    Suggest structural improvements based on molecular properties
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return {"error": "Invalid SMILES provided"}
    
    suggestions = []
    
    # Analyze properties and suggest improvements
    mw1, mw2 = Descriptors.MolWt(mol1), Descriptors.MolWt(mol2)
    logp1, logp2 = Descriptors.MolLogP(mol1), Descriptors.MolLogP(mol2)
    tpsa1, tpsa2 = Descriptors.TPSA(mol1), Descriptors.TPSA(mol2)
    
    # Suggest improvements based on properties
    if mw1 > 500 or mw2 > 500:
        suggestions.append("Consider reducing molecular weight for better bioavailability")
    
    if logp1 > 5 or logp2 > 5:
        suggestions.append("High lipophilicity detected - consider adding polar groups")
    
    if tpsa1 < 90 or tpsa2 < 90:
        suggestions.append("Low polar surface area - consider adding polar functional groups")
    
    return {
        "suggestions": suggestions,
        "monomer1_analysis": analyze_molecular_properties(smiles1),
        "monomer2_analysis": analyze_molecular_properties(smiles2)
    } 