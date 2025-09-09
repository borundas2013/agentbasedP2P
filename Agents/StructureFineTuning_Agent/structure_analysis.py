"""
Structure Analysis Module
Contains functions for comparing structures and generating reports.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np
from rdkit.Chem import DataStructs
from StructureFineTuning_Agent.molecular_properties import analyze_molecular_properties, calculate_compatibility_score, suggest_structural_improvements

def compare_structures(smiles1_original: str, smiles2_original: str, smiles1_modified: str, smiles2_modified: str) -> dict:
    """
    Compare original and modified structures
    """
    mol1_orig = Chem.MolFromSmiles(smiles1_original)
    mol2_orig = Chem.MolFromSmiles(smiles2_original)
    mol1_mod = Chem.MolFromSmiles(smiles1_modified)
    mol2_mod = Chem.MolFromSmiles(smiles2_modified)
    
    if any(mol is None for mol in [mol1_orig, mol2_orig, mol1_mod, mol2_mod]):
        return {"error": "Invalid SMILES provided"}
    
    # Calculate property changes
    def get_properties(mol):
        return {
            "molecular_weight": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "aromatic_rings": Descriptors.NumAromaticRings(mol)
        }
    
    orig_props1 = get_properties(mol1_orig)
    orig_props2 = get_properties(mol2_orig)
    mod_props1 = get_properties(mol1_mod)
    mod_props2 = get_properties(mol2_mod)
    
    # Calculate changes
    changes1 = {key: mod_props1[key] - orig_props1[key] for key in orig_props1}
    changes2 = {key: mod_props2[key] - orig_props2[key] for key in orig_props2}
    
    return {
        "monomer1_changes": changes1,
        "monomer2_changes": changes2,
        "original_properties": {
            "monomer1": orig_props1,
            "monomer2": orig_props2
        },
        "modified_properties": {
            "monomer1": mod_props1,
            "monomer2": mod_props2
        }
    }

def generate_structure_report(smiles1: str, smiles2: str) -> dict:
    """
    Generate comprehensive structure analysis report
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return {"error": "Invalid SMILES provided"}
    
    # Calculate various descriptors
    def get_comprehensive_properties(mol):
        return {
            "molecular_weight": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "aromatic_rings": Descriptors.NumAromaticRings(mol),
            "hbd": Descriptors.NumHDonors(mol),
            "hba": Descriptors.NumHAcceptors(mol),
            "ring_count": Descriptors.RingCount(mol),
            "fraction_csp3": Descriptors.FractionCSP3(mol),
            "heavy_atom_count": Descriptors.HeavyAtomCount(mol)
        }
    
    props1 = get_comprehensive_properties(mol1)
    props2 = get_comprehensive_properties(mol2)
    
    # Calculate similarity
    fp1 = AllChem.GetMorganFingerprint(mol1, 2)
    fp2 = AllChem.GetMorganFingerprint(mol2, 2)
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    
    return {
        "monomer1_analysis": props1,
        "monomer2_analysis": props2,
        "structural_similarity": similarity,
        "compatibility_score": calculate_compatibility_score(mol1, mol2),
        "suggestions": suggest_structural_improvements(smiles1, smiles2)["suggestions"]
    } 