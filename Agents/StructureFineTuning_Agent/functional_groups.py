"""
Functional Groups Module
Contains functions for analyzing and optimizing functional groups.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors

def optimize_functional_groups(smiles1: str, smiles2: str, optimization_type: str, target_monomer: str = "1") -> str:
    """
    Optimize functional groups for better properties
    """
    target_smiles = smiles1 if target_monomer == "1" else smiles2
    other_smiles = smiles2 if target_monomer == "1" else smiles1
    
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        return f"Invalid SMILES for monomer {target_monomer}"
    
    rw_mol = Chem.RWMol(mol)
    modifications_made = 0
    
    if optimization_type == "improve_solubility":
        # Convert hydrophobic groups to more hydrophilic ones
        # Look for terminal methyl groups (CH3) and convert to alcohols
        atoms_to_modify = []
        for i, atom in enumerate(rw_mol.GetAtoms()):
            if (atom.GetSymbol() == 'C' and 
                atom.GetDegree() == 1 and 
                atom.GetTotalNumHs() >= 2):  # Terminal carbon with hydrogens
                atoms_to_modify.append(i)
        
        # Apply modifications
        for atom_idx in atoms_to_modify[:2]:  # Limit to 2 modifications
            new_atom_idx = rw_mol.AddAtom(Chem.Atom('O'))
            rw_mol.AddBond(atom_idx, new_atom_idx, Chem.BondType.SINGLE)
            modifications_made += 1
        
        # Also look for secondary carbons (CH2) to convert to alcohols
        atoms_to_modify = []
        for i, atom in enumerate(rw_mol.GetAtoms()):
            if (atom.GetSymbol() == 'C' and 
                atom.GetDegree() == 2 and 
                atom.GetTotalNumHs() >= 1):
                atoms_to_modify.append(i)
        
        # Apply modifications
        for atom_idx in atoms_to_modify[:2]:  # Limit to 2 modifications
            new_atom_idx = rw_mol.AddAtom(Chem.Atom('O'))
            rw_mol.AddBond(atom_idx, new_atom_idx, Chem.BondType.SINGLE)
            modifications_made += 1
    
    elif optimization_type == "improve_stability":
        # Convert labile groups to more stable ones
        # Look for ester-like structures and convert to amides
        bonds_to_modify = []
        for bond in rw_mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                begin_atom = rw_mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
                end_atom = rw_mol.GetAtomWithIdx(bond.GetEndAtomIdx())
                if (begin_atom.GetSymbol() == 'C' and 
                    end_atom.GetSymbol() == 'O'):
                    bonds_to_modify.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        
        # Apply modifications
        for begin_idx, end_idx in bonds_to_modify[:1]:  # Limit to 1 modification
            nitrogen_atom_idx = rw_mol.AddAtom(Chem.Atom('N'))
            rw_mol.RemoveBond(begin_idx, end_idx)
            rw_mol.AddBond(begin_idx, nitrogen_atom_idx, Chem.BondType.SINGLE)
            modifications_made += 1
    
    elif optimization_type == "add_polar_groups":
        # Add polar groups to improve water solubility
        atoms_to_modify = []
        for i, atom in enumerate(rw_mol.GetAtoms()):
            if (atom.GetSymbol() == 'C' and 
                atom.GetDegree() == 1):
                atoms_to_modify.append(i)
        
        # Apply modifications
        for atom_idx in atoms_to_modify[:2]:  # Limit to 2 modifications
            # Add carboxylic acid group
            new_carbon_idx = rw_mol.AddAtom(Chem.Atom('C'))
            new_oxygen1_idx = rw_mol.AddAtom(Chem.Atom('O'))
            new_oxygen2_idx = rw_mol.AddAtom(Chem.Atom('O'))
            
            rw_mol.AddBond(atom_idx, new_carbon_idx, Chem.BondType.SINGLE)
            rw_mol.AddBond(new_carbon_idx, new_oxygen1_idx, Chem.BondType.DOUBLE)
            rw_mol.AddBond(new_carbon_idx, new_oxygen2_idx, Chem.BondType.SINGLE)
            modifications_made += 1
    
    try:
        Chem.SanitizeMol(rw_mol)
        modified_smiles = Chem.MolToSmiles(rw_mol)
        
        # Check if any changes were actually made
        if modified_smiles == target_smiles:
            return f"No suitable functional groups found for {optimization_type} optimization in monomer {target_monomer}. No changes made."
        
        if target_monomer == "1":
            return f"Here is the revised output: \n -- monomer1 = {modified_smiles} \n -- monomer2 = {other_smiles}"
        else:
            return f"Here is the revised output: \n -- monomer1 = {other_smiles} \n -- monomer2 = {modified_smiles}"
    except Exception as e:
        return f"Failed to optimize functional groups: {str(e)}"

def analyze_functional_groups(mol):
    """Analyze functional groups in a molecule."""
    fg = {
        "hydroxyl": 0,
        "carboxylic_acid": 0,
        "amine": 0,
        "vinyl": 0,
        "epoxide": 0,
        "ester": 0,
        "amide": 0,
        "halide": 0,
        "nitrile": 0,
        "aldehyde": 0,
        "ketone": 0,
        "imine": 0,
        "acrylate": 0
    }
    
    # Count functional groups using SMARTS patterns
    patterns = {
        "hydroxyl": "[OH]",
        "carboxylic_acid": "[C](=[O])[OH]",
        "amine": "[NH2]",
        "vinyl": "C=C",
        "epoxide": "C1OC1",
        "ester": "[C](=[O])[O][C]",
        "amide": "[C](=[O])[NH]",
        "halide": "[F,Cl,Br,I]",
        "nitrile": "C#N",
        "aldehyde": "[CH](=O)",
        "ketone": "[C](=[O])[C]",
        "imine": "[C]=[N]",
        "acrylate": "[C](=[O])[O][C]=[C]"
    }
    
    for group, pattern in patterns.items():
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
        fg[group] = len(matches)
    
    return fg

def has_functional_group(fg_dict, group_name):
    """Check if molecule has a specific functional group."""
    return fg_dict.get(group_name, 0) > 0

def suggest_functional_group_addition(fg1, fg2):
    """Suggest functional groups to add for compatibility."""
    suggestions = []
    
    # If monomer1 has carboxylic acid, suggest adding hydroxyl to monomer2
    if has_functional_group(fg1, "carboxylic_acid") and not has_functional_group(fg2, "hydroxyl"):
        suggestions.append("Add hydroxyl group (-OH)")
    
    # If monomer1 has hydroxyl, suggest adding carboxylic acid to monomer2
    if has_functional_group(fg1, "hydroxyl") and not has_functional_group(fg2, "carboxylic_acid"):
        suggestions.append("Add carboxylic acid group (-COOH)")
    
    # If monomer1 has carboxylic acid, suggest adding amine to monomer2
    if has_functional_group(fg1, "carboxylic_acid") and not has_functional_group(fg2, "amine"):
        suggestions.append("Add amine group (-NH2)")
    
    # If monomer1 has amine, suggest adding carboxylic acid to monomer2
    if has_functional_group(fg1, "amine") and not has_functional_group(fg2, "carboxylic_acid"):
        suggestions.append("Add carboxylic acid group (-COOH)")
    
    # For vinyl polymerization
    if has_functional_group(fg1, "vinyl") and not has_functional_group(fg2, "vinyl"):
        suggestions.append("Add vinyl group (C=C)")
    
    # For epoxide-imine polymerization (need at least 2 of each)
    if fg1["epoxide"] >= 2 and fg2["imine"] < 2:
        suggestions.append("Add at least 2 imine groups (-C=N-) for epoxide-imine polymerization")
    elif fg1["imine"] >= 2 and fg2["epoxide"] < 2:
        suggestions.append("Add at least 2 epoxide groups (C1OC1) for epoxide-imine polymerization")
    
    # For vinyl-acrylate polymerization
    if has_functional_group(fg1, "vinyl") and not has_functional_group(fg2, "acrylate"):
        suggestions.append("Add acrylate group (-C(=O)O-C=C) for vinyl-acrylate polymerization")
    elif has_functional_group(fg1, "acrylate") and not has_functional_group(fg2, "vinyl"):
        suggestions.append("Add vinyl group (C=C) for vinyl-acrylate polymerization")
    
    # For vinyl-hydroxyl polymerization
    if has_functional_group(fg1, "vinyl") and not has_functional_group(fg2, "hydroxyl"):
        suggestions.append("Add hydroxyl group (-OH) for vinyl-hydroxyl polymerization (requires catalyst)")
    elif has_functional_group(fg1, "hydroxyl") and not has_functional_group(fg2, "vinyl"):
        suggestions.append("Add vinyl group (C=C) for vinyl-hydroxyl polymerization (requires catalyst)")
    
    return suggestions

def suggest_functional_group_removal(fg1, fg2):
    """Suggest functional groups to remove for compatibility."""
    suggestions = []
    
    # Remove conflicting groups
    if has_functional_group(fg1, "carboxylic_acid") and has_functional_group(fg2, "carboxylic_acid"):
        suggestions.append("Remove carboxylic acid from one monomer (both have -COOH)")
    
    if has_functional_group(fg1, "hydroxyl") and has_functional_group(fg2, "hydroxyl"):
        suggestions.append("Remove hydroxyl from one monomer (both have -OH)")
    
    if has_functional_group(fg1, "amine") and has_functional_group(fg2, "amine"):
        suggestions.append("Remove amine from one monomer (both have -NH2)")
    
    return suggestions 