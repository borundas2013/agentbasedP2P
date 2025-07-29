"""
Stereochemistry Module
Contains functions for modifying stereochemistry and bond types.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors

def modify_stereochemistry(smiles1: str, smiles2: str, stereochemistry_type: str, target_monomer: str = "1") -> str:
    """
    Modify stereochemistry of chiral centers
    """
    target_smiles = smiles1 if target_monomer == "1" else smiles2
    other_smiles = smiles2 if target_monomer == "1" else smiles1
    
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        return f"Invalid SMILES for monomer {target_monomer}"
    
    # Find chiral centers
    chiral_centers = Chem.FindMolChiralCenters(mol)
    
    if not chiral_centers:
        # Provide informative message for non-chiral molecules
        return f"No chiral centers found in monomer {target_monomer}. This molecule is achiral (no stereochemistry to modify). Common in polymers and simple organic compounds."
    
    rw_mol = Chem.RWMol(mol)
    
    if stereochemistry_type == "invert":
        # Invert all chiral centers
        for atom_idx, chirality in chiral_centers:
            atom = rw_mol.GetAtomWithIdx(atom_idx)
            if chirality == 'R':
                atom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CCW)  # R -> S
            elif chirality == 'S':
                atom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CW)   # S -> R
    
    elif stereochemistry_type == "racemize":
        # Remove stereochemistry information
        for atom_idx, chirality in chiral_centers:
            atom = rw_mol.GetAtomWithIdx(atom_idx)
            atom.SetChiralTag(Chem.CHI_UNSPECIFIED)
    
    try:
        Chem.SanitizeMol(rw_mol)
        modified_smiles = Chem.MolToSmiles(rw_mol)
        
        # Provide detailed information about the changes
        if stereochemistry_type == "invert":
            change_info = f"Inverted {len(chiral_centers)} chiral center(s): "
            for atom_idx, chirality in chiral_centers:
                change_info += f"{chirality}->{('S' if chirality == 'R' else 'R')} "
        elif stereochemistry_type == "racemize":
            change_info = f"Removed stereochemistry from {len(chiral_centers)} chiral center(s)"
        
        if target_monomer == "1":
            return f"Here is the revised output: \n -- monomer1 = {modified_smiles} \n -- monomer2 = {other_smiles} \n -- Changes: {change_info}"
        else:
            return f"Here is the revised output: \n -- monomer1 = {other_smiles} \n -- monomer2 = {modified_smiles} \n -- Changes: {change_info}"
    except:
        return "Failed to modify stereochemistry"

def modify_bond_types(smiles1: str, smiles2: str, bond_smarts: str, new_bond_type: str, target_monomer: str = "1") -> str:
    """
    Modify bond types (single, double, triple, aromatic)
    """
    target_smiles = smiles1 if target_monomer == "1" else smiles2
    other_smiles = smiles2 if target_monomer == "1" else smiles1
    
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        return f"Invalid SMILES for monomer {target_monomer}"
    
    pattern = Chem.MolFromSmarts(bond_smarts)
    if pattern is None:
        return "Invalid bond SMARTS pattern"
    
    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return f"Bond pattern '{bond_smarts}' not found in monomer {target_monomer}"
    
    rw_mol = Chem.RWMol(mol)
    
    # Map bond type strings to RDKit bond types
    bond_type_map = {
        "single": Chem.BondType.SINGLE,
        "double": Chem.BondType.DOUBLE,
        "triple": Chem.BondType.TRIPLE,
        "aromatic": Chem.BondType.AROMATIC
    }
    
    new_bond_type_rdkit = bond_type_map.get(new_bond_type.lower())
    if new_bond_type_rdkit is None:
        return f"Invalid bond type: {new_bond_type}"
    
    # Helper function to get bond order
    def get_bond_order(bond_type):
        if bond_type == Chem.BondType.SINGLE:
            return 1
        elif bond_type == Chem.BondType.DOUBLE:
            return 2
        elif bond_type == Chem.BondType.TRIPLE:
            return 3
        elif bond_type == Chem.BondType.AROMATIC:
            return 1.5
        else:
            return 1
    
    # Check if modification is possible without violating valence rules
    modifications_made = 0
    for match in matches:
        for i in range(len(match) - 1):
            for j in range(i + 1, len(match)):
                bond = rw_mol.GetBondBetweenAtoms(match[i], match[j])
                if bond is not None:
                    # Check valence before modification
                    atom1 = rw_mol.GetAtomWithIdx(match[i])
                    atom2 = rw_mol.GetAtomWithIdx(match[j])
                    
                    # Calculate current valence
                    current_valence1 = atom1.GetTotalValence()
                    current_valence2 = atom2.GetTotalValence()
                    
                    # Calculate new valence if we change bond type
                    current_bond_order = get_bond_order(bond.GetBondType())
                    new_bond_order = get_bond_order(new_bond_type_rdkit)
                    valence_change = new_bond_order - current_bond_order
                    
                    new_valence1 = current_valence1 + valence_change
                    new_valence2 = current_valence2 + valence_change
                    
                    # Check if new valence is valid (carbon max 4, other atoms have their limits)
                    max_valence1 = 4 if atom1.GetSymbol() == 'C' else 6  # Simple rule
                    max_valence2 = 4 if atom2.GetSymbol() == 'C' else 6  # Simple rule
                    
                    if new_valence1 <= max_valence1 and new_valence2 <= max_valence2:
                        rw_mol.RemoveBond(match[i], match[j])
                        rw_mol.AddBond(match[i], match[j], new_bond_type_rdkit)
                        modifications_made += 1
    
    if modifications_made == 0:
        return f"No valid bond modifications possible. The requested change would violate valence rules for carbon atoms (max 4 bonds)."
    
    try:
        Chem.SanitizeMol(rw_mol)
        modified_smiles = Chem.MolToSmiles(rw_mol)
        if target_monomer == "1":
            return f"Here is the revised output: \n -- monomer1 = {modified_smiles} \n -- monomer2 = {other_smiles} \n -- Changes: Modified {modifications_made} bond(s) to {new_bond_type}"
        else:
            return f"Here is the revised output: \n -- monomer1 = {other_smiles} \n -- monomer2 = {modified_smiles} \n -- Changes: Modified {modifications_made} bond(s) to {new_bond_type}"
    except Exception as e:
        return f"Failed to modify bond types: {str(e)}" 