"""
Structure Modifications Module
Contains core functions for modifying molecular structures.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors

def remove_bond_by_smarts(smiles1: str, smiles2: str, bond_smarts: str, target_monomer: str = "1") -> str:
    """
    Remove bonds/groups from monomers using SMARTS patterns
    """
    # Select which molecule to modify
    target_smiles = smiles1 if target_monomer == "1" else smiles2
    other_smiles = smiles2 if target_monomer == "1" else smiles1
    
    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        return f"Invalid SMILES for monomer {target_monomer}"

    # Convert SMARTS pattern to molecule
    pattern = Chem.MolFromSmarts(bond_smarts)
    if pattern is None:
        return "Invalid SMARTS pattern"

    # Find where the pattern matches in the molecule
    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return f"Sorry, the group/bond '{bond_smarts}' is not found in monomer {target_monomer}"

    # Create editable molecule
    rw_mol = Chem.RWMol(mol)
    
    # Get all atoms in the pattern
    pattern_atoms = set()
    for match in matches:
        pattern_atoms.update(match)

    # Remove bonds first
    bonds_to_remove = []
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in pattern_atoms or end_idx in pattern_atoms:
            bonds_to_remove.append((begin_idx, end_idx))

    # Remove bonds in reverse order to maintain indices
    for begin_idx, end_idx in sorted(bonds_to_remove, reverse=True):
        rw_mol.RemoveBond(begin_idx, end_idx)

    # Now remove the atoms
    for idx in sorted(pattern_atoms, reverse=True):
        if idx < rw_mol.GetNumAtoms():
            rw_mol.RemoveAtom(idx)

    cleaned_mol = rw_mol.GetMol()

    # Get the remaining fragments
    frags = Chem.GetMolFrags(cleaned_mol, asMols=True, sanitizeFrags=False)

    smiles_list = []
    for frag in frags:
        try:
            Chem.SanitizeMol(frag)
            smiles_list.append(Chem.MolToSmiles(frag))
        except:
            continue

    if not smiles_list:
        return f"Sorry, after removing '{bond_smarts}', no valid fragments remain in monomer {target_monomer}"

    # Format output in the requested style
    modified_monomer = "".join(smiles_list)
    if target_monomer == "1":
        return f"Here is the revised output: \n -- monomer1 = {modified_monomer} \n -- monomer2 = {other_smiles}"
    else:
        return f"Here is the revised output: \n -- monomer1 = {other_smiles} \n -- monomer2 = {modified_monomer}"

def add_group_by_smarts(smiles1: str, smiles2: str, group_smarts: str, target_monomer: str = "1", attachment_atom_idx: int = 0) -> str:
    """
    Add functional groups to monomers using SMARTS patterns
    """
    # Select which molecule to modify
    target_smiles = smiles1 if target_monomer == "1" else smiles2
    other_smiles = smiles2 if target_monomer == "1" else smiles1

    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        return f"Invalid SMILES for monomer {target_monomer}"

    # Make editable molecule
    rw_mol = Chem.RWMol(mol)

    # Parse group with attachment point [*]
    group = Chem.MolFromSmarts(group_smarts)
    if group is None:
        return "Invalid group SMARTS"

    # Find attachment atom (the [*] atom) in group
    attachment_points = [atom.GetIdx() for atom in group.GetAtoms() if atom.GetSymbol() == '*']
    if not attachment_points:
        return "Group SMARTS must contain [*] as attachment point"
    group_attachment_idx = attachment_points[0]

    # Remove [*] atom and get neighbors
    rw_group = Chem.RWMol(group)
    group_neighbor = list(rw_group.GetAtomWithIdx(group_attachment_idx).GetNeighbors())[0]
    rw_group.RemoveAtom(group_attachment_idx)
    group = rw_group.GetMol()

    # Combine both molecules
    combo = Chem.CombineMols(rw_mol, group)
    rw_combo = Chem.RWMol(combo)

    # Calculate new atom index after combining
    offset = mol.GetNumAtoms()
    new_atom_idx = group_neighbor.GetIdx() + offset

    # Add bond between attachment site and new group
    rw_combo.AddBond(attachment_atom_idx, new_atom_idx, Chem.BondType.SINGLE)

    # Sanitize and return
    try:
        Chem.SanitizeMol(rw_combo)
        modified_smiles = Chem.MolToSmiles(rw_combo)
        if target_monomer == "1":
            return f"Here is the revised output: \n -- monomer1 = {modified_smiles} \n -- monomer2 = {other_smiles}"
        else:
            return f"Here is the revised output: \n -- monomer1 = {other_smiles} \n -- monomer2 = {modified_smiles}"
    except:
        return "Failed to sanitize combined molecule"

def modify_ring_system(smiles1: str, smiles2: str, ring_smarts: str, modification_type: str, target_monomer: str = "1") -> str:
    """
    Modify ring systems in monomers (aromatization, saturation, ring expansion/contraction)
    """
    target_smiles = smiles1 if target_monomer == "1" else smiles2
    other_smiles = smiles2 if target_monomer == "1" else smiles1
    
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        return f"Invalid SMILES for monomer {target_monomer}"
    
    pattern = Chem.MolFromSmarts(ring_smarts)
    if pattern is None:
        return "Invalid ring SMARTS pattern"
    
    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return f"Ring pattern '{ring_smarts}' not found in monomer {target_monomer}"
    
    rw_mol = Chem.RWMol(mol)
    
    # Check current aromaticity of the ring
    ring_atoms = set(matches[0])  # Get atoms in the first match
    current_aromatic_atoms = sum(1 for atom_idx in ring_atoms if rw_mol.GetAtomWithIdx(atom_idx).GetIsAromatic())
    total_ring_atoms = len(ring_atoms)
    
    if modification_type == "aromatize":
        # Check if ring is already aromatic
        if current_aromatic_atoms == total_ring_atoms:
            return f"Ring is already aromatic in monomer {target_monomer}. No changes needed."
        
        # Convert to aromatic ring
        for match in matches:
            for atom_idx in match:
                atom = rw_mol.GetAtomWithIdx(atom_idx)
                atom.SetIsAromatic(True)
                # Also set the bonds to aromatic
                for bond in rw_mol.GetAtomWithIdx(atom_idx).GetBonds():
                    if bond.GetOtherAtomIdx(atom_idx) in match:
                        bond.SetIsAromatic(True)
                        bond.SetBondType(Chem.BondType.AROMATIC)
    
    elif modification_type == "saturate":
        # Check if ring is already saturated
        if current_aromatic_atoms == 0:
            return f"Ring is already saturated in monomer {target_monomer}. No changes needed."
        
        # Convert to saturated ring
        for match in matches:
            for atom_idx in match:
                atom = rw_mol.GetAtomWithIdx(atom_idx)
                atom.SetIsAromatic(False)
                # Also set the bonds to single
                for bond in rw_mol.GetAtomWithIdx(atom_idx).GetBonds():
                    if bond.GetOtherAtomIdx(atom_idx) in match:
                        bond.SetIsAromatic(False)
                        bond.SetBondType(Chem.BondType.SINGLE)
    
    try:
        Chem.SanitizeMol(rw_mol)
        modified_smiles = Chem.MolToSmiles(rw_mol)
        
        # Check if any changes were actually made
        if modified_smiles == target_smiles:
            if modification_type == "aromatize":
                return f"Ring is already aromatic in monomer {target_monomer}. No changes made."
            elif modification_type == "saturate":
                return f"Ring is already saturated in monomer {target_monomer}. No changes made."
        
        if target_monomer == "1":
            return f"Here is the revised output: \n -- monomer1 = {modified_smiles} \n -- monomer2 = {other_smiles}"
        else:
            return f"Here is the revised output: \n -- monomer1 = {other_smiles} \n -- monomer2 = {modified_smiles}"
    except Exception as e:
        return f"Failed to modify ring system: {str(e)}"

def create_structural_variants(smiles1: str, smiles2: str, variant_type: str, target_monomer: str = "1") -> list:
    """
    Create structural variants of monomers
    """
    target_smiles = smiles1 if target_monomer == "1" else smiles2
    other_smiles = smiles2 if target_monomer == "1" else smiles1
    
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        return [{"error": f"Invalid SMILES for monomer {target_monomer}"}]
    
    variants = []
    
    if variant_type == "alkyl_variants":
        # Create variants with different alkyl chain lengths at different positions
        chain_lengths = [1, 2, 3]
        attachment_positions = []  # Collect all possible attachment points
        
        # Find all carbon atoms that can accept new bonds
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C' and atom.GetDegree() < 4:
                attachment_positions.append(atom.GetIdx())
        
        if not attachment_positions:
            return [{"error": "No suitable attachment points found"}]
        
        # Create variants by combining different chain lengths with different positions
        variant_count = 0
        for chain_length in chain_lengths:
            for pos_idx in attachment_positions[:3]:  # Limit to first 3 positions
                rw_mol = Chem.RWMol(mol)
                
                # Add alkyl chain
                prev_carbon = None
                for i in range(chain_length):
                    new_carbon = rw_mol.AddAtom(Chem.Atom('C'))
                    if i == 0:
                        # Attach to specific position
                        rw_mol.AddBond(pos_idx, new_carbon, Chem.BondType.SINGLE)
                        prev_carbon = new_carbon
                    else:
                        # Connect to previous carbon
                        rw_mol.AddBond(prev_carbon, new_carbon, Chem.BondType.SINGLE)
                        prev_carbon = new_carbon
                
                try:
                    Chem.SanitizeMol(rw_mol)
                    variant_smiles = Chem.MolToSmiles(rw_mol)
                    variants.append({
                        "variant_type": f"alkyl_chain_{chain_length}_pos_{pos_idx}",
                        "smiles": variant_smiles,
                        "target_monomer": target_monomer,
                        "chain_length": chain_length,
                        "attachment_position": pos_idx
                    })
                    variant_count += 1
                    if variant_count >= 4:  # Limit to 4 variants
                        break
                except:
                    continue
            if variant_count >= 4:
                break
    
    elif variant_type == "functional_group_variants":
        # Create variants with different functional groups
        functional_groups = ["[*]O", "[*]C(=O)O", "[*]N", "[*]S", "[*]F", "[*]Cl"]
        
        for fg in functional_groups:
            try:
                result = add_group_by_smarts(smiles1, smiles2, fg, target_monomer, 0)
                if "error" not in result.lower():
                    variants.append({
                        "variant_type": f"functional_group_{fg}",
                        "result": result,
                        "target_monomer": target_monomer
                    })
            except:
                continue
    
    return variants 