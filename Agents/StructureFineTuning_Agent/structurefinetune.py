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


def remove_bond_by_smarts(smiles1: str, smiles2: str, bond_smarts: str, target_monomer: str = "1") -> str:
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
    
    if "alkyl" in variant_type:
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

def modify_stereochemistry(smiles1: str, smiles2: str, stereochemistry_type: str, target_monomer: str = "1") -> str:
    """
    Modify stereochemistry of chiral centers
    """
    print(smiles1)
    print(smiles2)
    print(stereochemistry_type)
    print(target_monomer)
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
    
    if "invert" in stereochemistry_type:
        # Invert all chiral centers
        for atom_idx, chirality in chiral_centers:
            atom = rw_mol.GetAtomWithIdx(atom_idx)
            if chirality == 'R':
                atom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CCW)  # R -> S
            elif chirality == 'S':
                atom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CW)   # S -> R
    
    elif "racemize" in stereochemistry_type:
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
        
        # Check for halide-amine polymerization
        elif has_functional_group(fg1, "halide") and has_functional_group(fg2, "amine"):
            compatibility_analysis["reaction_types"].append("Halide-Amine Substitution")
            compatibility_analysis["suggestions"].append("Monomers are compatible for halide-amine substitution polymerization")
        elif has_functional_group(fg1, "amine") and has_functional_group(fg2, "halide"):
            compatibility_analysis["reaction_types"].append("Halide-Amine Substitution")
            compatibility_analysis["suggestions"].append("Monomers are compatible for halide-amine substitution polymerization")
        
        # Check for cyclic ether-halide polymerization
        elif has_functional_group(fg1, "cyclic_ether") and has_functional_group(fg2, "halide"):
            compatibility_analysis["reaction_types"].append("Cyclic Ether-Halide Ring-opening")
            compatibility_analysis["suggestions"].append("Monomers are compatible for cyclic ether-halide ring-opening polymerization")
        elif has_functional_group(fg1, "halide") and has_functional_group(fg2, "cyclic_ether"):
            compatibility_analysis["reaction_types"].append("Cyclic Ether-Halide Ring-opening")
            compatibility_analysis["suggestions"].append("Monomers are compatible for cyclic ether-halide ring-opening polymerization")
        
        # Check for carboxylic acid-halide polymerization
        elif has_functional_group(fg1, "carboxylic_acid") and has_functional_group(fg2, "halide"):
            compatibility_analysis["reaction_types"].append("Carboxylic Acid-Halide Substitution")
            compatibility_analysis["suggestions"].append("Monomers are compatible for carboxylic acid-halide substitution polymerization")
        elif has_functional_group(fg1, "halide") and has_functional_group(fg2, "carboxylic_acid"):
            compatibility_analysis["reaction_types"].append("Carboxylic Acid-Halide Substitution")
            compatibility_analysis["suggestions"].append("Monomers are compatible for carboxylic acid-halide substitution polymerization")
        
        # Check for epoxide-halide polymerization
        elif has_functional_group(fg1, "epoxide") and has_functional_group(fg2, "halide"):
            compatibility_analysis["reaction_types"].append("Epoxide-Halide Ring-opening")
            compatibility_analysis["suggestions"].append("Monomers are compatible for epoxide-halide ring-opening polymerization")
        elif has_functional_group(fg1, "halide") and has_functional_group(fg2, "epoxide"):
            compatibility_analysis["reaction_types"].append("Epoxide-Halide Ring-opening")
            compatibility_analysis["suggestions"].append("Monomers are compatible for epoxide-halide ring-opening polymerization")
        
        # If no direct compatibility, suggest modifications
        if not compatibility_analysis["reaction_types"]:
            compatibility_analysis["compatibility_issues"].append("No direct reaction compatibility found")
            
            # Get suggestions for adding groups to monomer2 (when monomer1 has specific groups)
            suggestions_for_monomer2 = suggest_functional_group_addition(fg1, fg2)
            if suggestions_for_monomer2:
                compatibility_analysis["suggestions"].extend([f"Add to monomer 2: {s}" for s in suggestions_for_monomer2])
            
            # Get suggestions for adding groups to monomer1 (when monomer2 has specific groups)
            suggestions_for_monomer1 = suggest_functional_group_addition(fg2, fg1)
            if suggestions_for_monomer1:
                compatibility_analysis["suggestions"].extend([f"Add to monomer 1: {s}" for s in suggestions_for_monomer1])
            
            # Suggest removing incompatible groups
            removal_suggestions = suggest_functional_group_removal(fg1, fg2)
            if removal_suggestions:
                compatibility_analysis["suggestions"].extend(removal_suggestions)
        
        return compatibility_analysis
        
    except Exception as e:
        return {"error": f"Error analyzing monomers: {str(e)}"}

def analyze_functional_groups(mol):
    """Analyze functional groups in a molecule."""
    fg = {
        "hydroxyl": 0,
        "carboxylic_acid": 0,
        "amine": 0,
        "vinyl": 0,
        "epoxide": 0,
        "cyclic_ether": 0,
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
        "epoxide": "C1OC1",  # 3-membered ring with oxygen (epoxide)
        "cyclic_ether": "C1OC1",  # 3-membered ring with oxygen (cyclic ether)
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
    
    # More specific detection for epoxides vs cyclic ethers
    # Epoxide: C1OC1 (3-membered ring with oxygen, no substituents on ring carbons)
    # Cyclic ether: C1OC1 (3-membered ring with oxygen, may have substituents)
    
    # Check for true epoxides (no substituents on ring carbons)
    epoxide_pattern = Chem.MolFromSmarts("C1OC1")
    epoxide_matches = mol.GetSubstructMatches(epoxide_pattern)
    
    true_epoxides = 0
    cyclic_ethers = 0
    
    for match in epoxide_matches:
        # Check if ring carbons have substituents
        has_substituents = False
        for atom_idx in match:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetSymbol() == 'C':
                # Count bonds to non-ring atoms
                ring_atoms = set(match)
                substituent_bonds = 0
                for bond in atom.GetBonds():
                    other_idx = bond.GetOtherAtomIdx(atom_idx)
                    if other_idx not in ring_atoms:
                        substituent_bonds += 1
                
                if substituent_bonds > 0:
                    has_substituents = True
                    break
        
        if has_substituents:
            cyclic_ethers += 1
        else:
            true_epoxides += 1
    
    fg["epoxide"] = true_epoxides
    fg["cyclic_ether"] = cyclic_ethers
    
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
    
    # For cyclic ether polymerization
    if has_functional_group(fg1, "cyclic_ether") and not has_functional_group(fg2, "cyclic_ether"):
        suggestions.append("Add cyclic ether group for ring-opening polymerization")
    
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
    
    # For cyclic ether polymerization (ring-opening polymerization)
    if has_functional_group(fg1, "cyclic_ether") and not has_functional_group(fg2, "hydroxyl"):
        suggestions.append("Add hydroxyl group (-OH) for ring-opening polymerization of cyclic ether")
    elif has_functional_group(fg1, "hydroxyl") and not has_functional_group(fg2, "cyclic_ether"):
        suggestions.append("Add cyclic ether group (C1OC1) for ring-opening polymerization")
    
    # For halide-amine polymerization
    if has_functional_group(fg1, "halide") and not has_functional_group(fg2, "amine"):
        suggestions.append("Add amine group (-NH2) for halide-amine polymerization")
    elif has_functional_group(fg1, "amine") and not has_functional_group(fg2, "halide"):
        suggestions.append("Add halide group (-Br, -Cl, -I) for halide-amine polymerization")
    
    # For cyclic ether-halide polymerization
    if has_functional_group(fg1, "cyclic_ether") and not has_functional_group(fg2, "halide"):
        suggestions.append("Add halide group (-Br, -Cl, -I) for cyclic ether-halide polymerization")
    elif has_functional_group(fg1, "halide") and not has_functional_group(fg2, "cyclic_ether"):
        suggestions.append("Add cyclic ether group (C1OC1) for cyclic ether-halide polymerization")
    
    # For carboxylic acid-halide polymerization
    if has_functional_group(fg1, "carboxylic_acid") and not has_functional_group(fg2, "halide"):
        suggestions.append("Add halide group (-Br, -Cl, -I) for carboxylic acid-halide polymerization")
    elif has_functional_group(fg1, "halide") and not has_functional_group(fg2, "carboxylic_acid"):
        suggestions.append("Add carboxylic acid group (-COOH) for carboxylic acid-halide polymerization")
    
    # For epoxide-halide polymerization
    if has_functional_group(fg1, "epoxide") and not has_functional_group(fg2, "halide"):
        suggestions.append("Add halide group (-Br, -Cl, -I) for epoxide-halide polymerization")
    elif has_functional_group(fg1, "halide") and not has_functional_group(fg2, "epoxide"):
        suggestions.append("Add epoxide group (C1OC1) for epoxide-halide polymerization")
    
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
    
    # For cyclic ethers (both have cyclic ethers but no complementary groups)
    if has_functional_group(fg1, "cyclic_ether") and has_functional_group(fg2, "cyclic_ether"):
        suggestions.append("Remove cyclic ether from one monomer or add complementary groups for polymerization")
    
    # For halides (both have halides but no complementary groups)
    if has_functional_group(fg1, "halide") and has_functional_group(fg2, "halide"):
        suggestions.append("Remove halide from one monomer or add complementary groups for polymerization")
    
    return suggestions


if __name__ == "__main__":
    # Test molecules
    smiles1 = "CCNC1OC1Cc1ccccc1CCCCBr"
    smiles2 = "CCC2OC2COOCC"
    
    print("=== Structure Fine-Tuning Agent Demo ===\n")
    
    # # 1. Basic functionality
    print("1. Basic Structure Modifications:")
    print("Remove CN group from monomer 1:")
    print(remove_bond_by_smarts(smiles1, smiles2, "CN", "1"))
    print()
    
    print("Add carboxylic acid group to monomer 2:")
    print(add_group_by_smarts(smiles1, smiles2, "[*]C(=O)O", "2", 0))
    print()
    
    # # 2. Structure analysis
    print("2. Structure Analysis:")
    analysis = validate_smiles_pair(smiles1, smiles2)
    print(f"Compatibility Score: {analysis['compatibility_score']:.3f}")
    print(f"Monomer 1 MW: {analysis['monomer1_properties']['molecular_weight']:.2f}")
    print(f"Monomer 2 MW: {analysis['monomer2_properties']['molecular_weight']:.2f}")
    print()
    
    # # 3. Ring modifications
    print("3. Ring System Modifications:")
    print("Aromatize ring in monomer 1 (already aromatic):")
    print(modify_ring_system(smiles1, smiles2, "c1ccccc1", "aromatize", "1"))
    print()
    
    # # Test with a non-aromatic ring
    print("Aromatize cyclohexane ring (non-aromatic):")
    cyclohexane_smiles = "CC1CCCCC1"
    print(modify_ring_system(cyclohexane_smiles, smiles2, "C1CCCCC1", "aromatize", "1"))
    print()
    
    print("Saturate benzene ring (aromatic to saturated):")
    print(modify_ring_system(smiles1, smiles2, "c1ccccc1", "saturate", "1"))
    print()
    
    # 4. Functional group optimization
    print("4. Functional Group Optimization:")
    print("Improve solubility of monomer 1:")
    print(optimize_functional_groups(smiles1, smiles2, "improve_solubility", "1"))
    print()
    
    print("Add polar groups to monomer 1:")
    print(optimize_functional_groups(smiles1, smiles2, "add_polar_groups", "1"))
    print()
    
    # Test with a simpler molecule that has more suitable groups
    simple_smiles = "CCCC"  # Butane - has terminal methyl groups
    print("Improve solubility of simple molecule (CCCC):")
    print(optimize_functional_groups(simple_smiles, smiles2, "improve_solubility", "1"))
    print()
    
    # Test with a molecule that has ester-like structures for stability improvement
    ester_smiles = "CC(=O)OC"  # Methyl acetate - has C-O ester bond
    print("Improve stability of ester molecule (CC(=O)OC):")
    print(optimize_functional_groups(smiles1, smiles2, "improve_stability", "1"))
    print()
    
    #5. Structural suggestions
    print("5. Structural Improvement Suggestions:")
    suggestions = suggest_structural_improvements(smiles1, smiles2)
    for suggestion in suggestions["suggestions"]:
        print(f"- {suggestion}")
    print()
    
    #6. Create structural variants
    print("6. Structural Variants:")
    variants = create_structural_variants(smiles1, smiles2, "alkyl_variants", "1")
    print(f"Generated {len(variants)} alkyl variants")
    for variant in variants:  # Show all variants
        print(f"- {variant['variant_type']}: {variant['smiles']}")
    print()
    
    # 7. Stereochemistry modifications
    print("7. Stereochemistry Modifications:")
    # Test with a chiral molecule
    chiral_smiles = "C[C@H](N)C(=O)O"  # L-alanine
    print(f"Original chiral molecule: {chiral_smiles}")
    print(modify_stereochemistry(chiral_smiles, smiles2, "invert", "1"))
    print()
    
    # Test with an achiral molecule (polymer-like)
    print(f"Original achiral molecule: {smiles1}")
    print(modify_stereochemistry(smiles1, smiles2, "invert", "1"))
    print()
    
    # 8. Bond type modifications
    print("8. Bond Type Modifications:")
    # Test with molecules that can have bonds modified
    test_smiles1 = "C=C"  # Ethene - already has double bond, can convert to triple
    test_smiles2 = "CCN"  # Ethylamine
    print(f"Original molecule 1: {test_smiles1}")
    print(f"Original molecule 2: {test_smiles2}")
    print("Convert C=C double bond to triple in ethene:")
    print(modify_bond_types(test_smiles1, test_smiles2, "C=C", "triple", "1"))
    print()
    
    # Test with a molecule that can have aromatic bonds converted
    test_smiles3 = "c1ccccc1C"  # Toluene - has both aromatic and aliphatic bonds
    print(f"Original molecule 3: {test_smiles3}")
    print("Convert aromatic bond to single in toluene:")
    print(modify_bond_types(test_smiles3, test_smiles2, "cc", "single", "1"))
    print()

    print(modify_bond_types("C1#CCCCC1", "CCN", "C#C", "single", "1"))
    
    # 9. Structure comparison
    print("9. Structure Comparison:")
    modified_smiles1 = "CCNC1OC1Cc1ccccc1CCCC"  # Removed Br
    comparison = compare_structures(smiles1, smiles2, modified_smiles1, smiles2)
    print(f"Molecular weight change: {comparison['monomer1_changes']['molecular_weight']:.2f}")
    print(f"LogP change: {comparison['monomer1_changes']['logp']:.2f}")
    print()
    
    #10. Comprehensive structure report
    print("10. Comprehensive Structure Report:")
    smiles1 = "CCOC"
    smiles2 = "CCN"
    report = generate_structure_report(smiles1, smiles2)
    print(f"Structural similarity: {report['structural_similarity']:.3f}")
    print(f"Compatibility score: {report['compatibility_score']:.3f}")
    print(f"Number of suggestions: {len(report['suggestions'])}")
    print()
    
    # 10. Monomer Reaction Compatibility
    print("10. Monomer Reaction Compatibility:")
    print("Testing monomer compatibility analysis:")
    
    # Test 1: Compatible monomers (COOH + OH)
    compatible_test = make_monomers_reaction_compatible("CCO", "CC(=O)O")
    print("Test 1 - Compatible monomers (ethanol + acetic acid):")
    print(f"Reaction types: {compatible_test.get('reaction_types', [])}")
    print(f"Suggestions: {compatible_test.get('suggestions', [])}")
    print()
    
    # Test 2: Incompatible monomers (both have OH)
    incompatible_test = make_monomers_reaction_compatible("CCO", "CCO")
    print("Test 2 - Incompatible monomers (both ethanol):")
    print(f"Compatibility issues: {incompatible_test.get('compatibility_issues', [])}")
    print(f"Suggestions: {incompatible_test.get('suggestions', [])}")
    print()
    
    # Test 3: Vinyl polymerization
    vinyl_test = make_monomers_reaction_compatible("C=C", "C=C")
    print("Test 3 - Vinyl monomers (ethene + ethene):")
    print(f"Reaction types: {vinyl_test.get('reaction_types', [])}")
    print(f"Suggestions: {vinyl_test.get('suggestions', [])}")
    print()
    
    # Test 4: Epoxide-Imine polymerization (with multiplicity)
    epoxide_imine_test = make_monomers_reaction_compatible("C1OC1C1OC1", "C=NC=NC")
    print("Test 4 - Epoxide-Imine monomers (2 epoxides + 2 imines):")
    print(f"Reaction types: {epoxide_imine_test.get('reaction_types', [])}")
    print(f"Suggestions: {epoxide_imine_test.get('suggestions', [])}")
    print()
    
    # Test 5: Vinyl-Acrylate polymerization
    vinyl_acrylate_test = make_monomers_reaction_compatible("C=C", "C(=O)OC=C")
    print("Test 5 - Vinyl-Acrylate monomers:")
    print(f"Reaction types: {vinyl_acrylate_test.get('reaction_types', [])}")
    print(f"Suggestions: {vinyl_acrylate_test.get('suggestions', [])}")
    print()
    
    # Test 6: Vinyl-Hydroxyl polymerization
    vinyl_hydroxyl_test = make_monomers_reaction_compatible("C=C", "CCO")
    print("Test 6 - Vinyl-Hydroxyl monomers:")
    print(f"Reaction types: {vinyl_hydroxyl_test.get('reaction_types', [])}")
    print(f"Suggestions: {vinyl_hydroxyl_test.get('suggestions', [])}")
    print()

    # 11. Comprehensive Structure Report