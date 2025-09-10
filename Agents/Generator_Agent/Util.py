from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np
from rdkit.Chem import DataStructs


def remove_bond_by_smarts(smiles1: str,  bond_smarts: str) -> str:
    # Select which molecule to modify
    target_smiles = smiles1
   
    

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
    return modified_monomer
    

def add_group_by_smarts(smiles1: str, group_smarts: str,attachment_atom_idx: int = 0) -> str:
   
    # Select which molecule to modify
    target_smiles = smiles1
   

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
        return modified_smiles
    except:
        return "Failed to sanitize combined molecule"
    
# ===== Minimal helpers for simple strategy application =================
# pip install rdkit-pypi
from rdkit import Chem
from rdkit.Chem import AllChem

# -- basic SMILES utils --
def cano(smi: str):
    m = Chem.MolFromSmiles(smi)
    if not m: return None
    try:
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m, canonical=True)
    except:
        return None

def get_mol(smi: str):
    try:
        m = Chem.MolFromSmiles(smi)
        if m: Chem.SanitizeMol(m)
        return m
    except:
        return None

def replace_once(smi: str, patt_smarts: str, repl_smiles: str):
    """One SMARTS substitution; returns canonical SMILES or None."""
    m = get_mol(smi)
    if not m: return None
    patt = Chem.MolFromSmarts(patt_smarts)
    repl = Chem.MolFromSmiles(repl_smiles)
    if not patt or not repl: return None
    rms = AllChem.ReplaceSubstructs(m, patt, repl, replaceAll=False)
    if not rms: return None
    return cano(Chem.MolToSmiles(rms[0], canonical=True))


def finite_influence_side(A: str, B: str, Tg_hat: float, Er_hat: float, rolesAB):
    """
    Decide whether monomer A or B is more 'influential' on Tg/Er.
    Does a tiny neutral tweak to each and checks change in predicted props.
    Returns 'A' or 'B'.
    """
    # neutral tweaks (soft, unlikely to break reactivity)
    forbidA = role_forbidden_idxs(rolesAB[0], get_mol(A))
    forbidB = role_forbidden_idxs(rolesAB[1], get_mol(B))

    probeA = tweak_aliphatic_run(A, +1) or insert_between_carbons_nonring(A, "O", forbidA) or A
    probeB = tweak_aliphatic_run(B, +1) or insert_between_carbons_nonring(B, "O", forbidB) or B

    dA = dB = 0.0
    try:
        t, e = predict_property(probeA, B)
        dA = abs(t - Tg_hat) + abs(e - Er_hat)
    except:
        pass
    try:
        t, e = predict_property(A, probeB)
        dB = abs(t - Tg_hat) + abs(e - Er_hat)
    except:
        pass

    return 'A' if dA >= dB else 'B'
def role_forbidden_idxs(role: str, mol: Chem.Mol):
    """
    Return atom indices in `mol` that should be protected
    (not touched during edits) for a given reactive role.
    """
    if role == "epoxy":    # protect the oxirane ring atoms
        return _epoxide_idxs(mol)
    if role == "thiol":    # protect the sulfur atom in –SH
        return _thiol_idxs(mol)
    if role == "vinyl":    # protect the two carbons of C=C
        return _vinyl_idxs(mol)
    if role == "acrylate": # protect the C=C and ester atoms of acrylate
        return _acrylate_idxs(mol)
    if role == "hydroxyl": # protect the oxygen atom of –OH
        return _hydroxyl_idxs(mol)
    # for amines we don’t target N directly — we avoid N substitutions in our ops
    return set()


def tweak_aliphatic_run(smi: str, delta: int):
    """
    Very simple –(CH2)n– +/- 1 using string patterns (fast & robust).
    """
    inc = [("CCC","CCCC"), ("CC","CCC")]
    dec = [("CCCCC","CCCC"), ("CCCC","CCC"), ("CCC","CC")]
    patterns = inc if delta>0 else dec
    for a,b in patterns:
        out = smi.replace(a, b, 1)
        if out != smi:
            out = cano(out)
            if out: return out
    return None


def _terminal_heavy_atoms(m):
    t = [a.GetIdx() for a in m.GetAtoms() if a.GetDegree()==1 and a.GetAtomicNum()>1]
    return t if len(t)>=2 else None

def insert_between_carbons_nonring2(smi: str, linker_smiles: str, forbid_idxs: set):
    m = get_mol(smi)
    if not m: return None

    # pick a C–C single bond not in ring, not touching forbidden atoms
    cand = None
    for b in m.GetBonds():
        if b.GetBondType()!=Chem.BondType.SINGLE or b.IsInRing(): continue
        i,j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in forbid_idxs or j in forbid_idxs: continue
        if m.GetAtomWithIdx(i).GetSymbol()=='C' and m.GetAtomWithIdx(j).GetSymbol()=='C':
            cand = (i,j); break
    if cand is None: return None
    i,j = cand

    link = get_mol(linker_smiles)
    if not link: return None
    ends = _terminal_heavy_atoms(link)
    first = 0 if not ends else ends[0]
    last  = (link.GetNumAtoms()-1) if not ends else ends[1]

    em = Chem.EditableMol(m); em.RemoveBond(i,j)
    base = em.GetMol()
    combo = Chem.CombineMols(base, link)
    off = base.GetNumAtoms()
    em2 = Chem.EditableMol(combo)
    em2.AddBond(i, off+first, Chem.BondType.SINGLE)
    em2.AddBond(off+last, j, Chem.BondType.SINGLE)
    try:
        mol2 = em2.GetMol(); Chem.SanitizeMol(mol2)
        return cano(Chem.MolToSmiles(mol2, canonical=True))
    except:
        return None


def insert_between_carbons_nonring(smi: str, linker_smiles: str, forbid_idxs: set):
    """
    Insert a tiny linker (e.g., 'O') into a random non-ring C–C bond
    that does NOT touch forbidden atom indices (protect reactive handles).
    """
    m = get_mol(smi)
    if not m: return None
    # choose a candidate bond
    cands = []
    for b in m.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing():
            continue
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in forbid_idxs or j in forbid_idxs: 
            continue
        ai, aj = m.GetAtomWithIdx(i), m.GetAtomWithIdx(j)
        if ai.GetSymbol() == "C" and aj.GetSymbol() == "C":
            cands.append((i, j))
    if not cands: return None

    i, j = cands[0]  # deterministic pick; swap to random.choice(cands) if you like
    em = Chem.EditableMol(m)
    em.RemoveBond(i, j)

    link = get_mol(linker_smiles)   # e.g., "O" or "CS(=O)(=O)C" etc.
    if not link: return None
    base = em.GetMol()
    combo = Chem.CombineMols(base, link)
    em2 = Chem.EditableMol(combo)
    off = base.GetNumAtoms()
    # connect base[i]—link[0] and link[last]—base[j]
    em2.AddBond(i, off+0, Chem.BondType.SINGLE)
    em2.AddBond(off + link.GetNumAtoms() - 1, j, Chem.BondType.SINGLE)
    try:
        mol2 = em2.GetMol()
        Chem.SanitizeMol(mol2)
        return cano(Chem.MolToSmiles(mol2, canonical=True))
    except:
        return None

# -- reactive-handle detectors (guards) --
def _epoxide_idxs(m):
    out=set()
    ri = m.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring)==3:
            syms=[m.GetAtomWithIdx(i).GetSymbol() for i in ring]
            if syms.count('O')==1 and syms.count('C')==2:
                out.update(ring)
    return out
def _thiol_idxs(m):
    return set(i for (i,) in m.GetSubstructMatches(Chem.MolFromSmarts("[SH]")))
def _vinyl_idxs(m):
    idxs=set()
    for a,b in m.GetSubstructMatches(Chem.MolFromSmarts("C=C")):
        idxs.update([a,b])
    return idxs
def _acrylate_idxs(m):
    idxs=set()
    patt = Chem.MolFromSmarts("C=CC(=O)O")
    for tpl in m.GetSubstructMatches(patt):
        idxs.update(tpl)
    return idxs
def _hydroxyl_idxs(m):
    return set(i for (i,) in m.GetSubstructMatches(Chem.MolFromSmarts("[OX2H]")))

def has_epoxide(m):  return len(_epoxide_idxs(m))>0
def has_amine(m):    return m.HasSubstructMatch(Chem.MolFromSmarts("[NX3H2,NX3H1]"))
def has_thiol(m):    return len(_thiol_idxs(m))>0
def has_vinyl(m):    return len(_vinyl_idxs(m))>0
def has_acrylate(m): return len(_acrylate_idxs(m))>0
def has_hydroxyl(m): return len(_hydroxyl_idxs(m))>0

def preserves_role(m, role: str) -> bool:
    return (
        (role=="epoxy"    and has_epoxide(m))  or
        (role=="amine"    and has_amine(m))    or
        (role=="thiol"    and has_thiol(m))    or
        (role=="vinyl"    and has_vinyl(m))    or
        (role=="acrylate" and has_acrylate(m)) or
        (role=="hydroxyl" and has_hydroxyl(m))
    )

def role_forbidden_idxs(role: str, mol: Chem.Mol):
    if role=="epoxy":    return _epoxide_idxs(mol)
    if role=="thiol":    return _thiol_idxs(mol)
    if role=="vinyl":    return _vinyl_idxs(mol)
    if role=="acrylate": return _acrylate_idxs(mol)
    if role=="hydroxyl": return _hydroxyl_idxs(mol)
    # for amines we avoid direct N edits by not using N-targeting ops
    return set()

def valid_pair_roles(A: str, B: str, rolesAB):
    mA, mB = get_mol(A), get_mol(B)
    if not (mA and mB): return False
    return preserves_role(mA, rolesAB[0]) and preserves_role(mB, rolesAB[1])

# -- family detection (very lightweight) --
_FAMILIES = [
    (("epoxide","amine"),         ("epoxy","amine")),
    (("thiol","vinyl"),           ("thiol","vinyl")),
    (("vinyl","vinyl"),           ("vinyl","vinyl")),
    (("vinyl","acrylate"),        ("vinyl","acrylate")),
    (("vinyl","hydroxyl"),        ("vinyl","hydroxyl")),
    (("acrylate","acrylate"),     ("acrylate","acrylate")),
]

def _handles(m):
    return {
        "epoxide":  has_epoxide(m),
        "amine":    has_amine(m),
        "thiol":    has_thiol(m),
        "vinyl":    has_vinyl(m),
        "acrylate": has_acrylate(m),
        "hydroxyl": has_hydroxyl(m),
    }

def assign_family(A: str, B: str):
    mA, mB = get_mol(A), get_mol(B)
    if not (mA and mB): return None
    hA, hB = _handles(mA), _handles(mB)
    # straight orientation
    for needA, needB in [x[0] for x in _FAMILIES]:
        roles = dict(_FAMILIES)[(needA, needB)]
        if hA[needA] and hB[needB]:
            return {"family": f"{needA}-{needB}".replace("epoxide","epoxy"), "roles": roles}
    # swap if needed
    for needA, needB in [x[0] for x in _FAMILIES]:
        roles = dict(_FAMILIES)[(needA, needB)]
        if hA[needB] and hB[needA]:
            return {"family": f"{needB}-{needA}".replace("epoxide","epoxy"), "roles": roles[::-1]}
    return None


def _epoxide_idxs(m):   # 3-membered oxirane ring O + 2 C
    out=set()
    ri = m.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring)==3:
            syms=[m.GetAtomWithIdx(i).GetSymbol() for i in ring]
            if syms.count('O')==1 and syms.count('C')==2:
                out.update(ring)
    return out

def _thiol_idxs(m):
    return set(i for (i,) in m.GetSubstructMatches(Chem.MolFromSmarts("[SH]")))

def _vinyl_idxs(m):
    idxs=set()
    for a,b in m.GetSubstructMatches(Chem.MolFromSmarts("C=C")):
        idxs.update([a,b])
    return idxs

def _acrylate_idxs(m):
    idxs=set()
    patt = Chem.MolFromSmarts("C=CC(=O)O")   # acrylate motif
    for tpl in m.GetSubstructMatches(patt):
        idxs.update(tpl)
    return idxs

def _hydroxyl_idxs(m):
    return set(i for (i,) in m.GetSubstructMatches(Chem.MolFromSmarts("[OX2H]")))

# ===== Minimal “apply strategy once” (pairs with your plan) =============
def apply_strategy_once(A, B, plan, rolesAB):
    """
    Apply the first 1–2 actions from plan to A (simple version).
    If an edit keeps roles valid, return the new pair immediately.
    """
    for act in plan["actions"]:
        name = act["name"]

        # 1) Aromatize (cyclohexyl -> phenyl) to raise Tg
        if name == "add_aromatic_ring":
            A1 = replace_once(A, "C1CCCCC1", "c1ccccc1")
            if A1 and valid_pair_roles(A1, B, rolesAB):
                return A1, B

        # 2) Insert ether/thio (lower Tg / raise Er a bit)
        if name == "insert_ether_or_thioether":
            forbid = role_forbidden_idxs(rolesAB[0], get_mol(A))
            A1 = insert_between_carbons_nonring(A, "O", forbid)
            if A1 and valid_pair_roles(A1, B, rolesAB):
                return A1, B

        # 3) Shorten or lengthen –(CH2)n–
        if name in {"shorten_aliphatic_spacer", "lengthen_aliphatic_spacer"}:
            delta = -1 if name=="shorten_aliphatic_spacer" else +1
            A1 = tweak_aliphatic_run(A, delta)
            if A1 and valid_pair_roles(A1, B, rolesAB):
                return A1, B

        # 4) Bulky pendant (lower Tg)
        if name == "add_bulky_pendant":
            A1 = replace_once(A, "[CH3]", "C(C)(C)C")
            if A1 and valid_pair_roles(A1, B, rolesAB):
                return A1, B

    # if none applied, return original
    return A, B

    # --- utilities to apply a single action to one monomer -----------------
def template_to_linker_smiles(template: str) -> str:
    """
    Convert two-ended template like '[*]O[*]' or '[*][Si]O[*]' into
    a real, insertable molecule with terminal heavy atoms.
    """
    # common mappings (extend as needed)
    mapping = {
        "[*]O[*]": "COC",                 # ether
        "[*]S[*]": "CSC",
        "[*]OC(=O)O[*]": "COC(=O)OC",                 # thioether
        "[*]OC(=O)N[*]": "COC(=O)NC",     # urethane
        "[*]NC(=O)[*]": "CNC(=O)C",       # amide (one of many choices)
        "[*]S(=O)(=O)[*]": "CS(=O)(=O)C", # sulfone
        "[*][Si]O[*]": "O[Si](C)(C)O",    # siloxane (use O-terminated)
    }
    if template in mapping:
        return mapping[template]
    # fallback: replace [*] with 'C' and hope the ends are heavy atoms
    # (works for simple linkers like [*]O[*] -> COC, but not for everything)
    return template.replace("[*]", "C").replace("[Si]", "[Si]")

def _apply_action_to_one_side(smi: str, act: dict, forbid_idxs: set):
    """
    Returns a list of mutated SMILES for `smi` according to `act`,
    respecting forbidden atom indices (so we don't break reactive handles).
    """
    out = []
    name = act["name"]
    pats = act.get("patterns", {})

    # 1) Replace aliphatic ring -> phenyl (aromatize)
    if name == "add_aromatic_ring":
        r = replace_once(smi, "C1CCCCC1", "c1ccccc1")
        if r: out.append(r)

    # 2) Insert sulfone / amide / urethane / ether / thio / siloxane
    elif name in {"insert_sulfone_linker", "insert_ether_or_thioether", "insert_siloxane",
                  "insert_amide_or_urethane", "sprinkle_ether_linkers",
                  "add_urethane_or_amide_islands"}:
        # unify into a set of linkers to try
        linkers = []
        if "insert" in pats: linkers.append(pats["insert"])
        linkers += [pats.get(k) for k in ("ether","thioether","siloxane","urethane","amide") if pats.get(k)]
        # defaults if not provided
        if not linkers:
            if name == "insert_sulfone_linker": linkers = ["[*]S(=O)(=O)[*]"]
            elif name == "insert_ether_or_thioether": linkers = ["[*]O[*]","[*]S[*]"]
            elif name == "insert_siloxane": linkers = ["[*][Si]O[*]"]
            elif name in {"insert_amide_or_urethane","add_urethane_or_amide_islands"}: linkers = ["[*]OC(=O)N[*]","[*]NC(=O)[*]"]
            elif name == "sprinkle_ether_linkers": linkers = ["[*]O[*]"]

        # try each linker by “materializing” it as a small molecule,
        # then inserting between non-ring C–C bonds:
        for lk in linkers:
            # Convert dummy-bridges [*]X[*] to a minimal linker SMILES for insertion:
            # We replace [*] with methyl "C" ends; the Edit op will re-connect ends to C–C
            lk_mono = template_to_linker_smiles(lk) #lk.replace("[*]", "C")
            r = insert_between_carbons_nonring(smi, lk_mono, forbid_idxs)
            if r: out.append(r)

    # 3) Change spacer length
    elif name in {"shorten_aliphatic_spacer", "lengthen_aliphatic_spacer"}:
        delta = -1 if name == "shorten_aliphatic_spacer" else +1
        r = tweak_aliphatic_run(smi, delta=delta)
        if r: out.append(r)

    # 4) Add bulky pendant (tBu)
    elif name == "add_bulky_pendant":
        r = replace_once(smi, "[CH3]", "C(C)(C)C")
        if r: out.append(r)

    # 5) π–π helper (phenyl -> biphenyl) only if aromatic exists
    elif name == "pi_pi_helper_if_aromatic_present":
        m = get_mol(smi)
        if m and m.HasSubstructMatch(Chem.MolFromSmarts("c1ccccc1")):
            r = replace_once(smi, "c1ccccc1", "c1ccc(cc1)c2ccccc2")
            if r: out.append(r)

    # 6) Mild rigidifiers for Er decrease
    elif name == "add_mild_rigidifiers":
        # try add phenyl and/or insert sulfone
        r = replace_once(smi, "C1CCCCC1", "c1ccccc1")
        if r: out.append(r)
        r = insert_between_carbons_nonring(smi, "CS(=O)(=O)C", forbid_idxs)  # minimal –SO2– insertion scaffold
        if r: out.append(r)

    # de-dup + canonicalize
    uniq, seen = [], set()
    for s in out:
        cs = cano(s)
        if cs and cs not in seen:
            seen.add(cs); uniq.append(cs)
    return uniq


def apply_action_to_pair(A: str, B: str, act: dict, rolesAB):
    """
    Generate candidate (A',B') by applying `act` to A or B,
    preserving reactive rolesAB.
    """
    mA, mB = get_mol(A), get_mol(B)
    forbA = role_forbidden_idxs(rolesAB[0], mA)
    forbB = role_forbidden_idxs(rolesAB[1], mB)

    cands = []
    # try editing A only
    for a1 in _apply_action_to_one_side(A, act, forbA):
        if valid_pair_roles(a1, B, rolesAB):
            cands.append((a1, B))
    # try editing B only
    for b1 in _apply_action_to_one_side(B, act, forbB):
        if valid_pair_roles(A, b1, rolesAB):
            cands.append((A, b1))
    # small number of two-sided combos (optional)
    for a1 in _apply_action_to_one_side(A, act, forbA)[:2]:
        for b1 in _apply_action_to_one_side(B, act, forbB)[:2]:
            if valid_pair_roles(a1, b1, rolesAB):
                cands.append((a1, b1))

    # de-dup
    uniq, seen = [], set()
    for a1, b1 in cands:
        key = a1 + "|" + b1
        if key not in seen:
            seen.add(key); uniq.append((a1, b1))
    return uniq

