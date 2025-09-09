

# Import all functions from the modular structure
from StructureFineTuning_Agent.molecular_properties import (
    analyze_molecular_properties,
    validate_smiles_pair,
    calculate_compatibility_score,
    suggest_structural_improvements
)

from StructureFineTuning_Agent.structure_modifications import (
    remove_bond_by_smarts,
    add_group_by_smarts,
    modify_ring_system,
    create_structural_variants
)

from StructureFineTuning_Agent.functional_groups import (
    optimize_functional_groups,
    analyze_functional_groups,
    has_functional_group,
    suggest_functional_group_addition,
    suggest_functional_group_removal
)

from StructureFineTuning_Agent.stereochemistry import (
    modify_stereochemistry,
    modify_bond_types
)

from StructureFineTuning_Agent.reaction_compatibility import (
    make_monomers_reaction_compatible
)

from StructureFineTuning_Agent.structure_analysis import (
    compare_structures,
    generate_structure_report
)

# Export all functions for easy access
__all__ = [
    # Molecular properties
    "analyze_molecular_properties",
    "validate_smiles_pair", 
    "calculate_compatibility_score",
    "suggest_structural_improvements",
    
    # Structure modifications
    "remove_bond_by_smarts",
    "add_group_by_smarts",
    "modify_ring_system",
    "create_structural_variants",
    
    # Functional groups
    "optimize_functional_groups",
    "analyze_functional_groups",
    "has_functional_group",
    "suggest_functional_group_addition",
    "suggest_functional_group_removal",
    
    # Stereochemistry
    "modify_stereochemistry",
    "modify_bond_types",
    
    # Reaction compatibility
    "make_monomers_reaction_compatible",
    
    # Structure analysis
    "compare_structures",
    "generate_structure_report"
]

class StructureFineTuner:
    """
    Convenience class providing access to all structure fine-tuning functionality.
    """
    
    def __init__(self):
        """Initialize the StructureFineTuner with all available functions."""
        pass
    
    # Molecular properties methods
    def analyze_properties(self, smiles: str) -> dict:
        """Analyze molecular properties of a SMILES string."""
        return analyze_molecular_properties(smiles)
    
    def validate_pair(self, smiles1: str, smiles2: str) -> dict:
        """Validate a pair of SMILES strings."""
        return validate_smiles_pair(smiles1, smiles2)
    
    def calculate_compatibility(self, mol1, mol2) -> float:
        """Calculate compatibility score between two molecules."""
        return calculate_compatibility_score(mol1, mol2)
    
    def suggest_improvements(self, smiles1: str, smiles2: str) -> dict:
        """Suggest structural improvements for a pair of molecules."""
        return suggest_structural_improvements(smiles1, smiles2)
    
    # Structure modification methods
    def remove_bond(self, smiles1: str, smiles2: str, smarts_pattern: str, target_monomer: str = "1") -> str:
        """Remove bonds matching a SMARTS pattern."""
        return remove_bond_by_smarts(smiles1, smiles2, smarts_pattern, target_monomer)
    
    def add_group(self, smiles1: str, smiles2: str, smarts_pattern: str, group_smiles: str, target_monomer: str = "1") -> str:
        """Add a functional group to a molecule."""
        return add_group_by_smarts(smiles1, smiles2, smarts_pattern, group_smiles, target_monomer)
    
    def modify_ring(self, smiles1: str, smiles2: str, ring_smarts: str, modification_type: str, target_monomer: str = "1") -> str:
        """Modify ring systems in a molecule."""
        return modify_ring_system(smiles1, smiles2, ring_smarts, modification_type, target_monomer)
    
    def create_variants(self, smiles1: str, smiles2: str, variant_type: str, target_monomer: str = "1") -> list:
        """Create structural variants of a molecule."""
        return create_structural_variants(smiles1, smiles2, variant_type, target_monomer)
    
    # Functional group methods
    def optimize_groups(self, smiles1: str, smiles2: str, optimization_type: str, target_monomer: str = "1") -> str:
        """Optimize functional groups in a molecule."""
        return optimize_functional_groups(smiles1, smiles2, optimization_type, target_monomer)
    
    def analyze_groups(self, mol) -> dict:
        """Analyze functional groups in a molecule."""
        return analyze_functional_groups(mol)
    
    def has_group(self, fg_dict: dict, group_name: str) -> bool:
        """Check if a molecule has a specific functional group."""
        return has_functional_group(fg_dict, group_name)
    
    def suggest_addition(self, fg_dict1: dict, fg_dict2: dict) -> list:
        """Suggest functional group additions."""
        return suggest_functional_group_addition(fg_dict1, fg_dict2)
    
    def suggest_removal(self, fg_dict1: dict, fg_dict2: dict) -> list:
        """Suggest functional group removals."""
        return suggest_functional_group_removal(fg_dict1, fg_dict2)
    
    # Stereochemistry methods
    def modify_stereo(self, smiles1: str, smiles2: str, stereochemistry_type: str, target_monomer: str = "1") -> str:
        """Modify stereochemistry of a molecule."""
        return modify_stereochemistry(smiles1, smiles2, stereochemistry_type, target_monomer)
    
    def modify_bonds(self, smiles1: str, smiles2: str, bond_smarts: str, new_bond_type: str, target_monomer: str = "1") -> str:
        """Modify bond types in a molecule."""
        return modify_bond_types(smiles1, smiles2, bond_smarts, new_bond_type, target_monomer)
    
    # Reaction compatibility methods
    def make_compatible(self, smiles1: str, smiles2: str) -> dict:
        """Make two monomers reaction-compatible."""
        return make_monomers_reaction_compatible(smiles1, smiles2)
    
    # Structure analysis methods
    def compare_structures(self, smiles1_orig: str, smiles2_orig: str, 
                         smiles1_mod: str, smiles2_mod: str) -> dict:
        """Compare original and modified structures."""
        return compare_structures(smiles1_orig, smiles2_orig, smiles1_mod, smiles2_mod)
    
    def generate_report(self, smiles1: str, smiles2: str) -> dict:
        """Generate a comprehensive structure analysis report."""
        return generate_structure_report(smiles1, smiles2)

# def test_main():
#     """
#     Comprehensive test function for all structure fine-tuning functionality.
#     Tests all functions with various input cases and provides detailed output.
#     """
#     print("=" * 80)
#     print("STRUCTURE FINE-TUNING PACKAGE - COMPREHENSIVE TEST SUITE")
#     print("=" * 80)
    
   
#     # Initialize the fine-tuner
#     tuner = StructureFineTuner()
    
#     print("\n1. TESTING MOLECULAR PROPERTIES FUNCTIONS")
#     print("-" * 50)
    
#     smiles1 = "CCNC1OC1Cc1ccccc1CCCCBr"
#     smiles2 = "CCC2OC2COOCC"
#     print("Original smiles1: ", smiles1)
#     print("Original smiles2: ", smiles2)
#     print("Remove CN bond from smiles1: ", tuner.remove_bond(smiles1, smiles2, "CN", "1"))
#     print("Add carboxylic acid group to smiles2: ", tuner.add_group(smiles1, smiles2, "[*]C(=O)O", "2", 0))
    
#     print("2. Structure Analysis:")
#     analysis = tuner.validate_pair(smiles1, smiles2)
#     print(f"Compatibility Score: {analysis['compatibility_score']:.3f}")
#     print(f"Monomer 1 MW: {analysis['monomer1_properties']['molecular_weight']:.2f}")
#     print(f"Monomer 2 MW: {analysis['monomer2_properties']['molecular_weight']:.2f}")

#     print("3. Ring System Modifications:")
#     print("Aromatize ring in monomer 1 (already aromatic):")
#     print(tuner.modify_ring(smiles1, smiles2, "c1ccccc1", "aromatize", "1"))
#     print()

#     print("Aromatize cyclohexane ring (non-aromatic):")
#     cyclohexane_smiles = "CC1CCCCC1"
#     print(tuner.modify_ring(cyclohexane_smiles, smiles2, "C1CCCCC1", "aromatize", "1"))
#     print()

#     print("Saturate benzene ring (aromatic to saturated):")
#     print(tuner.modify_ring(smiles1, smiles2, "c1ccccc1", "saturate", "1"))
#     print()
#     print("=" * 80)

#      # 4. Functional group optimization
#     print("4. Functional Group Optimization:")
#     print("Improve solubility of monomer 1:")
#     print(tuner.optimize_groups(smiles1, smiles2, "improve_solubility", "1"))
#     print()

#     print("Add polar groups to monomer 1:")
#     print(tuner.optimize_groups(smiles1, smiles2, "add_polar_groups", "1"))
#     print()
    
#     # Test with a simpler molecule that has more suitable groups
#     simple_smiles = "CCCC"  # Butane - has terminal methyl groups
#     print("Improve solubility of simple molecule (CCCC):")
#     print(tuner.optimize_groups(simple_smiles, smiles2, "improve_solubility", "1"))
#     print()
    
#     # Test with a molecule that has ester-like structures for stability improvement
#     ester_smiles = "CC(=O)OC"  # Methyl acetate - has C-O ester bond
#     print("Improve stability of ester molecule (CC(=O)OC):")
#     print(tuner.optimize_groups(smiles1, smiles2, "improve_stability", "1"))
#     print()

#     print("5. Structural Improvement Suggestions:")
#     suggestions = tuner.suggest_improvements(smiles1, smiles2)
#     for suggestion in suggestions["suggestions"]:
#         print(f"- {suggestion}")
#     print()

#     print("6. Structural Variants:")
#     variants = tuner.create_variants(smiles1, smiles2, "alkyl_variants", "1")
#     print(f"Generated {len(variants)} alkyl variants")
#     for variant in variants:  # Show all variants
#         print(f"- {variant['variant_type']}: {variant['smiles']}")
#     print()

#     print("7. Stereochemistry Modifications:")
#     # Test with a chiral molecule
#     chiral_smiles = "C[C@H](N)C(=O)O"  # L-alanine
#     print(f"Original chiral molecule: {chiral_smiles}")
#     print(tuner.modify_stereo(chiral_smiles, smiles2, "invert", "1"))
#     print()
    
#     # Test with an achiral molecule (polymer-like)
#     print(f"Original achiral molecule: {smiles1}")
#     print(tuner.modify_stereo(smiles1, smiles2, "invert", "1"))
#     print()

#     print("8. Bond Type Modifications:")
#     # Test with molecules that can have bonds modified
#     test_smiles1 = "C=C"  # Ethene - already has double bond, can convert to triple
#     test_smiles2 = "CCN"  # Ethylamine
#     print(f"Original molecule 1: {test_smiles1}")
#     print(f"Original molecule 2: {test_smiles2}")
#     print("Convert C=C double bond to triple in ethene:")
#     print(tuner.modify_bonds(test_smiles1, test_smiles2, "C=C", "triple", "1"))
#     print()
    
#     # Test with a molecule that can have aromatic bonds converted
#     test_smiles3 = "c1ccccc1C"  # Toluene - has both aromatic and aliphatic bonds
#     print(f"Original molecule 3: {test_smiles3}")
#     print("Convert aromatic bond to single in toluene:")
#     print(tuner.modify_bonds(test_smiles3, test_smiles2, "cc", "single", "1"))
#     print()

#     print(tuner.modify_bonds("C1#CCCCC1", "CCN", "C#C", "single", "1"))

#      # 9. Structure comparison
#     print("9. Structure Comparison:")
#     modified_smiles1 = "CCNC1OC1Cc1ccccc1CCCC"  # Removed Br
#     comparison = tuner.compare_structures(smiles1, smiles2, modified_smiles1, smiles2)
#     print(f"Molecular weight change: {comparison['monomer1_changes']['molecular_weight']:.2f}")
#     print(f"LogP change: {comparison['monomer1_changes']['logp']:.2f}")
#     print()

#     print("10. Comprehensive Structure Report:")
#     smiles1 = "CCOC"
#     smiles2 = "CCN"
#     report = generate_structure_report(smiles1, smiles2)
#     print(f"Structural similarity: {report['structural_similarity']:.3f}")
#     print(f"Compatibility score: {report['compatibility_score']:.3f}")
#     print(f"Number of suggestions: {len(report['suggestions'])}")
#     print()
    
#     # 10. Monomer Reaction Compatibility
#     print("10. Monomer Reaction Compatibility:")
#     print("Testing monomer compatibility analysis:")
    
#     # Test 1: Compatible monomers (COOH + OH)
#     compatible_test = tuner.make_compatible("CCO", "CC(=O)O")
#     print("Test 1 - Compatible monomers (ethanol + acetic acid):")
#     print(f"Reaction types: {compatible_test.get('reaction_types', [])}")
#     print(f"Suggestions: {compatible_test.get('suggestions', [])}")
#     print()
    
#     # Test 2: Incompatible monomers (both have OH)
#     incompatible_test = tuner.make_compatible("CCO", "CCO")
#     print("Test 2 - Incompatible monomers (both ethanol):")
#     print(f"Compatibility issues: {incompatible_test.get('compatibility_issues', [])}")
#     print(f"Suggestions: {incompatible_test.get('suggestions', [])}")
#     print()
    
#     # Test 3: Vinyl polymerization
#     vinyl_test = tuner.make_compatible("C=C", "C=C")
#     print("Test 3 - Vinyl monomers (ethene + ethene):")
#     print(f"Reaction types: {vinyl_test.get('reaction_types', [])}")
#     print(f"Suggestions: {vinyl_test.get('suggestions', [])}")
#     print()
    
#     # Test 4: Epoxide-Imine polymerization (with multiplicity)
#     epoxide_imine_test = make_monomers_reaction_compatible("C1OC1C1OC1", "C=NC=NC")
#     print("Test 4 - Epoxide-Imine monomers (2 epoxides + 2 imines):")
#     print(f"Reaction types: {epoxide_imine_test.get('reaction_types', [])}")
#     print(f"Suggestions: {epoxide_imine_test.get('suggestions', [])}")
#     print()
    
#     # Test 5: Vinyl-Acrylate polymerization
#     vinyl_acrylate_test = tuner.make_compatible("C=C", "C(=O)OC=C")
#     print("Test 5 - Vinyl-Acrylate monomers:")
#     print(f"Reaction types: {vinyl_acrylate_test.get('reaction_types', [])}")
#     print(f"Suggestions: {vinyl_acrylate_test.get('suggestions', [])}")
#     print()
    
#     # Test 6: Vinyl-Hydroxyl polymerization
#     vinyl_hydroxyl_test = tuner.make_compatible("C=C", "CCO")
#     print("Test 6 - Vinyl-Hydroxyl monomers:")
#     print(f"Reaction types: {vinyl_hydroxyl_test.get('reaction_types', [])}")
#     print(f"Suggestions: {vinyl_hydroxyl_test.get('suggestions', [])}")
#     print()

    
    

    
    
#     return "All tests completed successfully!"

# if __name__ == "__main__":
#     # Run the comprehensive test suite
#     test_main() 