# Structure Fine-Tuning Package

A modular Python package for molecular structure fine-tuning and analysis, designed for polymer chemistry and molecular optimization.

## Overview

This package has been modularized from a single large file into multiple focused modules for better maintainability and organization. The original functionality is preserved while providing a cleaner, more organized structure.

## Module Structure

### Core Modules

1. **`molecular_properties.py`** - Property analysis and validation
   - `analyze_molecular_properties()` - Analyze molecular properties
   - `validate_smiles_pair()` - Validate SMILES pairs
   - `calculate_compatibility_score()` - Calculate monomer compatibility
   - `suggest_structural_improvements()` - Suggest structural improvements

2. **`structure_modifications.py`** - Core structure modification functions
   - `remove_bond_by_smarts()` - Remove bonds/groups using SMARTS
   - `add_group_by_smarts()` - Add functional groups using SMARTS
   - `modify_ring_system()` - Modify ring systems (aromatization/saturation)
   - `create_structural_variants()` - Create structural variants

3. **`functional_groups.py`** - Functional group analysis and optimization
   - `optimize_functional_groups()` - Optimize functional groups for properties
   - `analyze_functional_groups()` - Analyze functional groups in molecules
   - `has_functional_group()` - Check for specific functional groups
   - `suggest_functional_group_addition()` - Suggest functional group additions
   - `suggest_functional_group_removal()` - Suggest functional group removals

4. **`stereochemistry.py`** - Stereochemistry and bond type modifications
   - `modify_stereochemistry()` - Modify chiral centers
   - `modify_bond_types()` - Modify bond types (single, double, triple, aromatic)

5. **`reaction_compatibility.py`** - Reaction compatibility analysis
   - `make_monomers_reaction_compatible()` - Analyze monomer reaction compatibility

6. **`structure_analysis.py`** - Structure comparison and reporting
   - `compare_structures()` - Compare original and modified structures
   - `generate_structure_report()` - Generate comprehensive structure reports

### Interface Files

7. **`main.py`** - Main interface and convenience class
   - Imports all functions from modular files
   - Provides `StructureFineTuner` convenience class
   - Exports all functions for easy access

8. **`__init__.py`** - Package initialization
   - Makes the directory a Python package
   - Provides easy imports for all functions

9. **`structurefinetune.py`** - Legacy interface (backward compatibility)
   - Maintains the original file name for backward compatibility
   - Imports all functions from modular structure
   - Preserves original functionality

## Usage

### Method 1: Direct imports from modules

```python
from agentbasedP2P.tools.StructureFineTuning.molecular_properties import analyze_molecular_properties
from agentbasedP2P.tools.StructureFineTuning.structure_modifications import remove_bond_by_smarts
from agentbasedP2P.tools.StructureFineTuning.reaction_compatibility import make_monomers_reaction_compatible

# Use functions directly
properties = analyze_molecular_properties("CCO")
result = remove_bond_by_smarts("CCO", "CCN", "OH", "1")
compatibility = make_monomers_reaction_compatible("CCO", "CC(=O)O")
```

### Method 2: Using the main interface

```python
from agentbasedP2P.tools.StructureFineTuning.main import (
    analyze_molecular_properties,
    remove_bond_by_smarts,
    make_monomers_reaction_compatible
)

# Use functions
properties = analyze_molecular_properties("CCO")
result = remove_bond_by_smarts("CCO", "CCN", "OH", "1")
compatibility = make_monomers_reaction_compatible("CCO", "CC(=O)O")
```

### Method 3: Using the convenience class

```python
from agentbasedP2P.tools.StructureFineTuning.main import StructureFineTuner

# Create instance
tuner = StructureFineTuner()

# Use methods
properties = tuner.analyze_molecular_properties("CCO")
result = tuner.remove_bond_by_smarts("CCO", "CCN", "OH", "1")
compatibility = tuner.make_monomers_reaction_compatible("CCO", "CC(=O)O")
```

### Method 4: Legacy interface (backward compatibility)

```python
from agentbasedP2P.tools.StructureFineTuning.structurefinetune import (
    analyze_molecular_properties,
    remove_bond_by_smarts,
    make_monomers_reaction_compatible
)

# Use functions (same as before)
properties = analyze_molecular_properties("CCO")
result = remove_bond_by_smarts("CCO", "CCN", "OH", "1")
compatibility = make_monomers_reaction_compatible("CCO", "CC(=O)O")
```

## Key Features

### Molecular Property Analysis
- Calculate molecular weight, LogP, TPSA, HBD, HBA, rotatable bonds
- Validate SMILES pairs and calculate compatibility scores
- Suggest structural improvements based on properties

### Structure Modifications
- Remove bonds/groups using SMARTS patterns
- Add functional groups with attachment points
- Modify ring systems (aromatization/saturation)
- Create structural variants (alkyl chains, functional groups)

### Functional Group Optimization
- Optimize for solubility, stability, and polar groups
- Analyze functional group composition
- Suggest additions/removals for reaction compatibility

### Stereochemistry and Bond Types
- Modify chiral centers (invert, racemize)
- Change bond types (single, double, triple, aromatic)
- Handle valence rules and chemical constraints

### Reaction Compatibility
- Analyze monomer pairs for polymerization reactions
- Support multiple reaction types:
  - Condensation (esterification, amidation)
  - Addition (vinyl polymerization)
  - Ring-opening (epoxide-hydroxyl, epoxide-imine)
  - Vinyl-acrylate and vinyl-hydroxyl reactions
- Suggest modifications for compatibility

### Structure Analysis
- Compare original and modified structures
- Generate comprehensive property reports
- Calculate structural similarity and compatibility scores

## Benefits of Modularization

1. **Maintainability**: Each module focuses on a specific area of functionality
2. **Readability**: Smaller files are easier to understand and navigate
3. **Reusability**: Functions can be imported individually as needed
4. **Testing**: Each module can be tested independently
5. **Extensibility**: New functionality can be added to appropriate modules
6. **Backward Compatibility**: Original interface is preserved

## File Sizes

- Original `structurefinetune.py`: ~1088 lines
- Modular files: 50-200 lines each
- Total modular structure: ~800 lines (more efficient organization)

## Dependencies

- RDKit (for molecular operations)
- NumPy (for numerical operations)
- RDKit Chem (for cheminformatics)

## Testing

Run the demo to test all functionality:

```bash
python agentbasedP2P/tools/StructureFineTuning/structurefinetune.py
```

This will run comprehensive tests of all modular functions and demonstrate the new structure. 