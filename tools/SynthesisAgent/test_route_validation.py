#!/usr/bin/env python3
"""
Test script to demonstrate route validation and connectivity improvement.
This script shows how the new features address the disconnected steps issue.
"""

import json
from synthesis import validate_route_continuity, improve_route_connectivity, filter_quality_routes

def test_route_validation():
    """Test the route validation functionality with sample data."""
    
    # Sample route with disconnected steps (similar to the issue you identified)
    sample_route = {
        "steps": [
            {
                "step_number": 1,
                "reactants": ["O=C(OO)c1cccc(Cl)c1 (in stock)"],
                "products": ["C=CCOC1CCC(C(C)(C)C2CCC(OCC(O)COC3CCC(C(C)(C)C4CCC(OCC5CO5)CC4)CC3)CC2)CC1"],
                "reaction_smiles": "[O:1]1[C:5]([CH2:4][O:3][CH3:2])[C:6]1>>O=C(O[O:1])c1cccc(Cl)c1.[CH3:2][O:3][CH2:4][C:5]=[C:6]",
                "template": "[C:2]-[#8:3]-[C:4]-[CH;D3;+0:5]1-[CH2;D2;+0:6]-[O;H0;D2;+0:1]-1>>Cl-c1:c:c:c:c(-C(=O)-O-[OH;D1;+0:1]):c:1.[C:2]-[#8:3]-[C:4]-[CH;D2;+0:5]=[CH2;D1;+0:6]"
            },
            {
                "step_number": 2,
                "reactants": [],  # Missing reactants - this is the problem!
                "products": [
                    "C=CCOC1CCC(C(C)(C)C2CCC(O)CC2)CC1",
                    "CC(C)(C1CCC(OCC(O)CCl)CC1)C1CCC(OCC2CO2)CC1"
                ],
                "reaction_smiles": "[C:1]([CH2:2][OH:3])[O:5][CH3:4]>>Cl[C:1][CH2:2][OH:3].[CH3:4][O:5]",
                "template": "[#8:3]-[C:2]-[CH2;D2;+0:1]-[O;H0;D2;+0:5]-[C:4]>>Cl-[CH2;D2;+0:1]-[C:2]-[#8:3].[C:4]-[OH;D1;+0:5]"
            },
            {
                "step_number": 3,
                "reactants": ["C=CCBr (in stock)", "CC(C)(C1CCC(O)CC1)C1CCC(O)CC1 (in stock)"],
                "products": [],
                "reaction_smiles": "[C:1]([CH:2]=[CH2:3])[O:5][CH3:4]>>Br[C:1][CH:2]=[CH2:3].[CH3:4][O:5]",
                "template": "[C:3]=[C:2]-[CH2;D2;+0:1]-[O;H0;D2;+0:5]-[C:4]>>Br-[CH2;D2;+0:1]-[C:2]=[C:3].[C:4]-[OH;D1;+0:5]"
            }
        ]
    }
    
    print("🔍 Testing Route Validation")
    print("=" * 50)
    
    # Test 1: Validate original route (should show disconnections)
    print("\n📋 Original Route (with disconnections):")
    continuity_score, disconnected_steps = validate_route_continuity(sample_route["steps"])
    print(f"Continuity Score: {continuity_score:.2f}")
    print(f"Disconnected Steps: {disconnected_steps}")
    
    # Test 2: Improve connectivity
    print("\n🔧 Improving Route Connectivity:")
    improved_steps = improve_route_connectivity(sample_route["steps"])
    
    # Test 3: Validate improved route
    print("\n📋 Improved Route:")
    improved_continuity_score, improved_disconnected_steps = validate_route_continuity(improved_steps)
    print(f"Improved Continuity Score: {improved_continuity_score:.2f}")
    print(f"Remaining Disconnected Steps: {improved_disconnected_steps}")
    
    # Test 4: Quality filtering
    print("\n🎯 Quality Filtering:")
    sample_routes = [
        {
            "score": 0.95,
            "continuity_score": 0.8,
            "is_continuous": True,
            "steps": improved_steps
        },
        {
            "score": 0.85,
            "continuity_score": 0.3,
            "is_continuous": False,
            "steps": sample_route["steps"]  # Original disconnected route
        }
    ]
    
    quality_routes = filter_quality_routes(sample_routes, min_score=0.9, min_continuity=0.8)
    print(f"Quality routes passed: {len(quality_routes)}/{len(sample_routes)}")
    
    return {
        "original_continuity": continuity_score,
        "improved_continuity": improved_continuity_score,
        "quality_routes_count": len(quality_routes)
    }

if __name__ == "__main__":
    results = test_route_validation()
    print(f"\n✅ Test completed successfully!")
    print(f"Route connectivity improved from {results['original_continuity']:.2f} to {results['improved_continuity']:.2f}")
    print(f"Quality filtering passed {results['quality_routes_count']} routes") 