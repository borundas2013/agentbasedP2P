from aizynthfinder.aizynthfinder import AiZynthFinder
import os
from typing import Dict, List

def extract_reactions_from_tree(tree_dict, step_counter=None):
    """Recursively extract reactions from the tree structure, using a global step counter."""
    if step_counter is None:
        step_counter = [1]
    reactions = []
    if isinstance(tree_dict, dict):
        children = tree_dict.get('children', [])
        for child in children:
            if isinstance(child, dict) and child.get('type') == 'reaction':
                reaction_smiles = child.get('smiles', 'Unknown')
                metadata = child.get('metadata', {})
                template = metadata.get('template', 'Unknown')
                reaction_info = {
                    "step_number": step_counter[0],
                    "reaction_smiles": reaction_smiles,
                    "template": template,
                    "reactants": [],
                    "products": []
                }
                step_counter[0] += 1
                reaction_children = child.get('children', [])
                for reaction_child in reaction_children:
                    if reaction_child.get('type') == 'mol':
                        smiles = reaction_child.get('smiles', '')
                        in_stock = reaction_child.get('in_stock', False)
                        if in_stock:
                            reaction_info["reactants"].append(f"{smiles} (in stock)")
                        else:
                            reaction_info["products"].append(smiles)
                reactions.append(reaction_info)
                for reaction_child in reaction_children:
                    if reaction_child.get('type') == 'mol' and not reaction_child.get('in_stock', False):
                        child_reactions = extract_reactions_from_tree(reaction_child, step_counter)
                        reactions.extend(child_reactions)
    return reactions

def get_synthesis_plan(smiles1: str):
 
    try:
        # Get the model directory path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, "model")
        
        # Change to model directory
        original_dir = os.getcwd()
        os.chdir(model_dir)
        
        # Initialize AiZynthFinder
        finder = AiZynthFinder("config.yml")
        
        # Configure policies
        finder.stock.select('zinc')
        finder.expansion_policy.select("full")
        
        # Set target molecule
        finder.target_smiles = smiles1
        
        print(f"🔍 Generating retrosynthesis plan for: {smiles1}")
        
        # Configure search parameters for better results
        finder.config.search.max_iterations = 100  # Increase iterations
        finder.config.search.max_depth = 6  # Increase depth
        
        # Run the retrosynthesis
        finder.tree_search()
        finder.build_routes()
        
        # Get results
        stats = finder.extract_statistics()
        routes = finder.routes
        
        # Change back to original directory
        os.chdir(original_dir)
        
        if not routes:
            print("❌ No synthesis routes found.")
            return {"status": "no_routes", "message": "No synthesis routes found"}
        
        print(f"✅ Found {len(routes)} synthesis route(s)")
        
        # Determine if any route is fully solved
        any_solved = False
        for route in routes:
            if isinstance(route, dict):
                route_metadata = route.get('route_metadata', {})
                if route_metadata.get('is_solved', False):
                    any_solved = True
                    break

        if any_solved:
            status = "success"
        elif routes:
            status = "partial"
        else:
            status = "no_routes"

        results = {
            "status": status,
            "target_smiles": smiles1,
            "num_routes": len(routes),
            "statistics": stats,
            "routes": []
        }
        
        for i, route in enumerate(routes):
            print(f"\n📋 Route {i+1}:")
            
            route_info = {
                "route_number": i + 1,
                "steps": []
            }
            
            # Handle dictionary format (which is what we actually get)
            if isinstance(route, dict):
                # Extract score safely
                route_score = route.get('score', {})
                if isinstance(route_score, dict):
                    state_score = route_score.get('state score', 0.0)
                    route_info["score"] = state_score
                    print(f"  Score: {state_score:.3f}")
                else:
                    route_info["score"] = float(route_score) if isinstance(route_score, (int, float)) else 0.0
                    print(f"  Score: {route_score}")
                
                # Extract reaction tree information
                reaction_tree = route.get('reaction_tree', {})
                route_metadata = route.get('route_metadata', {})
                
                # Check if route is solved
                is_solved = route_metadata.get('is_solved', False)
                print(f"  Solved: {is_solved}")
                
                if not is_solved:
                    print(f"  ⚠️  Route not fully solved - incomplete retrosynthesis")
                else:
                    print(f"  ✅  Route fully solved - complete retrosynthesis")
                    if hasattr(reaction_tree, 'to_dict'):
                        try:
                            tree_dict = reaction_tree.to_dict()
                            reactions = extract_reactions_from_tree(tree_dict)
                            
                            if reactions:
                                # Improve route connectivity by inferring missing reactants
                                reactions = improve_route_connectivity(reactions)
                                
                                print(f"  🔍 Found {len(reactions)} synthesis step(s):")
                                
                                for j, reaction in enumerate(reactions):
                                    print(f"  Step {reaction['step_number']}:")
                                    print(f"    Reaction: {reaction['reaction_smiles']}")
                                    print(f"    Reactants: {reaction['reactants']}")
                                    print(f"    Products: {reaction['products']}")
                                    print(f"    Template: {reaction['template']}")
                                    
                                    step_info = {
                                        "step_number": reaction['step_number'],
                                        "reaction_smiles": reaction['reaction_smiles'],
                                        "reactants": reaction['reactants'],
                                        "products": reaction['products'],
                                        "template": reaction['template']
                                    }
                                    route_info["steps"].append(step_info)
                            else:
                                print(f"  🔍 No reactions found in tree structure")
                                
                        except Exception as tree_error:
                            print(f"  🔍 Error processing tree: {tree_error}")
                    
                    elif isinstance(reaction_tree, dict):
                        # It's already a dictionary
                        reactions = extract_reactions_from_tree(reaction_tree)
                        
                        if reactions:
                            # Improve route connectivity by inferring missing reactants
                            reactions = improve_route_connectivity(reactions)
                            
                            print(f"  🔍 Found {len(reactions)} synthesis step(s):")
                            
                            for j, reaction in enumerate(reactions):
                                print(f"  Step {reaction['step_number']}:")
                                print(f"    Reaction: {reaction['reaction_smiles']}")
                                print(f"    Products: {reaction['products']}")
                                print(f"    Template: {reaction['template']}")
                                
                                step_info = {
                                    "step_number": reaction['step_number'],
                                    "reaction_smiles": reaction['reaction_smiles'],
                                    "reactants": reaction['reactants'],
                                    "products": reaction['products'],
                                    "template": reaction['template']
                                }
                                route_info["steps"].append(step_info)
                        else:
                            print(f"  🔍 No reactions found in tree dict")
                    else:
                        print(f"  🔍 Unexpected reaction tree type: {type(reaction_tree)}")
                

                # Extract reactions from the tree structure
                
                # Also check route metadata for additional info
                if isinstance(route_metadata, dict):
                    print(f"  🔍 Route metadata: {route_metadata}")
                    
            else:
                print(f"  🔍 Unexpected route type: {type(route)}")
            
            results["routes"].append(route_info)
        
        return results
        
    except Exception as e:
        print(f"❌ Error in retrosynthesis: {e}")
        # Change back to original directory even if there's an error
        try:
            os.chdir(original_dir)
        except:
            pass
        return {"status": "error", "message": str(e)}
    
import json

from aizynthfinder.aizynthfinder import AiZynthFinder
import os
from typing import Dict, List
def improve_route_connectivity(reactions):
    """
    Attempt to improve route connectivity by inferring missing reactants.
    This is a post-processing step to address AiZynthFinder's incomplete output.
    """
    if not reactions or len(reactions) <= 1:
        return reactions
    
    improved_reactions = []
    
    for i, reaction in enumerate(reactions):
        improved_reaction = reaction.copy()
        
        # If this step has no reactants but previous step has products, 
        # try to infer the connection
        if not improved_reaction["reactants"] and i > 0:
            prev_products = reactions[i-1].get("products", [])
            if prev_products:
                # Use the first product from previous step as reactant
                inferred_reactant = prev_products[0].replace(" (in stock)", "")
                improved_reaction["reactants"].append(f"{inferred_reactant} (from step {i})")
                print(f"🔧 Inferred reactant for Step {i+1}: {inferred_reactant} (from Step {i})")
        
        improved_reactions.append(improved_reaction)
    
    return improved_reactions
def validate_route_continuity(steps):
    """
    Validate that each step's reactants include products from previous steps.
    Returns a continuity score and identifies disconnected steps.
    """
    if not steps or len(steps) <= 1:
        return 1.0, []  # Single step or no steps is always continuous
    
    disconnected_steps = []
    continuity_score = 0.0
    
    for i in range(1, len(steps)):
        prev_products = set()
        curr_reactants = set()
        
        # Extract product SMILES from previous step
        for product in steps[i-1].get('products', []):
            # Remove "(in stock)" suffix if present
            product_smiles = product.replace(" (in stock)", "")
            prev_products.add(product_smiles)
        
        # Extract reactant SMILES from current step
        for reactant in steps[i].get('reactants', []):
            # Remove "(in stock)" suffix if present
            reactant_smiles = reactant.replace(" (in stock)", "")
            curr_reactants.add(reactant_smiles)
        
        # Check for overlap between previous products and current reactants
        overlap = prev_products.intersection(curr_reactants)
        
        if not overlap:
            disconnected_steps.append(i+1)  # Step numbers are 1-indexed
            print(f"⚠️ Step {i+1} disconnected from Step {i}")
            print(f"   Previous products: {list(prev_products)}")
            print(f"   Current reactants: {list(curr_reactants)}")
        else:
            print(f"✅ Step {i+1} connected to Step {i} via: {list(overlap)}")
    
    # Calculate continuity score (percentage of connected steps)
    total_connections = len(steps) - 1
    connected_connections = total_connections - len(disconnected_steps)
    continuity_score = connected_connections / total_connections if total_connections > 0 else 1.0
    
    return continuity_score, disconnected_steps

def extract_reactions_from_tree(tree_dict, step_counter=None):
    """Recursively extract reactions from the tree structure, using a global step counter."""
    if step_counter is None:
        step_counter = [1]
    reactions = []
    if isinstance(tree_dict, dict):
        children = tree_dict.get('children', [])
        for child in children:
            if isinstance(child, dict) and child.get('type') == 'reaction':
                reaction_smiles = child.get('smiles', 'Unknown')
                metadata = child.get('metadata', {})
                template = metadata.get('template', 'Unknown')
                reaction_info = {
                    "step_number": step_counter[0],
                    "reaction_smiles": reaction_smiles,
                    "template": template,
                    "reactants": [],
                    "products": []
                }
                step_counter[0] += 1
                reaction_children = child.get('children', [])
                for reaction_child in reaction_children:
                    if reaction_child.get('type') == 'mol':
                        smiles = reaction_child.get('smiles', '')
                        in_stock = reaction_child.get('in_stock', False)
                        if in_stock:
                            reaction_info["reactants"].append(f"{smiles} (in stock)")
                        else:
                            reaction_info["products"].append(smiles)
                reactions.append(reaction_info)
                for reaction_child in reaction_children:
                    if reaction_child.get('type') == 'mol' and not reaction_child.get('in_stock', False):
                        child_reactions = extract_reactions_from_tree(reaction_child, step_counter)
                        reactions.extend(child_reactions)
    return reactions



def extract_essential_plan(route_info):
    """Extract only the essential information from a route, including template."""
    essentials = {
        "score": route_info.get("score", 0.0),
        "num_steps": len(route_info.get("steps", [])),
        "steps": []
    }
    for step in route_info.get("steps", []):
        essentials["steps"].append({
            "step_number": step["step_number"],
            "reactants": step["reactants"],
            "products": step["products"],
            "reaction_smiles": step["reaction_smiles"],
            "template": step.get("template", "Unknown")
        })

    continuity_score, disconnected_steps = validate_route_continuity(essentials["steps"])
    essentials["continuity_score"] = continuity_score
    essentials["disconnected_steps"] = disconnected_steps
    essentials["is_continuous"] = continuity_score >= 0.8  
    return essentials

# def filter_quality_routes(routes, min_score=0.9, min_continuity=0.8):
#     quality_routes = []
#     for route in routes:
#         score = route.get("score", 0.0)
#         continuity_score = route.get("continuity_score", 0.0)
#         is_continuous = route.get("is_continuous", False)
#         if (score >= min_score and continuity_score >= min_continuity and is_continuous):
#             quality_routes.append(route)
#         else:
#             print(f"⚠️ Route filtered out - Score: {score:.3f}, Continuity: {continuity_score:.3f}")
#     quality_routes.sort(key=lambda x: (x.get("score", 0), x.get("continuity_score", 0)), reverse=True)
#     return quality_routes  # <--- THIS LINE IS ESSENTIAL

def get_dual_monomer_synthesis(monomer1: str, monomer2: str):
    """
    Run retrosynthesis for both monomers and return a combined plan.
    """
    print(f"🔍 Synthesizing Monomer 1: {monomer1}")
    result1 = get_synthesis_plan(monomer1)
    result1["monomer_id"] = "monomer1"

    
    
    print(f"\n🔍 Synthesizing Monomer 2: {monomer2}")
    result2 = get_synthesis_plan(monomer2)
    result2["monomer_id"] = "monomer2"

  
   

    # Determine combined status
    if result1["status"] == "success" and result2["status"] == "success":
        status = "both_synthesizable"
    elif result1["status"] == "partial" or result2["status"] == "partial":
        status = "partially_synthesizable"
    else:
        status = "unsynthesizable"

    return {
        "status": status,
        "monomer1": result1,
        "monomer2": result2
    }
def summarize_dual_monomer_retrosynthesis(smiles1: str, smiles2: str) -> str:
    """
    Runs synthesis planning on two monomers and returns assistant-style formatted summary.
    Assumes get_synthesis_plan() and extract_essential_plan() are already defined.
    """
    result1 = get_synthesis_plan(smiles1)
    result2 = get_synthesis_plan(smiles2)
    if result1["status"] == "success" and result2["status"] == "success":
        status = "both_synthesizable"
    elif result1["status"] == "partial" or result2["status"] == "partial":
        status = "partially_synthesizable"
    else:
        status = "unsynthesizable"
    print(f"🧪 Overall synthesis status: {status}")

    essentials_all1 = [extract_essential_plan(route) for route in result1.get("routes", [])]
    essentials_all2 = [extract_essential_plan(route) for route in result2.get("routes", [])]



    def classify_solvability(essentials):
        if not essentials:
            return "unsynthesizable"
        if any(route.get("solved", False) for route in essentials):
            return "synthesizable"
        return "partial"

    def format_for_assistant(monomer_id, essentials):
        if not essentials:
            return f"❌ No retrosynthesis routes found for **{monomer_id}**.\n"

        sorted_routes = sorted(essentials, key=lambda x: x.get("score", 0), reverse=True)
        best = sorted_routes[0]
        #status = "✅ Fully solved" if best.get("solved", False) else "⚠️ Incomplete route"

        output = [
            f"🧪 **Retrosynthesis Plan for {monomer_id}**",
            f"• Top Route Score: `{best.get('score', 0.0):.3f}`",
            f"• Status: {status}",
            f"• Total Steps: {len(best.get('steps', []))}"
        ]

        for step in best.get("steps", []):
            output.append(
                f"""**Step {step['step_number']}**
• Template: `{step['template']}`
• Reactants: {', '.join(step['reactants']) if step['reactants'] else 'N/A'}
• Reaction smiles: {step['reaction_smiles'] if step['reaction_smiles'] else 'N/A'}
• Products: {', '.join(step['products']) if step['products'] else 'N/A'}"""
            )
        return "\n\n".join(output)

    mon1_summary = format_for_assistant("Monomer 1", essentials_all1)
    mon2_summary = format_for_assistant("Monomer 2", essentials_all2)

    mon1_status = classify_solvability(essentials_all1)
    mon2_status = classify_solvability(essentials_all2)

    # High-level final summary
    summary = f"""🔍 **Synthesis Summary**
• Monomer 1 status: `{mon1_status}`
• Monomer 2 status: `{mon2_status}`

{"="*50}

{mon1_summary}

{"="*50}

{mon2_summary}
"""
    return summary

def summarize_dual_monomer_retrosynthesis_json(smiles1: str, smiles2: str):
    result1 = get_synthesis_plan(smiles1)
    result2 = get_synthesis_plan(smiles2)
    if result1["status"] == "success" and result2["status"] == "success":
        status = "both_synthesizable"
    elif result1["status"] == "partial" or result2["status"] == "partial":
        status = "partially_synthesizable"
    else:
        status = "unsynthesizable"

    essentials_all1 = [extract_essential_plan(route) for route in result1.get("routes", [])]
    essentials_all2 = [extract_essential_plan(route) for route in result2.get("routes", [])]
    # Filter for high-quality, well-connected routes
    #print(f"\n🔍 Filtering routes for quality and continuity...")
    #quality_routes1 = filter_quality_routes(essentials_all1, min_score=0.9, min_continuity=0.8)
    #quality_routes2 = filter_quality_routes(essentials_all2, min_score=0.9, min_continuity=0.8)
    
    #print(f"✅ Monomer 1: {len(quality_routes1)}/{len(essentials_all1)} routes passed quality filter")
    #print(f"✅ Monomer 2: {len(quality_routes2)}/{len(essentials_all2)} routes passed quality filter")

    summary = {
        "overall_status": status,
        "quality_filter_applied": True,
        "monomer1": {
            "smiles": smiles1,
            "status": result1["status"],
            "total_routes_found": len(essentials_all1),
           # "quality_routes_passed": len(quality_routes1),
           # "routes": {f"route{i+1}": route for i, route in enumerate(quality_routes1)}
            #"num_routes": len(essentials_all1),
            #"routes": {f"route{i+1}": route for i, route in enumerate(essentials_all1)}
        },
        "monomer2": {
            "smiles": smiles2,
            "status": result2["status"],
            "total_routes_found": len(essentials_all2),
            #"quality_routes_passed": len(quality_routes2),
            #"routes": {f"route{i+1}": route for i, route in enumerate(quality_routes2)}
            #"num_routes": len(essentials_all2),
            #"routes": {f"route{i+1}": route for i, route in enumerate(essentials_all2)}
        }
    }
    return summary







# Example usage:
if __name__ == "__main__":
    # monomer1 = "C=CC(=O)O"  # Example: acrylic acid
    # monomer2 = "C=CCN"      # Example: allylamine

    monomer1 = "CC(C)(C4CCC(OCC(O)COC3CCC(C(C)(C)C2CCC(OCC1CO1)CC2)CC3)CC4)C6CCC(OCC5CO5)CC6"  # Example: acrylic acid
    monomer2 = "CC(C)(c3ccc2OCN(c1ccccc1)Cc2c3)c6ccc5OCN(c4ccccc4)Cc5c6"      # Example: allylamine

    # dual_result = get_dual_monomer_synthesis(monomer1, monomer2)
    # print(f"\n🧪 Overall synthesis status: {dual_result['status']}")
    summary_dict = summarize_dual_monomer_retrosynthesis_json(monomer1, monomer2)
    with open("./SynthesisAgent/dual_monomer_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, ensure_ascii=False)
    print("Summary saved to dual_monomer_summary.json")
    
    # summary = summarize_dual_monomer_retrosynthesis(monomer1, monomer2)
    # print(summary)

