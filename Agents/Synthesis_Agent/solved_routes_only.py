from aizynthfinder.aizynthfinder import AiZynthFinder
import os

def extract_steps_from_tree(tree_dict, step_counter=None):
    """Recursively extract steps from the reaction tree."""
    if step_counter is None:
        step_counter = [1]
    steps = []
    if isinstance(tree_dict, dict):
        children = tree_dict.get('children', [])
        for child in children:
            if child.get('type') == 'reaction':
                step = {
                    "step_number": step_counter[0],
                    "reaction_smiles": child.get('smiles', 'Unknown'),
                    "template": child.get('metadata', {}).get('template', 'Unknown'),
                    "reactants": [],
                    "products": []
                }
                step_counter[0] += 1
                for mol in child.get('children', []):
                    if mol.get('type') == 'mol':
                        smiles = mol.get('smiles', '')
                        if mol.get('in_stock', False):
                            step["reactants"].append(f"{smiles} (in stock)")
                        else:
                            step["products"].append(smiles)
                steps.append(step)
                # Recursively extract further steps
                for mol in child.get('children', []):
                    if mol.get('type') == 'mol' and not mol.get('in_stock', False):
                        steps.extend(extract_steps_from_tree(mol, step_counter))
    return steps

def enforce_stepwise_connectivity(steps):
    """
    Ensure every step's reactants include the main product from the previous step (except the first step).
    If missing, add it as a reactant.
    """
    if not steps or len(steps) <= 1:
        return steps
    connected_steps = []
    previous_main_product = None
    for i, step in enumerate(steps):
        new_step = step.copy()
        if i > 0 and previous_main_product:
            reactants_clean = [r.split(" ")[0] for r in new_step['reactants']]
            if previous_main_product not in reactants_clean:
                # Add previous main product as a reactant
                new_step['reactants'].append(f"{previous_main_product} (from previous step)")
        # Set main product for next step
        if new_step['products']:
            previous_main_product = new_step['products'][0].split(" ")[0]
        else:
            previous_main_product = None
        connected_steps.append(new_step)
    return connected_steps

def get_only_solved_routes(smiles):
    """Return only fully solved, stepwise-connected routes and their steps for a given SMILES."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "model")
    original_dir = os.getcwd()
    try:
        os.chdir(model_dir)
        finder = AiZynthFinder("config.yml")
        finder.stock.select('zinc')
        finder.expansion_policy.select("full")
        #finder.filter_policy.select('default')
        finder.target_smiles = smiles
        finder.config.search.max_iterations = 100
        finder.config.search.max_depth = 6
        finder.tree_search()
        finder.build_routes()
        routes = finder.routes
    finally:
        os.chdir(original_dir)

    solved_routes = []
    for route in routes:
        if isinstance(route, dict) and route.get('route_metadata', {}).get('is_solved', False):
            reaction_tree = route.get('reaction_tree', {})
            if hasattr(reaction_tree, 'to_dict'):
                tree_dict = reaction_tree.to_dict()
            elif isinstance(reaction_tree, dict):
                tree_dict = reaction_tree
            else:
                continue
            steps = extract_steps_from_tree(tree_dict)
            steps = enforce_stepwise_connectivity(steps)
            solved_routes.append({
                "score": route.get('score', 0.0),
                "steps": steps
            })
    return solved_routes

# --- RDKit + Graphviz Drawing ---
from rdkit import Chem
from rdkit.Chem import Draw
from graphviz import Digraph
from PIL import Image

def smiles_to_image(smiles, img_dir, img_size=(200, 100)):
    """Generate a PNG image for a SMILES string using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Clean filename for filesystem
    safe_smiles = (
        smiles.replace('/', '_').replace('=', '_').replace('#', '_')
        .replace('(', '').replace(')', '').replace('[', '').replace(']', '')
        .replace('@', '').replace('+', '').replace('-', '').replace('.', '_')
    )
    img_path = os.path.join(img_dir, f"{safe_smiles}.png")
    if not os.path.exists(img_path):
        img = Draw.MolToImage(mol, size=img_size)
        img.save(img_path)
    return img_path

def draw_stepwise_synthesis_path(steps, output_file="stepwise_synthesis_path"):
    img_dir = "mol_images"
    os.makedirs(img_dir, exist_ok=True)
    dot = Digraph(comment="Stepwise Synthesis Path", format="png")
    dot.attr(rankdir='LR')

    molecule_nodes = {}
    previous_main_product = None

    for i, step in enumerate(steps):
        reaction_node = f"rxn_{i+1}"
        dot.node(reaction_node, f"Step {i+1}", shape="circle", style="filled", color="lightblue")

        # Main product for this step (assume first in list)
        main_product = step["products"][0] if step["products"] else None
        if main_product:
            main_product_smiles = main_product.split(" ")[0]
            if main_product_smiles not in molecule_nodes:
                img_path = smiles_to_image(main_product_smiles, img_dir)
                if img_path:
                    dot.node(main_product_smiles, image=img_path, label="", shape="box", width="0.8", height="0.5", imagescale="true", color="orange")
                else:
                    dot.node(main_product_smiles, main_product_smiles, shape="box", color="red")
                molecule_nodes[main_product_smiles] = True

        # Connect previous main product to this reaction node
        if previous_main_product:
            dot.edge(previous_main_product, reaction_node)

        # Add in-stock reactants (not the main product from previous step)
        for reactant in step["reactants"]:
            reactant_smiles = reactant.split(" ")[0]
            if previous_main_product and reactant_smiles == previous_main_product:
                continue  # Already connected
            if reactant_smiles not in molecule_nodes:
                img_path = smiles_to_image(reactant_smiles, img_dir)
                if img_path:
                    dot.node(reactant_smiles, image=img_path, label="", shape="box", width="0.8", height="0.5", imagescale="true", color="green")
                else:
                    dot.node(reactant_smiles, reactant_smiles, shape="box", color="red")
                molecule_nodes[reactant_smiles] = True
            dot.edge(reactant_smiles, reaction_node)

        # Connect reaction node to main product
        if main_product:
            dot.edge(reaction_node, main_product_smiles)
            previous_main_product = main_product_smiles

    dot.render(output_file, view=True)
    print(f"Stepwise synthesis path with RDKit images saved to {output_file}.png")




if __name__ == "__main__":

    smiles = "CC(C)(C4CCC(OCC(O)COC3CCC(C(C)(C)C2CCC(OCC1CO1)CC2)CC3)CC4)C6CCC(OCC5CO5)CC6"
    solved_routes = get_only_solved_routes(smiles)
    if not solved_routes:
        print("No fully solved routes found.")
    else:
        for idx, route in enumerate(solved_routes, 1):
            print(f"\nRoute {idx} (Score: {route['score']}):")
            for step in route["steps"]:
                print(f"  Step {step['step_number']}:")
                print(f"    Reactants: {step['reactants']}")
                print(f"    Products: {step['products']}")
                print(f"    Template: {step['template']}")
                print(f"    Reaction SMILES: {step['reaction_smiles']}")
        # Draw the first solved route as a stepwise path
        print("\nDrawing stepwise synthesis path for Route 1...")
        draw_stepwise_synthesis_path(solved_routes[0]["steps"])

        # --- Draw the full branched reaction tree using AiZynthFinder's ReactionTree class ---
      
    
    
    # smiles = "CC(C)(C4CCC(OCC(O)COC3CCC(C(C)(C)C2CCC(OCC1CO1)CC2)CC3)CC4)C6CCC(OCC5CO5)CC6"
    # solved_routes = get_only_solved_routes(smiles)
    # if not solved_routes:
    #     print("No fully solved routes found.")
    # else:
    #     for idx, route in enumerate(solved_routes, 1):
    #         print(f"\nRoute {idx} (Score: {route['score']}):")
    #         for step in route["steps"]:
    #             print(f"  Step {step['step_number']}:")
    #             print(f"    Reactants: {step['reactants']}")
    #             print(f"    Products: {step['products']}")
    #             print(f"    Template: {step['template']}")
    #             print(f"    Reaction SMILES: {step['reaction_smiles']}")
    #     # Draw the first solved route as a reaction tree with images
    #     print("\nDrawing reaction tree for Route 1...")
    #     #draw_stepwise_synthesis_path(solved_routes[0]["steps"]) 


    
