import time
import sys
import os


# Add the parent directory to the path so we can import from Predictor_Agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Predictor_Agent.TgEr.predict import predict_property
from Generator_Agent.Util import *

def optimize_structure(target_Tg, target_Er, tolerance_Tg, tolerance_Er, monomer1, monomer2,max_iterations, property_type):
    print("------------TSMP Optimization Started--------------------")
    print(f"Target: Tg = {target_Tg}±{tolerance_Tg}°C, Er = {target_Er}±{tolerance_Er} MPa")
    print(f"SMILES: {monomer1} + {monomer2}")
    print(f"Max iterations: {max_iterations}")
    current_ratio_1 = 0.5 
    current_ratio_2 = 0.5
    final_result=""
    current_monomer1 = monomer1
    current_monomer2 = monomer2
 

    
    optimization_log = []
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
        
        
        # Step 2: Predict properties
        try:
            properties = predict_property(current_monomer1, current_monomer2, current_ratio_1, current_ratio_2)
            print(properties)
            print("✓ Properties predicted")
        except Exception as e:
            return f"Error predicting properties in iteration {iteration + 1}: {str(e)}"
        
        # Step 3: Extract predicted values (simplified parsing)
        # In real implementation, you would parse the properties string to extract Tg and Er
        predicted_Tg = properties.get("tg_score")
        predicted_Er = properties.get("er_score")
        
        # Handle None values from failed predictions
        if predicted_Tg is None or predicted_Er is None:
            print(f"⚠ Property prediction failed - invalid SMILES detected")
            print(f"Current monomers: {current_monomer1} + {current_monomer2}")
            print("Skipping this iteration and reverting to previous valid structure...")
            continue
        
        print(f"Predicted: Tg = {predicted_Tg}°C, Er = {predicted_Er} MPa")
        
        # Step 4: Check if within tolerance
        tg_within_tolerance = abs(predicted_Tg - target_Tg) <= tolerance_Tg
        er_within_tolerance = abs(predicted_Er - target_Er) <= tolerance_Er
        
        optimization_log.append({
            "iteration": iteration + 1,
            "predicted_Tg": predicted_Tg,
            "predicted_Er": predicted_Er,
            "tg_within_tolerance": tg_within_tolerance,
            "er_within_tolerance": er_within_tolerance
        })
        
        if tg_within_tolerance and er_within_tolerance:
            print("✓ Optimization successful! Properties within tolerance.")
            break
        else:
            print("⚠ Properties outside tolerance, continuing optimization...")
            
            # Determine optimization strategy based on current vs target values
            optimization_strategy = determine_optimization_strategy(
                predicted_Tg, target_Tg, predicted_Er, target_Er, 
                tolerance_Tg, tolerance_Er
            )
            print(f"Optimization strategy: {optimization_strategy['actions']}")
            break
            
           

    return final_result


def determine_optimization_strategy(
    predicted_Tg: float, target_Tg: float,
    predicted_Er: float, target_Er: float,
    tolerance_Tg: float = 5.0, tolerance_Er: float = 5.0
):
    """
    Decide how to edit the TWO monomers to move Tg and Er into tolerance.
    Returns a dict with errors, directions, and prioritized edit actions
    (each action = what to add/remove/replace; includes example SMILES/SMARTS snippets).
    """

    # --- 1) Errors (signed & abs) ---
    tg_error = predicted_Tg - target_Tg      # >0 means Tg too high; <0 means too low
    er_error = predicted_Er - target_Er      # >0 means Er too high; <0 means too low
    tg_off   = abs(tg_error)
    er_off   = abs(er_error)

    # --- 2) Within-tolerance flags ---
    tg_ok = tg_off <= tolerance_Tg
    er_ok = er_off <= tolerance_Er



    # --- 3) Directional needs ---
    # For each property: "decrease", "increase", or "hold"
    if tg_ok:
        tg_dir = "hold"
    else:
        tg_dir = "decrease" if tg_error > 0 else "increase"

    if er_ok:
        er_dir = "hold"
    else:
        er_dir = "decrease" if er_error > 0 else "increase"

    # --- 4) Priority (which property to optimize first) ---
    # Normalize by tolerance so we focus on the bigger relative miss.
    tg_norm = tg_off / max(1e-6, tolerance_Tg)
    er_norm = er_off / max(1e-6, tolerance_Er)
    primary  = "Tg" if tg_norm >= er_norm else "Er"
    secondary = "Er" if primary == "Tg" else "Tg"

    print(f"Primary: {primary}, Secondary: {secondary}")
    print(f"Direction: Tg: {tg_dir}, Er: {er_dir}")

    # --- 5) Action library (chemistry-agnostic edits) ---
    # Use these as building blocks; your family guard should preserve the core reactive handles.
    # Notation: two-ended fragments use dummy atoms [*]...[*]
    # - Rigidifiers raise Tg (often reduce Er if overused)
    # - Softeners lower Tg (often help Er)
    ACTIONS = {
        # Tg INCREASE (rigidify / add polarity)
        "tg_increase": [
            {
                "name": "add_aromatic_ring",
                "effect": {"Tg": "up", "Er": "neutral_or_down"},
                "edit": "aliphatic_ring → aromatic_ring",
                "patterns": {"from": "C1CCCCC1", "to": "c1ccccc1"},
                "notes": "Swap cyclohexyl to phenyl; or insert phenyl island."
            },
            {
                "name": "insert_sulfone_linker",
                "effect": {"Tg": "up_strong", "Er": "neutral_or_down"},
                "edit": "insert_between_C-C",
                "patterns": {"insert": "[*]S(=O)(=O)[*]"},
                "notes": "High polarity/rigidity; use moderately to avoid brittleness."
            },
            {
                "name": "insert_amide_or_urethane",
                "effect": {"Tg": "up", "Er": "up"},
                "edit": "insert_between_C-C",
                "patterns": {"amide": "[*]NC(=O)[*]", "urethane": "[*]OC(=O)N[*]"},
                "notes": "Adds H-bonding; often improves Er while raising Tg."
            },
            {
                "name": "shorten_aliphatic_spacer",
                "effect": {"Tg": "up", "Er": "down_if_excess"},
                "edit": "reduce_(CH2)n_by_1",
                "patterns": {"from": "[*]CCC[*]", "to": "[*]CC[*]"},
                "notes": "Decreases mobility and free volume."
            },
        ],

        # Tg DECREASE (soften / add free volume)
        "tg_decrease": [
            {
                "name": "insert_ether_or_thioether",
                "effect": {"Tg": "down", "Er": "up"},
                "edit": "insert_between_C-C",
                "patterns": {"ether": "[*]O[*]", "thioether": "[*]S[*]"},
                "notes": "Boosts segmental mobility; good for Er too."
            },
            {
                "name": "insert_siloxane",
                "effect": {"Tg": "down_strong", "Er": "up"},
                "edit": "insert_between_C-C",
                "patterns": {"siloxane": "[*][Si]O[*]"},
                "notes": "Very flexible; strong Tg drop."
            },
            {
                "name": "lengthen_aliphatic_spacer",
                "effect": {"Tg": "down", "Er": "up_until_set"},
                "edit": "increase_(CH2)n_by_1",
                "patterns": {"from": "[*]CC[*]", "to": "[*]CCC[*]"},
                "notes": "Add one methylene unit in soft segment."
            },
            {
                "name": "add_bulky_pendant",
                "effect": {"Tg": "down", "Er": "neutral_or_up"},
                "edit": "replace_small_alkyl_with_tBu",
                "patterns": {"from": "[CH3]", "to": "C(C)(C)C"},
                "notes": "Increases free volume (internal plasticization)."
            },
        ],

        # Er INCREASE (reversible cohesion + some mobility)
        "er_increase": [
            {
                "name": "add_urethane_or_amide_islands",
                "effect": {"Tg": "up_slight", "Er": "up"},
                "edit": "insert_between_C-C",
                "patterns": {"urethane": "[*]OC(=O)N[*]", "amide": "[*]NC(=O)[*]"},
                "notes": "H-bonding domains improve elastic recovery."
            },
            {
                "name": "sprinkle_ether_linkers",
                "effect": {"Tg": "down_slight", "Er": "up"},
                "edit": "insert_between_C-C",
                "patterns": {"ether": "[*]O[*]"},
                "notes": "A little mobility helps recovery (avoid over-softening)."
            },
            {
                "name": "pi_pi_helper_if_aromatic_present",
                "effect": {"Tg": "up_slight", "Er": "up"},
                "edit": "phenyl → biphenyl",
                "patterns": {"from": "c1ccccc1", "to": "c1ccc(cc1)c2ccccc2"},
                "notes": "Only apply if an aromatic ring already exists."
            },
            
        ],

        # Er DECREASE (rare; to counter over-elasticity or permanent set)
        "er_decrease": [
            {
                "name": "add_mild_rigidifiers",
                "effect": {"Tg": "up", "Er": "down"},
                "edit": "add_aromatic_or_sulfone",
                "patterns": {"aromatic": "c1ccccc1", "sulfone": "[*]S(=O)(=O)[*]"},
                "notes": "Use lightly to avoid brittleness."
            }
        ],
    }


    extra_er_increase = [
    {"name": "insert_carbonate_linker",
     "effect": {"Tg":"up_slight","Er":"up"},
     "edit": "insert_between_C-C",
     "patterns": {"carbonate":"[*]OC(=O)O[*]"},
     "notes":"Mild cohesion without big rigidity."},

    {"name": "add_urea_island",
     "effect": {"Tg":"up_slight","Er":"up"},
     "edit": "insert_between_C-C",
     "patterns": {"urea":"[*]NC(=O)N[*]"},
     "notes":"Stronger H-bonding than amide/urethane."},

    {"name": "sprinkle_thioether_linkers",
     "effect": {"Tg":"down_slight","Er":"up"},
     "edit": "insert_between_C-C",
     "patterns": {"thio":"[*]S[*]"},
     "notes":"Adds mobility with modest Tg drop."},

    {"name":"add_pendant_nitrile",
     "effect":{"Tg":"up_slight","Er":"up"},
     "edit":"sidechain_substitution",
     "patterns":{"from":"[CH3]","to":"CC#N"},
     "notes":"Polar pendant improves cohesion; apply on soft block."}
        ]

    extra_er_decrease = [
    {"name":"replace_ether_with_methylene",
     "effect":{"Tg":"up_slight","Er":"down"},
     "edit":"replace_linker",
     "patterns":{"from":"[*]O[*]","to":"[*]C[*]"},
     "notes":"Reduce soft-segment mobility."},

    {"name":"reduce_hbond_density",
     "effect":{"Tg":"down_slight","Er":"down"},
     "edit":"replace_linker",
     "patterns":{"from":"[*]NC(=O)[*]","to":"[*]O[*]"},
     "notes":"Back off cohesive H-bonds in soft region."}
        ]


    # --- 6) Pick action lists for current need ---
    actions = []
    if tg_dir == "increase":
        actions += ACTIONS["tg_increase"]
    elif tg_dir == "decrease":
        actions += ACTIONS["tg_decrease"]

    if er_dir == "increase":
        actions += ACTIONS["er_increase"]
    elif er_dir == "decrease":
        actions += ACTIONS["er_decrease"]


    #ACTIONS["er_increase"].extend(extra_er_increase)
    #ACTIONS["er_decrease"].extend(extra_er_decrease)

    # --- 7) Prioritize actions (bigger normalized miss first) ---
    # Heuristic: prioritize edits aligned with the primary miss.
    def priority(a):
        # strong effects get a small bonus
        strong = ("up_strong" in a["effect"].values()) or ("down_strong" in a["effect"].values())
        base = 2 if strong else 1
        if primary == "Tg" and ("Tg" in a["effect"]):
            return 100*base
        if primary == "Er" and ("Er" in a["effect"]):
            return 100*base
        return 10*base

    actions_sorted = sorted(actions, key=priority, reverse=True)

    # --- 8) Output plan ---
    return {
        "errors": {
            "Tg_error": tg_error, "Tg_abs": tg_off, "Tg_within_tol": tg_ok,
            "Er_error": er_error, "Er_abs": er_off, "Er_within_tol": er_ok
        },
        "direction": {"Tg": tg_dir, "Er": er_dir},
        "optimize_order": [primary, secondary],
        "actions": actions_sorted,
        "notes": [
            "Apply edits on the backbone/side-chains, not on reactive handles.",
            "Run family/role guards (epoxy, amine, thiol, vinyl, acrylate, hydroxyl) before/after edits.",
            "If Er drops while raising Tg, mix in urethane/amide islands to rebalance.",
            "Avoid extremes: too many rigidifiers → brittleness; too many softeners → permanent set."
        ]
    }

    
    
    
def validate_smiles(smiles):
    """Validate if SMILES string is valid"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False



if __name__ == "__main__":
    #optimize_structure(100, 60, 5, 5, "CCNC1OC1Cc1ccccc1CCCCBr", "CCC2OC2COOCC",  5, "physical")
    A = "CCNC1OC1Cc1ccccc1CCCCBr"
    B = "CCC2OC2COOCCNC1CCCCC1"
    tgt_Tg = 150
    tgt_Er = 140
   
    tol_Tg = 5
    tol_Er = 5
    scores = predict_property(A, B,0.1,0.9)
    pred_Tg = scores["tg_score"]
    pred_Er = scores["er_score"]
    print("--------------------------------")
    print("Original Monomers:", A, B, "| Predicted Tg:", pred_Tg, "| Predicted Er:", pred_Er)
    print("Asking Tg:", tgt_Tg, "| Asking Er:", tgt_Er)
    print("Tolerance Tg:", tol_Tg, "| Tolerance Er:", tol_Er)
    print("--------------------------------")
   
    fam = assign_family(A, B)
    if not fam:
        raise ValueError("Could not detect a supported reactive family for this pair.")
    family, rolesAB = fam["family"], fam["roles"]

    # 2) Build the plan
    plan = determine_optimization_strategy(pred_Tg, tgt_Tg, pred_Er, tgt_Er, tol_Tg, tol_Er)

    # 3) Choose which monomer to prefer mutating (influence probe)
    side_pref = finite_influence_side(A, B, pred_Tg, pred_Er, rolesAB)   # 'A' or 'B'

    # 4) Apply the top 1–2 actions
    best_pair = (A, B)
    best_pred = (pred_Tg, pred_Er)
    best_err  = abs(pred_Tg - tgt_Tg) + abs(pred_Er - tgt_Er)
    n_best_tg = abs(pred_Tg - tgt_Tg)
    n_best_er = abs(pred_Er - tgt_Er)

    tried_keys = {cano(A) + "|" + cano(B)}

    for act in plan["actions"]:
        # generate candidate edits, preserving roles
        cands = apply_action_to_pair(A, B, act, rolesAB)
        if not cands:
            continue

        # bias evaluation order to the preferred side (so we look at edits that changed that side first)
        def side_changed(pair):
            a1, b1 = pair
            return (side_pref == 'A' and a1 != A) or (side_pref == 'B' and b1 != B)
        cands.sort(key=lambda ab: (not side_changed(ab), ))  # preferred first

        for a1, b1 in cands:
            key = a1 + "|" + b1
            if key in tried_keys:
                continue
            tried_keys.add(key)

            scores = predict_property(a1, b1,0.1,0.9)
            Tg_hat = scores["tg_score"]
            Er_hat = scores["er_score"]
            err = abs(Tg_hat - tgt_Tg) + abs(Er_hat - tgt_Er)

            # keep the best improvement so far
            if err < best_err:
                best_pair = (a1, b1)
                best_pred = (Tg_hat, Er_hat)
                best_err  = err

            # early exit if within tolerance for both
            if abs(Tg_hat - tgt_Tg) <= tol_Tg and abs(Er_hat - tgt_Er) <= tol_Er:
                A, B = a1, b1
                pred_Tg, pred_Er = Tg_hat, Er_hat
                break  # stop applying more actions

        # update A,B to the current best after this action
        A, B = best_pair
        pred_Tg, pred_Er = best_pred

    # After applying up to 2 actions:
    final_pair = best_pair
    final_pred = best_pred
    status = ("ok" if (abs(final_pred[0]-tgt_Tg) <= tol_Tg and
                      abs(final_pred[1]-tgt_Er) <= tol_Er)
             else "updated_but_outside_tolerance")

    result = {
        "family": family,
        "roles": rolesAB,
        "pair": {"A": final_pair[0], "B": final_pair[1]},
        "pred": {"Tg": final_pred[0], "Er": final_pred[1]},
        "status": status
    }

    print("--------------After Optimization------------------")
    print("Final Pair:", final_pair, "| Predicted Tg:", final_pred[0], "| Predicted Er:", final_pred[1])
    print("--------------------------------")

    ratio_1 = 0.1
    ratio_2 = 0.9
    best_ratio_1 = 0.1
    best_ratio_2 = 0.9
    best_err1 = n_best_tg
    best_err2 = n_best_er

    for i in range(9):
        A, B = final_pair
        pred_Tg, pred_Er = final_pred
        scores = predict_property(A, B, ratio_1, ratio_2)
        Tg_hat = scores["tg_score"]
        Er_hat = scores["er_score"]
        err1 = abs(Tg_hat - tgt_Tg) 
        err2 = abs(Er_hat - tgt_Er)
      
        if err1 < best_err1 and err2 < best_err2:
            best_pair = (A, B)
            best_pred = (Tg_hat, Er_hat)
            best_err1  = err1
            best_err2  = err2
            best_ratio_1 = ratio_1
            best_ratio_2 = ratio_2
        ratio_1 = round(ratio_1 + 0.1,2)
        ratio_2 = round(ratio_2 - 0.1,2)

    print("--------------After Ratio Optimization------------------")
    print("Final:", best_pair, "| Pred:", best_pred, "| Ratio:", best_ratio_1, best_ratio_2)

    print("--------------------------------")



    