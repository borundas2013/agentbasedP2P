from aizynthfinder.aizynthfinder import AiZynthFinder
import os

def test_aizynthfinder():
    print("Testing AiZynthFinder import...")
    
    # Get the absolute path to the config file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "model")
    config_path = os.path.join(model_dir, "config.yml")
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found at: {config_path}")
        print("You need to download the model data first.")
        print("Please download from: https://github.com/MolecularAI/aizynthfinder-models/releases")
        print("And extract to a 'model' folder in this directory.")
        return False
    
    print(f"✅ Config file found at: {config_path}")
    
    # Check if the ONNX model file exists
    onnx_file = os.path.join(model_dir, "uspto_model.onnx")
    if not os.path.exists(onnx_file):
        print(f"❌ ONNX model file not found at: {onnx_file}")
        return False
    
    print(f"✅ ONNX model file found at: {onnx_file}")
    
    try:
        # Change to the model directory so AiZynthFinder can find the files
        original_dir = os.getcwd()
        os.chdir(model_dir)
        
        # Try to load the app with just the config filename (not full path)
        app = AiZynthFinder("config.yml")
        print("✅ AiZynthFinder loaded successfully!")
        
        # Use the correct policy names from the config
        app.stock.select('zinc')
        app.expansion_policy.select("full")  # Changed from "uspto" to "full"
        # app.filter_policy.select("uspto")  # Commented out - may not be configured
        
        # Test with a simple molecule
        app.target_smiles = "CCO"  # Simple ethanol molecule for testing
        app.tree_search()
        app.build_routes()
        stats = app.extract_statistics()
        print("✅ Retrosynthesis completed successfully!")
        print(f"Statistics: {stats}")
        
        # Change back to original directory
        os.chdir(original_dir)
        return True
    except Exception as e:
        print(f"❌ Error loading AiZynthFinder: {e}")
        # Change back to original directory even if there's an error
        try:
            os.chdir(original_dir)
        except:
            pass
        return False

if __name__ == "__main__":
    test_aizynthfinder() 