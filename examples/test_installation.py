"""
Test if LiteCallLLM is installed correctly.

This script imports key components from the library to verify that
the installation is working properly.
"""

def test_imports():
    try:
        # Import key components
        from litecallllm import structured_completion, astructured_completion, Tool
        from litecallllm import UnifiedResponse, StructuredValidationError
        
        # If we get here, all imports worked
        print("✅ LiteCallLLM is installed correctly!")
        print("Available components:")
        print("  - structured_completion")
        print("  - astructured_completion")
        print("  - Tool")
        print("  - UnifiedResponse")
        print("  - StructuredValidationError")
        
        # Print installation instructions
        print("\nTo use LiteCallLLM in your projects, add it to your requirements.txt:")
        print("git+https://github.com/AmirDadi/LiteCallLLM.git")
        print("\nOr install with pip:")
        print("pip install git+https://github.com/AmirDadi/LiteCallLLM.git")
        
        return True
    except ImportError as e:
        print(f"❌ Error importing LiteCallLLM: {e}")
        print("\nMake sure you've installed the package with:")
        print("pip install git+https://github.com/AmirDadi/LiteCallLLM.git")
        return False

if __name__ == "__main__":
    test_imports() 