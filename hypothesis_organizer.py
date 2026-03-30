import os
import json
from datetime import datetime

class HypothesisOrganizer:
    """Manages storage of artifacts for each hypothesis run."""
    
    def __init__(self, base_dir='hypotheses'):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            
    def create_hypothesis_folder(self, hypothesis):
        """Creates a unique folder for the hypothesis based on its content and timestamp."""
        # Create a short slug from hypothesis
        slug = "_".join(hypothesis.split()[:5]).lower().replace(" ", "_")
        slug = "".join([c for c in slug if c.isalnum() or c == "_"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{slug}"
        path = os.path.join(self.base_dir, folder_name)
        
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def save_artifacts(self, folder_path, state):
        """Saves all state artifacts to the specified folder."""
        # Save Metadata context
        with open(os.path.join(folder_path, 'metadata_context.txt'), 'w', encoding='utf-8') as f:
            f.write(state.get('metadata_context', ''))
            
        # Save Methodology
        with open(os.path.join(folder_path, 'methodology.md'), 'w', encoding='utf-8') as f:
            f.write(state.get('methodology', ''))
            
        # Save Pseudo-code (handle both dict and string formats)
        pseudocode = state.get('pseudocode', '')
        if isinstance(pseudocode, dict):
            # Per-KPI format: save as structured text
            with open(os.path.join(folder_path, 'pseudocode.txt'), 'w', encoding='utf-8') as f:
                if pseudocode:
                    f.write("# Per-KPI Pseudocode\n\n")
                    for kpi_name, kpi_pseudocode in pseudocode.items():
                        f.write(f"## {kpi_name}\n\n{kpi_pseudocode}\n\n{'='*80}\n\n")
                else:
                    f.write('')
        else:
            # Legacy string format
            with open(os.path.join(folder_path, 'pseudocode.txt'), 'w', encoding='utf-8') as f:
                f.write(pseudocode)
            
        # Save Python Code (handle both dict and string formats)
        python_code = state.get('python_code', '')
        if isinstance(python_code, dict):
            # Per-KPI format: save individual files
            if python_code:
                for kpi_name, code in python_code.items():
                    kpi_filename = kpi_name.lower().replace(' ', '_').replace('/', '_')
                    code_path = os.path.join(folder_path, f'code_{kpi_filename}.py')
                    with open(code_path, 'w', encoding='utf-8') as f:
                        f.write(f"# KPI: {kpi_name}\n\n{code}")
                
                # Also save a combined file for reference
                with open(os.path.join(folder_path, 'code.py'), 'w', encoding='utf-8') as f:
                    f.write("# Combined Code for All KPIs\n\n")
                    for kpi_name, code in python_code.items():
                        f.write(f"# {'='*80}\n# KPI: {kpi_name}\n# {'='*80}\n\n{code}\n\n")
        else:
            # Legacy string format
            with open(os.path.join(folder_path, 'code.py'), 'w', encoding='utf-8') as f:
                f.write(python_code)
            
        # Save Execution Results (handle both dict and string formats)
        execution_results = state.get('execution_results', '')
        kpi_execution_results = state.get('kpi_execution_results', {})
        
        with open(os.path.join(folder_path, 'results.log'), 'w', encoding='utf-8') as f:
            if kpi_execution_results:
                # Per-KPI execution results
                f.write("# Per-KPI Execution Results\n\n")
                for kpi_name, result in kpi_execution_results.items():
                    f.write(f"## {kpi_name}\n\n{result}\n\n{'='*80}\n\n")
            elif execution_results:
                # Legacy single result
                f.write(execution_results)
            else:
                f.write('')
            
        # Save full state as JSON
        with open(os.path.join(folder_path, 'state.json'), 'w', encoding='utf-8') as f:
            # We filter out non-serializable objects if any
            clean_state = {k: v for k, v in state.items() if isinstance(v, (str, int, float, bool, list, dict))}
            json.dump(clean_state, f, indent=4)
            
        print(f"Artifacts saved to {folder_path}")

if __name__ == "__main__":
    organizer = HypothesisOrganizer()
    path = organizer.create_hypothesis_folder("Test Hypothesis for Churn")
    organizer.save_artifacts(path, {"hypothesis": "Test", "methodology": "# Test"})
