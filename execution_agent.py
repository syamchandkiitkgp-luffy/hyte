import io
import sys as os_sys
import os as os_lib
import subprocess
from state import GraphState
from hypothesis_organizer import HypothesisOrganizer
from observability import trace_node, trace_tool

class ExecutionAgent:
    """
    Execution Environment Agent.
    Safely runs Python code, captures stdout, and organizes results into persistent artifacts.
    Now includes self-healing logic for missing dependencies.
    """
    
    def __init__(self):
        self.organizer = HypothesisOrganizer()
        
    def _install_dependency(self, module_name):
        """Attempts to install a missing module using pip."""
        print(f"  [EXECUTION] 🛠️ Missing module '{module_name}' detected. Attempting to install...")
        try:
            # Note: Some modules have different pip names than import names (e.g. sklearn vs scikit-learn)
            # This is a simple mapper for common ones. Expansion might be needed.
            pkg_map = {
                "sklearn": "scikit-learn",
                "PIL": "Pillow",
            }
            pkg_name = pkg_map.get(module_name, module_name)
            
            subprocess.check_call([os_sys.executable, "-m", "pip", "install", pkg_name])
            print(f"  [EXECUTION] ✅ Successfully installed '{pkg_name}'.")
            return True
        except Exception as install_err:
            print(f"  [EXECUTION] ❌ Failed to install '{module_name}': {install_err}")
            return False

    @trace_node("Execution")
    def run(self, state: GraphState):
        """Executes all KPI Python scripts and saves artifacts."""
        python_code_dict = state.get("python_code", {})
        hypothesis = state["hypothesis"]
        
        # Check if we have per-KPI code (dict) or legacy single code (str)
        if isinstance(python_code_dict, dict) and python_code_dict:
            # Per-KPI execution
            return self._execute_all_kpis(state, python_code_dict, hypothesis)
        elif isinstance(python_code_dict, str):
            # Legacy single-script execution (backward compatibility)
            return self._execute_single_script(state, python_code_dict, hypothesis)
        else:
            return {
                "execution_results": "No Python code to execute",
                "current_step": "execution_failed",
                "messages": [{"role": "assistant", "content": "Error: No Python code found."}]
            }
    
    def _execute_all_kpis(self, state, python_code_dict, hypothesis):
        """Execute all KPI scripts sequentially."""
        kpi_execution_results = {}
        kpi_datasets = {}
        all_outputs = []
        
        folder_path = self.organizer.create_hypothesis_folder(hypothesis)
        

        
        for kpi_name, code in python_code_dict.items():

            
            try:
                result = self.tool_python_execution(kpi_name, code, folder_path)
                kpi_execution_results[kpi_name] = result["output"]
                
                # Track generated CSV
                kpi_filename = kpi_name.lower().replace(' ', '_').replace('/', '_')
                expected_csv = f"master_dataset_{kpi_filename}.csv"
                
                if os_lib.path.exists(os_lib.path.join(folder_path, expected_csv)):
                    kpi_datasets[kpi_name] = expected_csv
                    all_outputs.append(f"✅ **{kpi_name}**: {expected_csv} created")
                else:
                    all_outputs.append(f"⚠️ **{kpi_name}**: Dataset file not found")
                
                all_outputs.append(f"```\n{result['output'][:500]}...\n```\n")
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                kpi_execution_results[kpi_name] = error_msg
                all_outputs.append(f"❌ **{kpi_name}**: {error_msg}")

        
        combined_output = "\n\n".join(all_outputs)
        
        # Save artifacts
        updated_state = state.copy()
        updated_state["kpi_execution_results"] = kpi_execution_results
        updated_state["kpi_datasets"] = kpi_datasets
        updated_state["artifact_path"] = folder_path
        self.organizer.save_artifacts(folder_path, updated_state)
        
        return {
            "kpi_execution_results": kpi_execution_results,
            "kpi_datasets": kpi_datasets,
            "artifact_path": folder_path,
            "current_step": "executed",
            "messages": [{
                "role": "assistant",
                "content": f"## 🚀 Execution Complete\n\n{combined_output}\n\n---\n\nAll artifacts saved to: `{folder_path}`\n\nProceed with merging datasets?"
            }]
        }
    
    @trace_tool("Python Execution")
    def tool_python_execution(self, kpi_name, code, folder_path):
        """Execute a single KPI script and return output."""
        stdout = io.StringIO()
        old_stdout = os_sys.stdout
        old_cwd = os_lib.getcwd()
        
        try:
            os_sys.stdout = stdout
            
            # IMPORTANT: Don't chdir to hypothesis folder - stay in project root
            # so that relative paths like './Data_Dictionary/Datalake/' work correctly.
            # Instead, inject the output directory path into the execution globals.
            
            # Modify the code to prepend output directory to CSV saves
            kpi_filename = kpi_name.lower().replace(' ', '_').replace('/', '_')
            output_csv = f"master_dataset_{kpi_filename}.csv"
            output_path = os_lib.path.join(folder_path, output_csv)
            
            # Replace the save path in the code
            # ESCAPE BACKSLASHES for code injection
            safe_output_path = output_path.replace("\\", "\\\\")
            
            modified_code = code.replace(
                f"to_csv('{output_csv}'",
                f"to_csv('{safe_output_path}'"
            )
            modified_code = modified_code.replace(
                f'to_csv("{output_csv}"',
                f'to_csv("{safe_output_path}"'
            )
            
            exec_globals = {
                "pd": __import__("pandas"),
                "plt": __import__("matplotlib.pyplot"),
                "sns": __import__("seaborn"),
                "np": __import__("numpy"),
                "os": __import__("os"),
                "__OUTPUT_DIR__": folder_path  # Provide output directory if needed
            }
            
            exec(modified_code, exec_globals)
            
            result_output = stdout.getvalue()
            return {"output": result_output, "success": True}
            
        finally:
            os_sys.stdout = old_stdout
    
    def _execute_single_script(self, state, code, hypothesis):
        """Legacy single-script execution (backward compatibility)."""
        folder_path = self.organizer.create_hypothesis_folder(hypothesis)
        
        stdout = io.StringIO()
        old_stdout = os_sys.stdout
        old_cwd = os_lib.getcwd()
        
        max_dep_retries = 3
        dep_retries = 0
        
        while dep_retries <= max_dep_retries:
            try:
                stdout = io.StringIO()
                os_sys.stdout = stdout
                
                exec_globals = {
                    "pd": __import__("pandas"),
                    "plt": __import__("matplotlib.pyplot"),
                    "sns": __import__("seaborn"),
                    "np": __import__("numpy"),
                    "os": __import__("os"),
                    "__OUTPUT_DIR__": folder_path
                }
                
                # Modify code to save to hypothesis folder instead of project root
                modified_code = code.replace(
                    "to_csv('master_dataset.csv'",
                    f"to_csv('{os_lib.path.join(folder_path, 'master_dataset.csv')}'"
                )
                modified_code = modified_code.replace(
                    'to_csv("master_dataset.csv"',
                    f'to_csv("{os_lib.path.join(folder_path, "master_dataset.csv")}"'
                )
                
                if dep_retries == 0:
                    pass
                else:
                    pass

                try:
                    exec(modified_code, exec_globals)
                    
                    output_csv = os_lib.path.join(folder_path, "master_dataset.csv")
                    if os_lib.path.exists(output_csv):
                        pass
                    else:
                        pass

                    break 
                finally:
                    os_sys.stdout = old_stdout

            except ModuleNotFoundError as e:
                if os_lib.getcwd() != old_cwd: os_lib.chdir(old_cwd)
                if os_sys.stdout != old_stdout: os_sys.stdout = old_stdout
                
                module_name = e.name
                if module_name and self._install_dependency(module_name):
                    dep_retries += 1
                    continue
                else:
                    raise e
            except Exception as e:
                raise e

        try:
            results = stdout.getvalue()

            
            updated_state = state.copy()
            updated_state["execution_results"] = results
            updated_state["artifact_path"] = folder_path
            self.organizer.save_artifacts(folder_path, updated_state)
            
            return {
                "execution_results": results,
                "artifact_path": folder_path,
                "current_step": "executed",
                "messages": [{"role": "assistant", "content": f"**Analysis Complete!** 🚀\n\nI've successfully executed the code and processed the datalake. Here are the findings:\n\n---\n{results}\n---\n\nAll artifacts, including the source code and generated plots, have been archived in:\n`{folder_path}`"}]
            }
            
        except Exception as e:
            if os_lib.getcwd() != old_cwd: os_lib.chdir(old_cwd)
            if os_sys.stdout != old_stdout: os_sys.stdout = old_stdout
            
            error_msg = f"Execution Error: {str(e)}"

            
            return {
                "execution_results": error_msg,
                "current_step": "execution_failed",
                "retry_count": state["retry_count"] + 1,
                "messages": [{"role": "assistant", "content": f"I encountered an error during execution:\n`{error_msg}`\n\nI will try to analyze the code and apply a fix (Attempt {state['retry_count'] + 1}/3)."}]
            }
