import sys
import os
import json
import ast
from state import GraphState
from observability import trace_node, trace_tool, log_decision

# Add Data_Dictionary to path
if os.path.join(os.getcwd(), 'Data_Dictionary') not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), 'Data_Dictionary'))
from gemini_client import call_gemini

"""
### Concept: Code Synthesis & Context Injection
The **Code Generation Agent** is a specialized coder. Its primary role is **Code Synthesis**—the process of turning abstract logic (Pseudocode) into executable instructions (Python).

A critical part of this is **Context Injection**. The agent isn't just writing generic Python; it's writing code that "knows" about your specific environment.

#### Key Mechanics:
1. **Metadata Injection**: We pass information about the actual CSV files (table names, column headers) directly into the prompt.
2. **Logic Steering**: We use the validated Pseudocode as a strict guide, ensuring the code follows the agreed-upon plan.
3. **Standardization**: The agent is instructed to use specific "guardrail" code (like `df.columns.str.strip()`) to ensure the resulting script is robust against common data issues.
"""

def clean_code_artifacts(code):
    """
    Remove all markdown and other artifacts from generated code.
    """
    if not code:
        return ""
    
    # Remove markdown code blocks
    lines = code.split('\n')
    cleaned_lines = []
    in_code_block = False
    
    for line in lines:
        stripped = line.strip()
        
        # Skip markdown fence lines
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
        
        # Keep all other lines
        cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    
    # Additional cleanup - remove any stray backticks
    result = result.replace('```python', '').replace('```', '')
    
    return result

class CodeGenerationAgent:
    """Agent responsible for generating SQL and Python code using specialized roles."""
    
    @trace_tool("Brainstorm CodeGen Strategy")
    def tool_brainstorm_codegen_strategy(self, kpi_name, methodology, pseudocode, feedback):
        """Tool to brainstorm implementation details (libs, functions) before coding."""
        prompt = f"""
        You are a Principal Software Engineer. The user wants to discuss the implementation strategy for KPI "{kpi_name}".
        
        Methodology:
        {methodology}
        
        Pseudocode:
        {pseudocode}
        
        User Feedback/Context:
        "{feedback}"
        
        Task:
        1. Analyze the feedback/requirements.
        2. Propose a technical approach (libraries, key functions, performance considerations).
        3. Do NOT write full code yet, just the strategy.
        
        Return a concise strategy summary.
        """
        return call_gemini(prompt)

    @trace_tool("Generate Code")
    def tool_generate_code(self, kpi_name, methodology, metadata, pseudocode):
        """Tool to generate Python code for a KPI."""
        return self.generate_python_for_kpi(kpi_name, methodology, metadata, pseudocode)

    @trace_tool("Refine Code")
    def tool_refine_code(self, original_code, error_message, methodology, pseudocode):
        """Tool to refine existing Python code."""
        return self.refine_python(original_code, error_message, methodology, pseudocode)

    @trace_tool("CodeGen Router")
    def _codegen_router(self, state_context):
        """
        Internal Router to decide CodeGen Agent actions.
        Decides between: "brainstorm_strategy", "generate_code", "refine_code".
        """
        prompt = f"""
        You are the CodeGen Router for the HyTE Deep Agent.
        Your goal is to produce robust, executable Python code.
        
        Current Context:
        {json.dumps(state_context, indent=2)}
        
        Available Tools:
        - "brainstorm_strategy": Use if 'feedback' suggests a change in implementation approach OR if requirements are complex.
        - "generate_code": Use if we have valid pseudocode but NO Python code yet.
        - "refine_code": Use if we HAVE Python code but it failed execution or received negative feedback.
        
        Logic:
        1. If 'feedback' is present and suggests conceptual changes -> "brainstorm_strategy".
        2. If 'feedback' is present and suggests bug fixes -> "refine_code".
        3. If 'has_code' is False -> "generate_code".
        4. If 'has_code' is True and 'feedback' is empty -> "refine_code" (self-correction/optimization) or END.
           (Assume called because action is needed).
        
        Return ONLY a JSON object:
        {{
            "thought_process": "Reasoning...",
            "action": "call_tool",
            "tool": "tool_name",
            "status_message": "User facing update."
        }}
        """
        try:
            res = call_gemini(prompt)
            return json.loads(res.replace("```json", "").replace("```", "").strip())
        except:
             # Fallback
            if state_context.get("has_code"):
                return {"action": "call_tool", "tool": "refine_code", "status_message": "Refining code..."}
            else:
                return {"action": "call_tool", "tool": "generate_code", "status_message": "Generating code..."}

    @trace_tool("Identify KPIs to Refine Code")
    def _identify_kpis_to_refine(self, feedback, kpi_list, existing_code_dict):
        """Uses LLM to identify which specific KPIs the user wants code refinement for."""
        kpi_names = list(existing_code_dict.keys()) if existing_code_dict else kpi_list
        prompt = f"""
        The user provided feedback about generated Python code for KPIs. Identify which specific KPIs they want to refine.
        
        Available KPIs: {json.dumps(kpi_names)}
        User Feedback: "{feedback}"
        
        Rules:
        - If the feedback mentions specific KPI names or numbers, return only those.
        - If the feedback is generic (e.g., "refine all", "fix bugs"), return ALL KPI names.
        - Match KPI names case-insensitively and handle partial matches.
        
        Return ONLY a JSON list of KPI names to refine:
        ["KPI Name 1", "KPI Name 2"]
        """
        try:
            res = call_gemini(prompt).strip()
            res = res.replace("```json", "").replace("```", "").strip()
            return json.loads(res)
        except:
            return kpi_names  # Fallback: refine all

    @trace_node("CodeGen")
    def run(self, state: GraphState):
        """Batch-processes ALL KPIs for code generation/refinement."""
        methodology = state["methodology"]
        metadata = state["metadata_context"]
        pseudocode_dict = state.get("pseudocode", {})
        internal_memory = state.get("internal_memory", {})
        kpi_list = state.get("kpi_list", [])
        current_step = state.get("current_step", "")
        
        if not isinstance(pseudocode_dict, dict): pseudocode_dict = {}
            
        existing_code_dict = state.get("python_code", {})
        if not isinstance(existing_code_dict, dict): existing_code_dict = {}

        if not pseudocode_dict:
            return {"python_code": {}, "current_step": "code_generated", 
                    "messages": [{"role": "assistant", "content": "Error: No pseudocode available for any KPI."}]}

        step_outputs = []
        
        # ── GENERATE MODE: Process ALL KPIs ──
        if current_step == "trigger_codegen":
            all_messages = []
            
            for kpi_name, kpi_pseudocode in pseudocode_dict.items():
                step_outputs.append({"step": "codegen", "kpi": kpi_name, "status": "⏳", "output": "Generating code..."})
                
                if not kpi_pseudocode:
                    all_messages.append(f"⚠️ Skipped **{kpi_name}**: No pseudocode available")
                    step_outputs[-1] = {"step": "codegen", "kpi": kpi_name, "status": "⚠️", "output": "No pseudocode"}
                    continue
                
                try:
                    python_code = self.tool_generate_code(kpi_name, methodology, metadata, kpi_pseudocode)
                    
                    # Auto-Verify (Logical Debug)
                    reflection = self._logical_debug(python_code, kpi_pseudocode, metadata)
                    if "REDO" in reflection.upper():
                        python_code = self.tool_refine_code(python_code, reflection, methodology, kpi_pseudocode)
                        status_msg = f"✅ Generated & Verified Code for **{kpi_name}**"
                    else:
                        status_msg = f"✅ Generated Code for **{kpi_name}**"
                    
                    existing_code_dict[kpi_name] = python_code
                    all_messages.append(status_msg)
                    step_outputs[-1] = {"step": "codegen", "kpi": kpi_name, "status": "✅",
                                       "output": python_code[:200] + "..." if len(python_code) > 200 else python_code}
                    
                except Exception as e:
                    error_msg = f"❌ Failed to generate code for **{kpi_name}**: {str(e)}"
                    all_messages.append(error_msg)
                    step_outputs[-1] = {"step": "codegen", "kpi": kpi_name, "status": "❌", "output": str(e)}
            
            # Build consolidated message
            summary = f"## 💻 Code Generated for All KPIs ({len(existing_code_dict)}/{len(pseudocode_dict)})\n\n"
            summary += "\n".join(f"- {msg}" for msg in all_messages)
            summary += "\n\n---\n\nReview the generated code. Type **approve** to execute all scripts, or provide feedback to refine specific KPIs."
            
            return {
                "python_code": existing_code_dict,
                "current_step": "code_generated",
                "step_outputs": step_outputs,
                "messages": [{"role": "assistant", "content": summary}]
            }
        
        # ── REFINE MODE: Selectively update specific KPIs ──
        elif current_step == "refine_codegen":
            feedback = state.get("latest_feedback", "")
            
            # Identify which KPIs to refine
            kpis_to_refine = self._identify_kpis_to_refine(feedback, kpi_list, existing_code_dict)
            
            log_decision("CodeGen Refine", "selective_refine", {
                "kpis_to_refine": kpis_to_refine,
                "feedback": feedback
            })
            
            all_messages = []
            for kpi_name in kpis_to_refine:
                step_outputs.append({"step": "codegen_refine", "kpi": kpi_name, "status": "⏳", "output": "Refining code..."})
                
                existing_code = existing_code_dict.get(kpi_name, "")
                kpi_pseudocode = pseudocode_dict.get(kpi_name, "")
                
                # Check if we should brainstorm first
                context = {
                    "kpi": kpi_name,
                    "has_code": bool(existing_code),
                    "feedback": feedback
                }
                decision = self._codegen_router(context)
                tool = decision.get("tool", "refine_code")
                
                if tool == "brainstorm_strategy":
                    strategy = self.tool_brainstorm_codegen_strategy(kpi_name, methodology, kpi_pseudocode, feedback)
                    internal_memory[f"codegen_strategy_{kpi_name}"] = strategy
                    enhanced_methodology = methodology + f"\n\nImplementation Strategy for {kpi_name}: {strategy}"
                    python_code = self.tool_generate_code(kpi_name, enhanced_methodology, metadata, kpi_pseudocode)
                    msg = f"✅ Brainstormed & Regenerated Code for **{kpi_name}**"
                elif existing_code:
                    python_code = self.tool_refine_code(existing_code, feedback, methodology, kpi_pseudocode)
                    msg = f"✅ Refined Code for **{kpi_name}**"
                else:
                    python_code = self.tool_generate_code(kpi_name, methodology, metadata, kpi_pseudocode)
                    msg = f"✅ Generated Code for **{kpi_name}** (was missing)"
                
                existing_code_dict[kpi_name] = python_code
                all_messages.append(msg)
                step_outputs[-1] = {"step": "codegen_refine", "kpi": kpi_name, "status": "✅",
                                   "output": python_code[:200] + "..." if len(python_code) > 200 else python_code}
            
            summary = f"## 🔄 Code Refined ({len(kpis_to_refine)} KPIs updated)\n\n"
            summary += "\n".join(f"- {msg}" for msg in all_messages)
            summary += "\n\n---\n\nType **approve** to execute all scripts, or provide more feedback."
            
            return {
                "python_code": existing_code_dict,
                "internal_memory": internal_memory,
                "current_step": "code_generated",
                "step_outputs": step_outputs,
                "messages": [{"role": "assistant", "content": summary}]
            }
        
        # ── FALLBACK: Single KPI (backward compat) ──
        else:
            current_kpi = state.get("current_kpi", "")
            if not current_kpi:
                return {"python_code": existing_code_dict, "current_step": "code_generated",
                        "messages": [{"role": "assistant", "content": "Error: No KPI specified."}]}
            
            kpi_pseudocode = pseudocode_dict.get(current_kpi, "")
            if not kpi_pseudocode:
                return {"python_code": existing_code_dict, "current_step": "code_generated",
                        "messages": [{"role": "assistant", "content": f"Error: No pseudocode for {current_kpi}."}]}
            
            python_code = self.tool_generate_code(current_kpi, methodology, metadata, kpi_pseudocode)
            reflection = self._logical_debug(python_code, kpi_pseudocode, metadata)
            if "REDO" in reflection.upper():
                python_code = self.tool_refine_code(python_code, reflection, methodology, kpi_pseudocode)
            
            existing_code_dict[current_kpi] = python_code
            return {
                "python_code": existing_code_dict,
                "current_step": "code_generated",
                "step_outputs": [{"step": "codegen", "kpi": current_kpi, "status": "✅",
                                 "output": python_code[:200] + "..."}],
                "messages": [{"role": "assistant", "content": f"✅ Generated Code for **{current_kpi}**."}]
            }

    @trace_tool("Logical Debug Reflection")
    def _logical_debug(self, python_code, pseudocode, metadata):
        """Reflection step: Checks for obvious logical or syntax errors."""
        prompt = f"""
        Review this Python code against the Approved Pseudocode and Metadata.
        
        Python Code:
        {python_code}
        
        Pseudocode:
        {pseudocode}
        
        Metadata:
        {metadata}
        
        Task:
        1. Check for Python syntax errors.
        2. Check if all table/column names match the Metadata EXACTLY.
        3. Check if all steps in the Pseudocode are implemented.
        
        If perfect, return "APPROVED".
        If issues found, return "REDO: [Detailed explanation of bugs or mismatches]".
        """
        return call_gemini(prompt)

    # Remove the following methods as they are no longer used in the unified agent flow:
    # - generate_sql
    # - generate_python
    # - refine_python
    # - generate_sql_code
    # - generate_python_code
    # - refine_python_code
