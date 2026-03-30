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
### Concept: Intermediate Logic Representation (Pseudocode)
When moving from a high-level plan (Methodology) to actual code (Python), directly generating a large script code can lead to errors. 

The **Pseudocode Agent** acts as a bridge. It creates an **Intermediate Representation**—a logical "blueprint"—that describes the data transformations in plain English before a single line of code is written.

#### Why use Pseudocode?
1. **Validation**: We can verify the logic (e.g., checking if the correct tables and columns are used) at a high level.
2. **Reviewability**: Non-technical users can review "Step 1: Load customer data" much easier than a Pandas SQL join.
3. **Consistency**: It gives the CodeGen agent a strict list of instructions, significantly reducing syntax and logic errors in the final script.
"""

class PseudocodeAgent:
    """
    Agent responsible for translating the methodology into technical logic steps.
    Includes a validation loop to ensure all referenced tables and columns exist.
    """
    
    @trace_tool("Brainstorm Strategy")
    def tool_brainstorm_strategy(self, kpi_name, methodology, data_context, feedback):
        """Tool to brainstorm/refine the approach for a KPI before coding."""
        prompt = f"""
        You are a Principal Data Architect. The user wants to refine the strategy for KPI "{kpi_name}".
        
        Methodology:
        {methodology}
        
        Metadata:
        {data_context}
        
        User Feedback/Context:
        "{feedback}"
        
        Task:
        1. Analyze the feedback.
        2. Propose a refined logical approach (text description, not code) for calculating this KPI.
        3. Explain *why* this approach is better.
        
        Return a concise strategy summary (max 3 sentences).
        """
        return call_gemini(prompt)

    @trace_tool("Generate Pseudocode")
    def tool_generate_pseudocode(self, kpi_name, methodology, data_context, granularity_analysis):
        """Tool to generate pseudocode for a KPI."""
        return self.generate_for_kpi(kpi_name, methodology, data_context, granularity_analysis)

    @trace_tool("Refine Pseudocode")
    def tool_refine_pseudocode(self, methodology, previous_code, feedback):
        """Tool to refine existing pseudocode."""
        return self.refine(methodology, previous_code, feedback)

    @trace_tool("Pseudocode Router")
    def _pseudocode_router(self, state_context):
        """
        Internal Router to decide Pseudocode Agent actions.
        Decides between: "brainstorm_strategy", "generate_pseudocode", "refine_pseudocode".
        """
        prompt = f"""
        You are the Pseudocode Router for the HyTE Deep Agent.
        Your goal is to produce high-quality logical blueprints (pseudocode) for KPIs.
        
        Current Context:
        {json.dumps(state_context, indent=2)}
        
        Available Tools:
        - "brainstorm_strategy": Use if 'feedback' suggests a change in approach OR if we need to plan before generating.
        - "generate_pseudocode": Use if we have a clear strategy/hypothesis but NO pseudocode yet.
        - "refine_pseudocode": Use if we HAVE pseudocode but need to fix it (based on feedback or validation).
        
        Logic:
        1. If 'feedback' is generic (e.g., "start"), and 'has_pseudocode' is False -> "generate_pseudocode".
        2. If 'feedback' is specific (e.g., "wrong table", "change logic") -> "brainstorm_strategy" (to discuss) OR "refine_pseudocode" (to fix).
           - Prefer "refine_pseudocode" if it's a direct fix.
           - Prefer "brainstorm_strategy" if it's a conceptual change.
        3. If 'has_pseudocode' is True AND 'feedback' is empty -> "refine_pseudocode" (self-correction) or END (if valid). 
           (For this router, assume we are called because action is needed).
        
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
            if state_context.get("has_pseudocode"):
                return {"action": "call_tool", "tool": "refine_pseudocode", "status_message": "Refining logic..."}
            else:
                return {"action": "call_tool", "tool": "generate_pseudocode", "status_message": "Generating logic..."}

    @trace_tool("Identify KPIs to Refine")
    def _identify_kpis_to_refine(self, feedback, kpi_list, existing_pseudocode_dict):
        """Uses LLM to identify which specific KPIs the user wants to refine based on feedback."""
        kpi_names = list(existing_pseudocode_dict.keys()) if existing_pseudocode_dict else kpi_list
        prompt = f"""
        The user provided feedback about pseudocode for KPIs. Identify which specific KPIs they want to refine.
        
        Available KPIs: {json.dumps(kpi_names)}
        User Feedback: "{feedback}"
        
        Rules:
        - If the feedback mentions specific KPI names or numbers, return only those.
        - If the feedback is generic (e.g., "refine all", "improve"), return ALL KPI names.
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

    @trace_node("Pseudocode")
    def run(self, state: GraphState):
        """Batch-processes ALL KPIs for pseudocode generation/refinement."""
        methodology = state["methodology"]
        data_context = state["metadata_context"]
        kpi_list = state.get("kpi_list", [])
        granularity_analysis = state.get("granularity_analysis", {})
        internal_memory = state.get("internal_memory", {})
        current_step = state.get("current_step", "")
        
        existing_pseudocode_dict = state.get("pseudocode", {})
        if not isinstance(existing_pseudocode_dict, dict): existing_pseudocode_dict = {}
        
        if not kpi_list:
            return {"pseudocode": {}, "current_step": "pseudocode_generated", 
                    "messages": [{"role": "assistant", "content": "Error: No KPIs found in kpi_list."}]}

        step_outputs = []
        
        # ── GENERATE MODE: Process ALL KPIs ──
        if current_step == "trigger_pseudocode":
            all_messages = []
            
            for i, kpi_name in enumerate(kpi_list):
                step_outputs.append({"step": "pseudocode", "kpi": kpi_name, "status": "⏳", "output": "Generating..."})
                
                try:
                    pseudocode = self.tool_generate_pseudocode(kpi_name, methodology, data_context, granularity_analysis)
                    
                    # Auto-Verify via reflection
                    reflection = self._reflect_on_logic(pseudocode, data_context, kpi_name=kpi_name)
                    if "REDO" in reflection.upper():
                        pseudocode = self.tool_refine_pseudocode(methodology, pseudocode, reflection)
                        status_msg = f"✅ Generated & Self-Corrected Logic for **{kpi_name}**"
                    else:
                        status_msg = f"✅ Generated Logic for **{kpi_name}**"
                    
                    existing_pseudocode_dict[kpi_name] = pseudocode
                    all_messages.append(status_msg)
                    # Update step output to show success
                    step_outputs[-1] = {"step": "pseudocode", "kpi": kpi_name, "status": "✅", 
                                       "output": pseudocode[:200] + "..." if len(pseudocode) > 200 else pseudocode}
                    
                except Exception as e:
                    error_msg = f"❌ Failed to generate logic for **{kpi_name}**: {str(e)}"
                    all_messages.append(error_msg)
                    step_outputs[-1] = {"step": "pseudocode", "kpi": kpi_name, "status": "❌", "output": str(e)}
            
            # Build consolidated message
            summary = f"## 📝 Pseudocode Generated for All KPIs ({len(existing_pseudocode_dict)}/{len(kpi_list)})\n\n"
            summary += "\n".join(f"- {msg}" for msg in all_messages)
            summary += "\n\n---\n\nReview the pseudocode for each KPI above. Type **approve** to proceed to code generation, or provide feedback to refine specific KPIs."
            
            return {
                "pseudocode": existing_pseudocode_dict,
                "current_step": "pseudocode_generated",
                "step_outputs": step_outputs,
                "messages": [{"role": "assistant", "content": summary}]
            }
        
        # ── REFINE MODE: Selectively update specific KPIs ──
        elif current_step == "refine_pseudocode":
            feedback = state.get("latest_feedback", "")
            
            # Identify which KPIs to refine
            kpis_to_refine = self._identify_kpis_to_refine(feedback, kpi_list, existing_pseudocode_dict)
            
            log_decision("Pseudocode Refine", "selective_refine", {
                "kpis_to_refine": kpis_to_refine,
                "feedback": feedback
            })
            
            all_messages = []
            for kpi_name in kpis_to_refine:
                step_outputs.append({"step": "pseudocode_refine", "kpi": kpi_name, "status": "⏳", "output": "Refining..."})
                
                existing_code = existing_pseudocode_dict.get(kpi_name, "")
                
                # Check if we should brainstorm first
                context = {
                    "kpi": kpi_name,
                    "has_pseudocode": bool(existing_code),
                    "feedback": feedback
                }
                decision = self._pseudocode_router(context)
                tool = decision.get("tool", "refine_pseudocode")
                
                if tool == "brainstorm_strategy":
                    strategy = self.tool_brainstorm_strategy(kpi_name, methodology, data_context, feedback)
                    internal_memory[f"strategy_{kpi_name}"] = strategy
                    # After brainstorming, regenerate with new strategy
                    enhanced_methodology = methodology + f"\n\nRefined Strategy for {kpi_name}: {strategy}"
                    pseudocode = self.tool_generate_pseudocode(kpi_name, enhanced_methodology, data_context, granularity_analysis)
                    msg = f"✅ Brainstormed & Regenerated Logic for **{kpi_name}**"
                elif existing_code:
                    pseudocode = self.tool_refine_pseudocode(methodology, existing_code, feedback)
                    msg = f"✅ Refined Logic for **{kpi_name}**"
                else:
                    pseudocode = self.tool_generate_pseudocode(kpi_name, methodology, data_context, granularity_analysis)
                    msg = f"✅ Generated Logic for **{kpi_name}** (was missing)"
                
                existing_pseudocode_dict[kpi_name] = pseudocode
                all_messages.append(msg)
                step_outputs[-1] = {"step": "pseudocode_refine", "kpi": kpi_name, "status": "✅",
                                   "output": pseudocode[:200] + "..." if len(pseudocode) > 200 else pseudocode}
            
            summary = f"## 🔄 Pseudocode Refined ({len(kpis_to_refine)} KPIs updated)\n\n"
            summary += "\n".join(f"- {msg}" for msg in all_messages)
            summary += "\n\n---\n\nType **approve** to proceed to code generation, or provide more feedback."
            
            return {
                "pseudocode": existing_pseudocode_dict,
                "internal_memory": internal_memory,
                "current_step": "pseudocode_generated",
                "step_outputs": step_outputs,
                "messages": [{"role": "assistant", "content": summary}]
            }
        
        # ── FALLBACK: Single KPI (backward compat) ──
        else:
            current_kpi = state.get("current_kpi", "")
            if not current_kpi:
                return {"pseudocode": existing_pseudocode_dict, "current_step": "pseudocode_generated",
                        "messages": [{"role": "assistant", "content": "Error: No KPI specified."}]}
            
            pseudocode = self.tool_generate_pseudocode(current_kpi, methodology, data_context, granularity_analysis)
            reflection = self._reflect_on_logic(pseudocode, data_context, kpi_name=current_kpi)
            if "REDO" in reflection.upper():
                pseudocode = self.tool_refine_pseudocode(methodology, pseudocode, reflection)
            
            existing_pseudocode_dict[current_kpi] = pseudocode
            return {
                "pseudocode": existing_pseudocode_dict,
                "current_step": "pseudocode_generated",
                "step_outputs": [{"step": "pseudocode", "kpi": current_kpi, "status": "✅", 
                                 "output": pseudocode[:200] + "..."}],
                "messages": [{"role": "assistant", "content": f"✅ Generated Logic for **{current_kpi}**."}]
            }

    @trace_tool("Pseudocode Reflection")
    def _reflect_on_logic(self, pseudocode, data_context, kpi_name):
        """Reflection step: Verifies logic against metadata."""
        prompt = f"""
        Review this pseudocode for KPI "{kpi_name}" against the available Metadata.
        
        Pseudocode:
        {pseudocode}
        
        Metadata:
        {data_context}
        
        Task:
        1. Check if ANY tables or columns in the pseudocode ARE NOT in the Metadata.
        2. Check if the final join/aggregation achieves the required KPI.
        
        If perfect, return "APPROVED".
        If issues found, return "REDO: [Detailed explanation of logic or schema mismatch]".
        """
        return call_gemini(prompt)

    def _extract_context_list(self, pseudocode, candidates):
        """No longer used in batch mode; previously extracted metadata usage."""
        return None

