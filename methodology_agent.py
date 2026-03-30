import sys
import os
import re
import json
from state import GraphState
from observability import trace_node, trace_tool, log_decision

# Add Data_Dictionary to path
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
if os.path.join(os.getcwd(), 'Data_Dictionary') not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), 'Data_Dictionary'))

from gemini_client import call_gemini
from rag_retriever import RAGRetriever
from pseudocode_agent import PseudocodeAgent

"""
### Concept: Multi-Persona Review (Self-Correction)
One of the best ways to improve LLM output is to simulate a team of experts collaborating. 
This is called the **Multi-Persona** pattern.

In the Methodology Agent, we use three distinct roles:
1. **Consultant (Drafting)**: Focuses on creativity and strategic frameworks (McKinsey/Bain style).
2. **Lead Consultant (Critique)**: Focuses on identifying gaps, logical errors, and constraint violations.
3. **Principal Consultant (Finalization)**: Weighs the draft against the critique to produce a polished, high-quality final plan.

#### Why use multiple personas?
- **Reduced Hallucinations**: Having a separate "Critique" phase forces the AI to check its own work.
- **Higher Quality**: It separates "idea generation" from "quality control," mirroring how high-stakes professional work is done.
"""

class MethodologyAgent:
    """
    Agent responsible for designing the analysis strategy.
    Uses a multi-stage process: Data-Agnostic Strategy -> Data Retrieval -> Data-Aware Methodology.
    """
    
    @trace_node("Methodology")
    def run(self, state: dict):
        hypothesis = state.get("hypothesis")
        current_step = state.get("current_step")
        internal_memory = state.get("internal_memory", {})
        metadata_context = state.get("metadata_context", "")
        latest_feedback = state.get("latest_feedback", "")
        refinement_count = state.get("hypothesis_refinement_count", 0)
        
        # ── PHASE 0: Hypothesis Refinement ──
        if current_step == "trigger_hypothesis_refinement":
            refinement_res = self.tool_refine_hypothesis(hypothesis, latest_feedback, refinement_count)
            status = refinement_res.get("status")
            
            if status == "refined" and refinement_count < 3:
                # Ask a clarifying question
                return {
                    "messages": [{"role": "assistant", "content": refinement_res["message"]}],
                    "current_step": "hypothesis_refinement", # Wait for user response
                    "hypothesis_refinement_count": refinement_count + 1
                }
            else:
                # Finalize
                final_hypothesis = refinement_res.get("final_hypothesis", hypothesis)
                return {
                    "hypothesis": final_hypothesis,
                    "current_step": "trigger_initial_strategy", # Proceed to next step
                    "messages": [{"role": "assistant", "content": f"✅ **Hypothesis Finalized**: {final_hypothesis}\n\nProceeding to strategy generation..."}]
                }

        # Determine the phase and context for the router
        is_data_aware = current_step in ["trigger_final_methodology", "refine_methodology"]
        
        # Call the internal router to manage the methodology lifecycle
        router_result = self._methodology_router(
            hypothesis=hypothesis,
            metadata=metadata_context if is_data_aware else None,
            feedback=latest_feedback,
            existing_methodology=state.get("methodology") or state.get("initial_strategy")
        )

        # Update Internal Memory
        internal_memory["methodology_reasoning"] = router_result.get("reasoning", "")
        
        log_decision("Methodology Router", router_result.get("status_message", "generate"), {
            "phase": "data_aware" if is_data_aware else "initial_strategy",
            "reasoning": router_result.get("reasoning", ""),
            "kpis_found": len(router_result.get("kpis_dict", {}))
        })
        
        updates = {
            "internal_memory": internal_memory,
        }

        if is_data_aware:
            # Stage 2 specific updates
            updates.update({
                "methodology": router_result["methodology"],
                "current_step": "methodology_generated",
                "kpis_dict": router_result.get("kpis_dict", {}),
            })
            
            # If it's the first time in Stage 2, we still need feasibility analysis
            if "granularity_analysis" not in state or current_step == "trigger_final_methodology":
                granularity_analysis = self._perform_feasibility_analysis(
                    hypothesis, 
                    router_result.get("kpis_dict", {}), 
                    metadata_context
                )
                updates["granularity_analysis"] = granularity_analysis
                feasibility_str = f"\n\n### 🔬 Feasibility & Granularity\n\n{self._format_feasibility_analysis(granularity_analysis)}"
            else:
                feasibility_str = ""

            final_msg = f"## 📋 Final Data-Aware Methodology\n\n{router_result['methodology']}{feasibility_str}\n\n---\n\n**Deep Agent Status**: {router_result['status_message']}"
            updates["messages"] = [{"role": "assistant", "content": final_msg}]
        else:
            # Stage 1 specific updates
            # Extract KPI list from the router result for downstream agents (RAG)
            kpis_dict = router_result.get("kpis_dict", {})
            kpi_names = list(kpis_dict.keys()) if isinstance(kpis_dict, dict) else []
            
            strategy_dict = self.parse_sections(router_result["methodology"])
            # Inject the structured kpis into strategy_dict for RAG compatibility
            strategy_dict["kpis"] = kpis_dict
            
            updates.update({
                "initial_strategy": router_result["methodology"],
                "initial_strategy_dict": strategy_dict,
                "kpi_list": kpi_names,
                "current_step": "strategy_generated",
                "messages": [{
                    "role": "assistant",
                    "content": f"## Phase 1: Strategic Strategy\n\n{router_result['methodology']}\n\n---\n\n**Strategy Complete.** {router_result['status_message']}\n\n**KPIs Identified**: {', '.join(kpi_names) if kpi_names else 'Parsing KPIs...'}"
                }]
            })

        return updates

    @trace_tool("Methodology Router")
    def _methodology_router(self, hypothesis, metadata=None, feedback=None, existing_methodology=None):
        """Internal Router for managing methodology generation, refinement, and brainstorming."""
        
        context = {
            "hypothesis": hypothesis,
            "has_metadata": bool(metadata),
            "has_feedback": bool(feedback),
            "is_refinement": bool(existing_methodology and feedback)
        }
        
        prompt = f"""
        You are the Methodology Router for the HyTE Deep Agent.
        Your task is to coordinate the creation or refinement of a hypothesis testing methodology.
        
        Available Tools:
        - "approach": Focuses on strategic frameworks and scenarios.
        - "assumptions": Identifies logical constraints and data dependencies.
        - "kpis": Defines measurable, actionable metrics.
        - "visualizations": Designs charts for KPI monitoring.
        
        Current Context:
        {json.dumps(context, indent=2)}
        
        Available Metadata:
        {metadata if metadata else "Not retrieved yet."}
        
        User Feedback:
        "{feedback if feedback else "None"}"
        
        Existing Methodology:
        "{existing_methodology if existing_methodology else "None"}"
        
        Decision Logic:
        - If generating for the first time: Call all tools ["approach", "assumptions", "kpis", "visualizations"].
        - If refining: Analyze the feedback and call ONLY the necessary tools to update the methodology.
        - If the user is asking for ideas or brainstorming: Set action to "brainstorm".
        
        Return ONLY a JSON object:
        {{
            "thought_process": "Why you chose these tools/actions.",
            "action": "generate" | "refine" | "brainstorm",
            "tools_to_call": ["approach", "assumptions", "kpis", "visualizations"],
            "status_message": "What you are telling the user about your progress."
        }}
        """
        
        try:
            res_text = call_gemini(prompt).strip()
            res_text = re.sub(r'```json\s*|\s*```', '', res_text)
            decision = json.loads(res_text)
        except Exception:
            decision = {"action": "generate_full", "tools_to_call": ["approach", "assumptions", "kpis", "visualizations"]}

        # Execute Tools
        results = {
            "approach": "",
            "assumptions": "",
            "kpis": "",
            "visualizations": ""
        }
        
        if decision["action"] == "brainstorm":
             # Brainstorming doesn't use individual tools yet, it creates options
             methodology = self._brainstorm_methodology(hypothesis, metadata, feedback)
             status_msg = "Brainstorming alternative approaches based on your feedback."
        else:
            # Call tools based on decision
            tools = decision.get("tools_to_call", [])
            if not tools and decision["action"] in ["generate", "refine"]:
                tools = ["approach", "assumptions", "kpis", "visualizations"]
            
            # Maintain sections from existing if refining
            if existing_methodology:
                sections = self.parse_sections(existing_methodology)
                results = {
                    "approach": sections.get("meth_approach", ""),
                    "assumptions": sections.get("meth_assumptions", ""),
                    "kpis": sections.get("meth_kpis", ""),
                    "visualizations": sections.get("meth_visualizations", "")
                }
            
            # Helper to check for tool presence (case insensitive and fuzzy)
            def is_tool(name):
                return any(name in t.lower() for t in tools)

            if is_tool("approach") or not existing_methodology:
                results["approach"] = self.tool_approach(hypothesis, current_approach=results.get("approach", ""), metadata=metadata, feedback=feedback)
            if is_tool("assumptions") or not existing_methodology:
                results["assumptions"] = self.tool_assumptions(hypothesis, results.get("approach", ""), current_assumptions=results.get("assumptions", ""), metadata=metadata, feedback=feedback)
            if is_tool("kpi") or not existing_methodology:
                results["kpis"] = self.tool_kpis(hypothesis, results.get("approach", ""), results.get("assumptions", ""), current_kpis=results.get("kpis", ""), metadata=metadata, feedback=feedback)
            if is_tool("visual") or not existing_methodology:
                results["visualizations"] = self.tool_visualizations(hypothesis, results.get("approach", ""), results.get("assumptions", ""), results.get("kpis", ""), current_visuals=results.get("visualizations", ""), metadata=metadata, feedback=feedback)
            
            methodology = self._format_draft(results)
            status_msg = decision.get("status_message", "Refined the methodology based on your input.")

        # Parse KPIs for state updates
        kpis_dict = {}
        if results.get("kpis"):
            try:
                kpis_json = re.sub(r'```json\n?|\n?```', '', results["kpis"]).strip()
                kpis_dict = json.loads(kpis_json)
            except:
                pass

        return {
            "methodology": methodology,
            "reasoning": decision.get("thought_process", ""),
            "status_message": status_msg,
            "kpis_dict": kpis_dict
        }

    def tool_approach(self, hypothesis, current_approach="", metadata=None, feedback=None):
        """Tool for generating/refining the Approach section."""
        if metadata:
            return self._generate_data_aware_approach(hypothesis, {"approach": current_approach}, metadata)
        return self._generate_approach(hypothesis, data_context=None)

    def tool_assumptions(self, hypothesis, approach, current_assumptions="", metadata=None, feedback=None):
        """Tool for generating/refining the Assumptions section."""
        if metadata:
            return self._generate_data_aware_assumptions(hypothesis, {"assumptions": current_assumptions}, approach, metadata)
        return self._generate_assumptions(hypothesis, approach)

    def tool_kpis(self, hypothesis, approach, assumptions, current_kpis="", metadata=None, feedback=None):
        """Tool for generating/refining the KPIs section."""
        if metadata:
            return self._generate_data_aware_kpis(hypothesis, {"kpis": current_kpis}, approach, assumptions, metadata)
        return self._generate_kpis(hypothesis, approach, assumptions)

    def tool_visualizations(self, hypothesis, approach, assumptions, kpis, current_visuals="", metadata=None, feedback=None):
        """Tool for generating/refining the Visualizations section."""
        if metadata:
            return self._generate_data_aware_visualizations(hypothesis, {"visualizations": current_visuals}, approach, assumptions, kpis, metadata)
        return self._generate_visualizations(hypothesis, approach, assumptions, kpis)

    def _brainstorm_methodology(self, hypothesis, metadata, feedback):
        """Interacts with the user to brainstorm multiple methodology options."""
        prompt = f"""
        You are an expert Strategic Consultant. The user wants to brainstorm ideas for testing their hypothesis.
        Hypothesis: "{hypothesis}"
        Metadata: {metadata if metadata else "Not available."}
        Feedback: "{feedback}"
        
        Task:
        1. Propose 3 distinct strategic directions (e.g., Conservative, Aggressive, Cross-Sectional).
        2. For each, briefly explain the logic and the primary KPIs.
        3. Explain the pros and cons of each.
        
        Keep it high-level and collaborative.
        """
        return call_gemini(prompt)


    def _generate_approach(self, hypothesis, data_context=None):
        """Step 1: Generate Approach."""

        context_instruction = f"Context: {data_context}" if data_context else "Context: Data not yet retrieved. Focus on the STRATEGY."
        
        prompt = f"""
        You are an expert Data Science and Strategy Consultant.
        Draft the **Approach** section for the hypothesis: "{hypothesis}"
        
        {context_instruction}
        
        Task:
        - Identify at max 2-3 real world scenarios that impact the core intent of the hypothesis.
        - Provide a comprehensive strategy to study those scenarios & hypothesis and its potential impact.
        - The objective of the strategy should be to develop a step by step plan which is measurable and testable to provide a clear and concise actionalble insights to the hypothesis.
        - If required, use the frameworks and best practices used by the McKinsey, Bain, BCG, GE or top teir consulting firms etc. for data driven decision making.
        - List down the steps of how to recreate, study, analyze the real world scenarios with the possible outcomes of accepting or rejecting the hypothesis.

        Constraints:
        - Max 5 bullet points.
        - Max 200 words.
        
        Output format:
        Return ONLY the bullet points for the Approach.
        """
        return call_gemini(prompt)

    def _generate_assumptions(self, hypothesis, approach):
        """Step 2: Generate Assumptions."""

        prompt = f"""
        You are an expert Data Science Consultant.
        Based on the Hypothesis and Approach, list the Assumptions and Clarifications to make the approach actionable more clear.
        
        Hypothesis: "{hypothesis}"
        Approach: "{approach}"
        
        Task:
        - List assumptions made while creating the methodology.
        - Ask clarifying questions if ambiguities exist.
        
        Constraints:
        - Max 4 bullet points.
        - Max 200 words.
        
        Output format:
        Return ONLY the bullet points.
        """
        return call_gemini(prompt)

    def _generate_kpis(self, hypothesis, approach, assumptions):
        """Step 3: Generate KPIs."""

        prompt = f"""
        You are an expert Data Science Consultant.
        Identify the KPIs required to test the hypothesis based on the approach and assumptions.
        
        Hypothesis: "{hypothesis}"
        Approach: "{approach}"
        Assumptions: "{assumptions}"
        
        Task:
        Identify a list of a maximum of 5-6 KPIs.
        - KPIs should be relevant to the hypothesis.
        - KPIs should be measurable.
        - KPIs should be actionable.
        
        Constraints:
        - Return ONLY a JSON dictionary where keys are KPI names and values are descriptions.
        - Description must be < 20 words.
        
        Example Output:
        {{
          "ARPU": "Monthly average revenue per user to track monetization.",
          "Churn Rate": "Percentage of subscribers leaving the network monthly."
        }}
        """
        return call_gemini(prompt)

    def _generate_visualizations(self, hypothesis, approach, assumptions, kpis):
        """Step 4: Generate Visualizations."""

        prompt = f"""
        You are an expert Data Science Consultant.
        Suggest Visualizations to monitor the KPIs based on the approach and assumptions.
        
        Hypothesis: "{hypothesis}"
        Approach: "{approach}"
        Assumptions: "{assumptions}"
        KPIs: "{kpis}"
        
        Task:
        - List the Visualizations.
        - Visualizations should be relevant to the KPIs & hypothesis.
        - Visualizations should be measurable.
        - Visualizations should be actionable.
        
        Constraints:
        - Max 5-6 Visualizations.
        - **Format**: Return a JSON-style Array List where each item is a Key-Value pair:
          - Key: Chart Type with Dimensions (e.g., 'Line Chart (Date vs Churn)')
          - Value: Description and Intent (MUST be < 15 words).
        
        Example Output:
        - Line Chart (Date vs Revenue): Trends in revenue over the last 12 months.
        - Bar Chart (Region vs Churn): Comparing churn rates across different geographic regions.
        
        Output format:
        Return ONLY the list.
        """
        return call_gemini(prompt)

    def _get_consultant_draft(self, hypothesis, data_context=None):
        """Phase 1: Data Science and Strategy Consultant prepares a draft using sequential steps."""

        
        approach = self._generate_approach(hypothesis, data_context)
        assumptions = self._generate_assumptions(hypothesis, approach)
        kpis = self._generate_kpis(hypothesis, approach, assumptions)
        visualizations = self._generate_visualizations(hypothesis, approach, assumptions, kpis)
        
        return {
            "approach": approach,
            "assumptions": assumptions,
            "kpis": kpis,
            "visualizations": visualizations
        }

    def _format_draft(self, draft_dict):
        """Helper to format the draft dictionary into markdown."""
        return f"""
                1. **Approach**:
                {draft_dict.get('approach', '')}

                2. **KPIs**:
                {draft_dict.get('kpis', '')}

                3. **Visualizations**:
                {draft_dict.get('visualizations', '')}

                4. **Assumptions/Clarifications Needed**:
                {draft_dict.get('assumptions', '')}
                """

    def _get_lead_critique(self, hypothesis, draft):
        """Phase 2: Lead Consultant provides feedback on the draft."""

        
        # Ensure draft is string for prompt
        draft_text = draft if isinstance(draft, str) else self._format_draft(draft)
        
        prompt = f"""
        You are a Lead Data Science Consultant. Critique this draft for hypothesis: "{hypothesis}"
        Draft: {draft_text}
        
        Task:
        Provide critical feedback on the draft methodology for each of the sections seperately. Focus on the strategic value, logic, and adherence to constraints.
        Approach:
        1. Strategy Alignment: Are the scenarios identified relevant to the hypothesis? Does the methodology effectively test the hypothesis?
        2. Logic & Soundness: Is the analytical approach sound?
        3. Clarity: Does the approach has clear and understandable steps?
        4. Completeness: Does the approach cover all the necessary steps to test the hypothesis?
        
        Assumptions/Clarifications:
        1. Assumptions/Clarifications: Are the assumptions and clarifying questions relevant and helpful?
        
        KPIs:
        1. KPIs: Are there max 5-6 relevant KPIs?
        2. KPIs: Are the KPIs relevant to the hypothesis?
        3. KPIs: Are the KPIs measurable?
        4. KPIs: Are the KPIs actionable?
        
        Visualizations:
        1. Visualizations: Are there max 5-6 relevant visualizations?
        2. Visualizations: Are the visualizations relevant to the KPIs & hypothesis?
        3. Visualizations: Are the visualizations measurable?
        4. Visualizations: Are the visualizations actionable?
        
        Provide clear, actionable feedback points. Do not rewrite the methodology, just critique it.
        """
        return call_gemini(prompt)

    def _get_refined_draft(self, hypothesis, draft_dict, critique):
        """Phase 3: Principal Consultant creates the refined plan sequentially."""

        
        refined_approach = self._refine_approach(hypothesis, draft_dict['approach'], critique)
        refined_assumptions = self._refine_assumptions(hypothesis, refined_approach, draft_dict['assumptions'], critique)
        refined_kpis = self._refine_kpis(hypothesis, refined_approach, refined_assumptions, draft_dict['kpis'], critique)
        refined_visualizations = self._refine_visualizations(hypothesis, refined_approach, refined_assumptions, refined_kpis, draft_dict['visualizations'], critique)
        
        return {
            "approach": refined_approach,
            "assumptions": refined_assumptions,
            "kpis": refined_kpis,
            "visualizations": refined_visualizations
        }

    def _refine_approach(self, hypothesis, original_approach, critique):
        """Step 3.1: Refine Approach."""

        prompt = f"""
        You are a Principal Data Science Consultant. Refine the Approach based on the Lead Consultant's critique.
        
        Hypothesis: "{hypothesis}"
        Original Approach: "{original_approach}"
        Critique: "{critique}"
        
        Task:
        Refine the Approach section to address the critique while staying within constraints.
        - Identify at max 2-3 real world scenarios that impact the core intent of the hypothesis.
        - Provide a comprehensive strategy to study those scenarios & hypothesis and its potential impact.
        - The objective of the strategy should be to develop a step by step plan which is measurable and testable to provide a clear and concise actionalble insights to the hypothesis.
        - If required, use the frameworks and best practices used by the McKinsey, Bain, BCG, GE or top teir consulting firms etc. for data driven decision making.
        - List down the steps of how to recreate, study, analyze the real world scenarios with the possible outcomes of accepting or rejecting the hypothesis.

        Constraints:
        - Max 6 bullet points.
        - Max 200 words.
        
        Output format:
        Return ONLY the refined Approach bullet points.
        """
        return call_gemini(prompt)

    def _refine_assumptions(self, hypothesis, refined_approach, original_assumptions, critique):
        """Step 3.2: Refine Assumptions."""

        prompt = f"""
        You are a Principal Data Science Consultant. Refine the Assumptions/Clarifications based on the refined Approach and Lead Consultant's critique to make the approach actionable and more clear.
        
        Hypothesis: "{hypothesis}"
        Refined Approach: "{refined_approach}"
        Original Assumptions: "{original_assumptions}"
        Critique: "{critique}"
        
        Task:
        - Refine the Assumptions/Clarifications section to address the critique and align with the refined approach.
        - List assumptions made while creating the methodology.
        - Ask clarifying questions if ambiguities exist.
        
        Constraints:
        - Max 4 bullet points.
        - Max 200 words.
        
        Output format:
        Return ONLY the refined bullet points.
        """
        return call_gemini(prompt)

    def _refine_kpis(self, hypothesis, refined_approach, refined_assumptions, original_kpis, critique):
        """Step 3.3: Refine KPIs with validation."""

        
        for attempt in range(3):
            prompt = f"""
            You are a Principal Data Science Consultant. Refine the KPIs based on the refined Approach, refined Assumptions, and Lead Consultant's critique.
            
            Hypothesis: "{hypothesis}"
            Refined Approach: "{refined_approach}"
            Refined Assumptions: "{refined_assumptions}"
            Original KPIs: "{original_kpis}"
            Critique: "{critique}"
            
            Task:
            Identify a list of a maximum of 5-6 refined KPIs. Address the critique and align with the refined strategy.
            - KPIs should be relevant to the hypothesis.
            - KPIs should be measurable.
            - KPIs should be actionable.
            
            Constraints:
            - Return ONLY a JSON dictionary where keys are KPI names and values are descriptions.
            - Description must be < 20 words.
            
            Example Output:
            {{
              "ARPU": "Monthly average revenue per user to track monetization.",
              "Churn Rate": "Percentage of subscribers leaving the network monthly."
            }}
            """
            response = call_gemini(prompt)
            
            # Validation logic
            try:
                # Basic JSON check for dictionary
                data = json.loads(re.sub(r'```json\n?|\n?```', '', response).strip())
                if isinstance(data, dict) and len(data) > 0:
                    return response
            except:
                pass
                
        return original_kpis # Fallback

    def _refine_visualizations(self, hypothesis, refined_approach, refined_assumptions, refined_kpis, original_visuals, critique):
        """Step 3.4: Refine Visualizations with validation."""

        
        for attempt in range(3):
            prompt = f"""
            You are a Principal Data Science Consultant. Refine the Visualizations based on the refined sections and Lead Consultant's critique to monitor the KPIs.
            
            Hypothesis: "{hypothesis}"
            Refined Approach: "{refined_approach}"
            Refined Assumptions: "{refined_assumptions}"
            Refined KPIs: "{refined_kpis}"
            Critique: "{critique}"
            
            Task:
            - List 5-6 refined Visualizations. Address the critique and ensure relevance to the refined KPIs.
            - Visualizations should be relevant to the KPIs & hypothesis.
            - Visualizations should be measurable.
            - Visualizations should be actionable.
            
            Constraints:
            - Max 5-6 Visualizations.
            - **Format**: Return a JSON-style Array List where each item is a Key-Value pair:
              - Key: Chart Type with Dimensions (e.g., 'Line Chart (Date vs Churn)')
              - Value: Description and Intent (MUST be < 15 words).
            
            Output format:
            Return ONLY the list.
            """
            response = call_gemini(prompt)
            
            # Validation logic: Check if it looks like a list with at least one item
            if response and (("-" in response) or ("[" in response)):
                return response

            
        return original_visuals # Fallback

    def _generate_data_aware_approach(self, hypothesis, strategy_dict, data_context):
        """Step 4.1: Generate Data-Aware Approach."""

        prompt = f"""
        You are the Data Strategy Manager. Refine the Approach to reference specific tables and columns.
        
        Hypothesis: "{hypothesis}"
        Initial Strategy Approach: "{strategy_dict['approach']}"
        Data Context (Tables & Columns):
        {data_context}
        
        Task:
        1. Review the Strategy and the Data Context.
        2. Replace generic data references with specific Table and Column names.
        3. Ensure the approach remains strategic while becoming feasible and testable.
        
        Constraints:
        - Max 5 bullet points.
        - Max 200 words.
        
        Output format:
        Return ONLY the refined Data-Aware Approach bullet points.
        """
        return call_gemini(prompt)

    def _generate_data_aware_assumptions(self, hypothesis, strategy_dict, refined_approach, data_context):
        """Step 4.2: Generate Data-Aware Assumptions."""

        prompt = f"""
        You are the Data Strategy Manager. Refine the Assumptions to be data-specific and actionable.
        
        Hypothesis: "{hypothesis}"
        Initial Strategy Assumptions: "{strategy_dict['assumptions']}"
        Refined Data-Aware Approach: "{refined_approach}"
        Data Context (Tables & Columns):
        {data_context}
        
        Task:
        1. List data-specific assumptions based on the available metadata.
        2. Ask clarifying questions regarding data quality or availability if ambiguities exist.
        
        Constraints:
        - Max 4 bullet points.
        - Max 200 words.
        
        Output format:
        Return ONLY the refined bullet points.
        """
        return call_gemini(prompt)

    def _generate_data_aware_kpis(self, hypothesis, strategy_dict, refined_approach, refined_assumptions, data_context):
        """Step 4.3: Generate Data-Aware KPIs."""

        prompt = f"""
        You are the Data Strategy Manager. Refine the KPIs to map to specific source columns and tables.
        
        Hypothesis: "{hypothesis}"
        Initial Strategy KPIs: "{strategy_dict['kpis']}"
        Refined Data-Aware Approach: "{refined_approach}"
        Refined Data-Aware Assumptions: "{refined_assumptions}"
        Data Context (Tables & Columns):
        {data_context}
        
        Task:
        Identify a list of a maximum of 5-6 refined KPIs.
        - Replace generic KPIs with those that can be calculated from the provided tables.
        - Specify the source table/column for each KPI.
        
        Constraints:
        - Return ONLY a JSON dictionary where keys are KPI names and values are descriptions.
        - Description must include the source table/column.
        
        Output format:
        Return ONLY the JSON dictionary.
        """
        return call_gemini(prompt)

    def _generate_data_aware_visualizations(self, hypothesis, strategy_dict, refined_approach, refined_assumptions, refined_kpis, data_context):
        """Step 4.4: Generate Data-Aware Visualizations."""

        prompt = f"""
        You are the Data Strategy Manager. Refine the Visualizations to monitor the data-aware KPIs.
        
        Hypothesis: "{hypothesis}"
        Refined Data-Aware Approach: "{refined_approach}"
        Refined Data-Aware Assumptions: "{refined_assumptions}"
        Refined Data-Aware KPIs: "{refined_kpis}"
        Data Context (Tables & Columns):
        {data_context}
        
        Task:
        - List 5-6 refined Visualizations feasible with the available data.
        - Ensure relevance to the refined KPIs.
        
        Constraints:
        - Max 5-6 Visualizations.
        - **Format**: Return a JSON-style Array List where each item is a Key-Value pair:
          - Key: Chart Type with Dimensions (e.g., 'Line Chart (Date vs Churn)')
          - Value: Description and Intent.
        
        Output format:
        Return ONLY the list.
        """
        return call_gemini(prompt)

    def _get_data_strategy_manager_final(self, hypothesis, strategy_dict, feedback=""):
        """
        Phase 4: Data Strategy Manager - Orchestrates the full data-aware flow sequentially.
        """

        
        # 1. RAG Retrieval (Internal)

        # Use keys from the refined strategy's KPI dictionary
        refined_kpis = strategy_dict.get('kpis', {})
        if isinstance(refined_kpis, str):
            # If it's still a JSON string, parse it
            try:
                refined_kpis = json.loads(re.sub(r'```json\n?|\n?```', '', refined_kpis).strip())
            except:
                refined_kpis = {}
        
        kpi_names = list(refined_kpis.keys()) if isinstance(refined_kpis, dict) else []
        
        retriever = RAGRetriever()
        kpi_candidates = retriever.retrieve_candidates_for_kpis(kpi_names)
        
        # 2. Pseudocode & Metadata Generation (Internal)

        ps_agent = PseudocodeAgent()
        initial_strategy_text = self._format_draft(strategy_dict)
        pseudocode, context_list = ps_agent.generate_with_validation(hypothesis, initial_strategy_text, kpi_candidates, feedback=feedback)
        
        # Convert context_list to markdown table for the prompt
        def to_markdown(ctx_list):
            val = "| S.No | Table | KPIs | Columns | Reasoning |\n|---|---|---|---|---|\n"
            for row in ctx_list:
                if len(row) >= 5:
                    val += f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |\n"
            return val
        data_context_table = to_markdown(context_list)
        
        # Pack data_context into strategy_dict for sub-functions
        strategy_dict['data_context'] = data_context_table

        # 3. Sequential Data-Aware Generation
        da_approach = self._generate_data_aware_approach(hypothesis, strategy_dict, data_context_table)
        da_assumptions = self._generate_data_aware_assumptions(hypothesis, strategy_dict, da_approach, data_context_table)
        da_kpis = self._generate_data_aware_kpis(hypothesis, strategy_dict, da_approach, da_assumptions, data_context_table)
        da_visualizations = self._generate_data_aware_visualizations(hypothesis, strategy_dict, da_approach, da_assumptions, da_kpis, data_context_table)
        
        final_dict = {
            "approach": da_approach,
            "assumptions": da_assumptions,
            "kpis": da_kpis,
            "visualizations": da_visualizations
        }
        final_methodology = self._format_draft(final_dict)
        
        # Parse KPIs into dict for feasibility analysis
        try:
            kpis_json = re.sub(r'```json\n?|\n?```', '', da_kpis).strip()
            kpis_dict = json.loads(kpis_json)
        except Exception as e:

            kpis_dict = {}

        return {
            "methodology": final_methodology,
            "methodology_pseudocode": pseudocode,
            "pseudocode": {},  # Initialize as empty dict for per-KPI logic
            "metadata_context": data_context_table,
            "context_list": context_list,
            "kpis_dict": kpis_dict
        }


    @staticmethod
    def parse_sections(methodology_text):
        sections = {
            "meth_approach": "",
            "meth_kpis": "",
            "meth_visualizations": "",
            "meth_assumptions": ""
        }
        # Updated patterns to match the new 4-section structure:
        # 1. **Approach**:
        # 2. **KPIs**:
        # 3. **Visualizations**:
        # 4. **Assumptions/Clarifications Needed**:
        patterns = {
            "meth_approach": r"1\.\s*\*\*Approach\*\*:(.*?)(?=2\.\s*\*\*KPIs|$)",
            "meth_kpis": r"2\.\s*\*\*KPIs\*\*:(.*?)(?=3\.\s*\*\*Visualizations|$)",
            "meth_visualizations": r"3\.\s*\*\*Visualizations\*\*:(.*?)(?=4\.\s*\*\*Assumptions|$)",
            "meth_assumptions": r"4\.\s*\*\*Assumptions/Clarifications Needed\*\*:(.*)"
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, methodology_text, re.DOTALL | re.IGNORECASE)
            if match:
                sections[key] = match.group(1).strip()
        return sections
    
    def _perform_feasibility_analysis(self, hypothesis, kpis_dict, metadata_context):
        """
        Analyze feasibility of aligning all KPI datasets to the same granularity.
        Returns a dict with granularity, mergeable groups, and joining keys.
        """

        
        kpi_names = list(kpis_dict.keys()) if kpis_dict else []
        if not kpi_names:
            return {
                "common_granularity": "unknown",
                "mergeable_groups": [],
                "joining_keys": {},
                "analysis_notes": "No KPIs identified for analysis"
            }
        
        prompt = f"""
        You are a Principal Data Architect analyzing a hypothesis testing workflow.
        
        Hypothesis: "{hypothesis}"
        
        KPIs to Calculate:
        {json.dumps(kpis_dict, indent=2)}
        
        Available Metadata (Tables and Columns):
        {metadata_context[:2000]}  # Truncate for token limit
        
        Task:
        Analyze if all KPI datasets can be aligned to the same granularity level and identify optimal merge strategy.
        
        Questions to Answer:
        1. What is the most appropriate common granularity? (e.g., customer-level, transaction-level, tower-level, daily-level)
        2. Which KPIs can share the same master dataset (mergeable)?
        3. What are the common joining keys for each mergeable group?
        4. Which KPIs require separate datasets due to incompatible granularity?
        
        Return ONLY a JSON object:
        {{
          "common_granularity": "customer-level",  // Most common granularity across KPIs
          "mergeable_groups": [
            {{"group_id": 1, "kpis": ["Churn Rate", "ARPU"], "granularity": "customer-level"}},
            {{"group_id": 2, "kpis": ["Network Uptime"], "granularity": "tower-daily-level"}}
          ],
          "joining_keys": {{
            "group_1": ["cust_id", "msisdn"],
            "group_2": ["tower_id", "date"]
          }},
          "analysis_notes": "Brief explanation of the grouping strategy"
        }}
        """
        
        try:
            response = call_gemini(prompt).strip().replace("```json", "").replace("```", "")
            analysis = json.loads(response)

            return analysis
        except Exception as e:

            # Fallback: treat all KPIs as separate
            return {
                "common_granularity": "mixed",
                "mergeable_groups": [{
                    "group_id": i+1,
                    "kpis": [kpi],
                    "granularity": "unknown"
                } for i, kpi in enumerate(kpi_names)],
                "joining_keys": {},
                "analysis_notes": f"Fallback: treating all KPIs separately due to analysis error: {str(e)}"
            }
    
    def _format_feasibility_analysis(self, analysis):
        """Format feasibility analysis for display."""
        if not analysis:
            return "No analysis available."
        
        mergeable_groups = analysis.get("mergeable_groups", [])
        joining_keys = analysis.get("joining_keys", {})
        
        output = f"**Common Granularity:** {analysis.get('common_granularity', 'Unknown')}\\n\\n"
        output += f"**Mergeable Groups:** {len(mergeable_groups)}\\n\\n"
        
        for group in mergeable_groups:
            group_id = group.get("group_id", "?")
            kpis = ", ".join(group.get("kpis", []))
            granularity = group.get("granularity", "unknown")
            keys = ", ".join(joining_keys.get(f"group_{group_id}", []))
            output += f"- **Group {group_id}**: {kpis} ({granularity})\\n"
            if keys:
                output += f"  - Joining Keys: `{keys}`\\n"
        
        output += f"\\n**Analysis Notes:** {analysis.get('analysis_notes', 'N/A')}"
        return output

    @trace_tool("Refine Hypothesis")
    def tool_refine_hypothesis(self, hypothesis, feedback, refinement_count):
        """
        Analyzes the hypothesis and either asks a clarifying question or finalizes it.
        """
        prompt = f"""
        You are a Hypothesis Clarification Expert. Your goal is to ensure the user's research hypothesis is clear, specific, and testable.
        
        Current Hypothesis: "{hypothesis}"
        User Feedback/Clarification: "{feedback}"
        Refinement Round: {refinement_count} / 3
        
        Task:
        1. Evaluate the hypothesis. Is it too vague? Does it lack a target metric, population, or timeframe?
        2. If it needs clarification and Round < 3, provide exactly ONE concise clarifying question to help the user improve it.
        3. If it's clear enough, or if this is Round 3, finalize the hypothesis by rewriting it to be professional and testable.
        
        Return ONLY a JSON object:
        {{
            "status": "refined" | "finalized",
            "message": "Concentrated question if status is 'refined'",
            "final_hypothesis": "The polished final version if status is 'finalized'"
        }}
        """
        try:
            res = call_gemini(prompt).strip().replace("```json", "").replace("```", "")
            return json.loads(res)
        except:
            return {"status": "finalized", "final_hypothesis": hypothesis}

    

