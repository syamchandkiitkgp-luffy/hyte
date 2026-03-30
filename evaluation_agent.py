import sys
import os
import json
from datetime import datetime
from state import GraphState
from observability import trace_node

# Add Data_Dictionary to path
sys.path.append(os.path.join(os.getcwd(), 'Data_Dictionary'))
from gemini_client import call_gemini

class EvaluationAgent:
    """
    Quality Assurance and Performance Evaluation Agent.
    Analyzes the entire hypothesis testing workflow to provide insights and suggestions.
    """
    
    @trace_node("Evaluation")
    def run(self, state: GraphState):
        """Analyzes the state and generates an evaluation report."""
        hypothesis = state.get("hypothesis", "")
        methodology = state.get("methodology", "")
        pseudocode = state.get("pseudocode", "")
        python_code = state.get("python_code", "")
        execution_results = state.get("execution_results", "")
        retry_count = state.get("retry_count", 0)
        user_feedback = state.get("user_feedback", [])
        
        prompt = f"""
        You are a Senior AI Evaluator. Your task is to analyze the performance of multiple AI agents involved in a hypothesis testing workflow.
        
        **Objective**: Evaluate the quality, accuracy, and reliability of the analysis and provide actionable feedback for improvement.
        
        **Workflow Context**:
        - Hypothesis: {hypothesis}
        - Methodology: {methodology[:1000]}...
        - Pseudo-code: {pseudocode[:1000]}...
        - Python Code: {python_code[:1000]}...
        - Execution Results: {execution_results[:1000]}...
        - Execution Retries: {retry_count}
        - User Feedback So Far: {user_feedback}
        
        **Evaluation Criteria**:
        1. **Logic Consistency**: Does the methodology align with the hypothesis? Does the code follow the pseudo-code?
        2. **Technical Quality**: Is the code efficient and error-prone? Why did it fail (if it did)?
        3. **Business Value**: Are the results actionable for a telecom business?
        4. **User Satisfaction**: How did the user perceive the agent's interactions based on feedback?
        
        **Output Format (JSON)**:
        {{
            "overall_score": 1-10,
            "agent_performance": {{
                "orchestrator": "feedback",
                "methodology_agent": "feedback",
                "pseudocode_agent": "feedback",
                "codegen_agent": "feedback",
                "execution_agent": "feedback"
            }},
            "failures_analysis": "Root cause analysis of any code failures/retries.",
            "prompt_refinement_suggestions": ["suggestion 1", "suggestion 2"],
            "guardrail_recommendations": ["recommendation 1"]
        }}
        
        Output ONLY the JSON object.
        """
        
        response = call_gemini(prompt).strip()
        
        # Clean response if LLM includes backticks
        if response.startswith("```"):
            response = "\n".join(response.split("\n")[1:])
        if response.endswith("```"):
            response = "\n".join(response.split("\n")[:-1])
            
        try:
            eval_data = json.loads(response)
            print(f"  [EVALUATION] Overall Score: {eval_data.get('overall_score')}/10")
        except:
            eval_data = {"error": "Failed to parse evaluator response", "raw_response": response}
            
        # Add timestamp and current step
        eval_data["timestamp"] = datetime.now().isoformat()
        eval_data["hypothesis"] = hypothesis
        
        # Save evaluation report to artifact path if it exists
        artifact_path = state.get("artifact_path")
        if artifact_path and os.path.exists(artifact_path):
            with open(os.path.join(artifact_path, "evaluation_report.json"), "w", encoding="utf-8") as f:
                json.dump(eval_data, f, indent=4)
                
        return {
            "evaluations": [eval_data],
            "messages": [{"role": "assistant", "content": f"**System Evaluation Complete** 🔍\n\nI've analyzed the workflow performance. \nOverall Score: **{eval_data.get('overall_score', 'N/A')}/10**\n\nSuggestions for improvement have been logged in the evaluation report."}]
        }
