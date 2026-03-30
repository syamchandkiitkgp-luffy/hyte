from langgraph.graph import StateGraph, END
from state import GraphState
from rag_retriever import RAGRetriever

"""
### Concept: LangGraph Architecture
LangGraph builds upon the concept of a **State Machine**. 
The workflow is represented as a directed graph where:

1. **Nodes**: These are Python functions. Each node receives the current `GraphState`, performing some logic (like calling an AI agent), and returns an *update* to the state.
2. **Edges**: These define the transition from one node to another (e.g., from Node A to Node B).
3. **Conditional Edges (Routing)**: Instead of a straight line, the graph can "branch". A **Router** function decides which node to go to next based on the current state.
4. **Graph Compilation**: Before running, the graph is "compiled". This validates the connections and prepares it for execution.

#### Why LangGraph?
- **Cyclic Workflows**: Unlike standard chains, LangGraph allows for loops (e.g., a "Retry" loop if code fails).
- **Human-in-the-loop**: You can pause execution at an `END` state, wait for user input, and then resume.
"""

# --- Compatibility Patch for LangChain ---
import langchain
for attr in ['debug', 'verbose', 'llm_cache']:
    if not hasattr(langchain, attr):
        setattr(langchain, attr, False if attr != 'llm_cache' else None)

# Import Agents
from orchestrator_agent import OrchestratorAgent
from methodology_agent import MethodologyAgent
from pseudocode_agent import PseudocodeAgent
from codegen_agent import CodeGenerationAgent
from execution_agent import ExecutionAgent
from merge_agent import MergeAgent
from evaluation_agent import EvaluationAgent

# Concept: Agent Initialization
# These are the actual AI "brains" that will be called by our graph nodes.
orchestrator = OrchestratorAgent()
methodology_gen = MethodologyAgent()
pseudocode_gen = PseudocodeAgent()
codegen = CodeGenerationAgent()
execution_executor = ExecutionAgent()
merge_agent = MergeAgent()
evaluator = EvaluationAgent()
rag_agent = RAGRetriever()

# --- Node Wrappers ---
# These are the functions that LangGraph calls. 
# They act as "gatekeepers" that pass the State to an agent and return the updated result.

def orchestrator_node(state: GraphState):
    """
    Concept: The Orchestrator
    This is the first node to run. It evaluates the user's intent and decides:
    1. Is the hypothesis clear? (Validation)
    2. Should we proceed to the next step? (Approval)
    3. Should we go back and fix something? (Refinement)
    
    It updates the 'current_step' in the state, which is used by the Router.
    """

    result = orchestrator.run(state)

    return result

def methodology_node(state: GraphState):
    result = methodology_gen.run(state)
    return result

def rag_node(state: GraphState):
    """
    RAG Agent Node: Uses Deep Agent logic (Internal Router) to manage data retrieval.
    """
    result = rag_agent.run(state)
    return result

def pseudocode_node(state: GraphState):
    result = pseudocode_gen.run(state)
    return result

def codegen_node(state: GraphState):
    result = codegen.run(state)
    return result

def execution_node(state: GraphState):

    result = execution_executor.run(state)
    
    # Enhance message with execution results
    if "kpi_execution_results" in result:
        results = result["kpi_execution_results"]
        summary = f"Executed {len(results)} KPI scripts.\n\n"
        for kpi, output in results.items():
            status = "✅" if "error" not in output.lower() else "❌"
            summary += f"- {status} **{kpi}**: {output[:100]}...\n"
        
        result["messages"] = [{
            "role": "assistant",
            "content": f"## 🚀 Multi-KPI Execution Complete\n\n{summary}\n\n---\n\nType 'approve' or 'merge' to intelligently join these datasets."
        }]
    

    return result

def merge_node(state: GraphState):

    result = merge_agent.run(state)
    
    if "merge_report" in result:
        report = result["merge_report"]
        result["messages"] = [{
            "role": "assistant",
            "content": f"## 🔗 Smart Merge Complete\n\n{report}\n\n---\n\nThe hypothesis testing datasets are ready for evaluation."
        }]
    

    return result

def evaluation_node(state: GraphState):

    result = evaluator.run(state)

    return result

# --- Graph Routing Logic ---

def router(state: GraphState):
    """
    Concept: The Router (Conditional Edges)
    In a standard linear chain, A always goes to B. 
    In HyTE, we need flexibility. 
    The Router looks at 'current_step' in the State and decides the next Node.
    
    Example:
    - If 'current_step' is 'trigger_codegen', it routes to the 'codegen' node.
    - If 'current_step' is 'code_generated', it returns 'END' to wait for user approval.
    """
    step = state.get("current_step")
    
    # Routing logic based on triggers
    if step in ["trigger_initial_strategy", "refine_strategy", "trigger_hypothesis_refinement"]:
        next_node = "methodology"  # Initial strategy generation/refinement/hypothesis refinement
    elif step == "trigger_rag":
        next_node = "rag"          # Data retrieval
    elif step == "rag_generated":
        next_node = "orchestrator" # Decision loop
    elif step == "strategy_generated":
        next_node = END  # Wait for user approval of strategy
    elif step in ["trigger_final_methodology", "refine_methodology"]:
        next_node = "methodology"  # Generate final data-aware methodology
    elif step == "methodology_generated":
        next_node = END  # Wait for user to approve/refine (triggers per-KPI workflow on approval)
    elif step in ["trigger_pseudocode", "refine_pseudocode"]:
        next_node = "pseudocode"  # Batch pseudocode generation for all KPIs
    elif step == "pseudocode_generated":
        next_node = END  # Wait for user to review all KPI pseudocodes
    elif step in ["trigger_codegen", "refine_codegen"]:
        next_node = "codegen"  # Batch code generation for all KPIs
    elif step == "code_generated":
        next_node = END  # Wait for user to review all KPI codes
    elif step == "trigger_execution":
        next_node = "execution"
    elif step == "executed":
        next_node = END  # Wait for user to trigger merge
    elif step == "trigger_merge":
        next_node = "merge"  # Smart dataset merging
    elif step == "merge_completed":
        next_node = END  # Wait for user to proceed to evaluation
    elif step == "execution_failed" and state.get("retry_count", 0) < 3:
        next_node = "codegen"
    elif step == "hypothesis_refinement":
        next_node = END  # Wait for user input
    else:
        next_node = END
    

    return next_node

def create_hyte_graph():
    """
    Concept: Graph Construction
    This is where we define the structure of the workflow.
    """
    # 1. Initialize the StateGraph with our GraphState schema
    workflow = StateGraph(GraphState)
    
    # 2. Add Nodes: Register the wrapper functions
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("methodology", methodology_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("pseudocode", pseudocode_node)
    workflow.add_node("codegen", codegen_node)
    workflow.add_node("execution", execution_node)
    workflow.add_node("merge", merge_node)
    workflow.add_node("evaluation", evaluation_node)
    
    # 3. Define the Entry Point: Where the execution starts
    workflow.set_entry_point("orchestrator")
    
    # 4. Define Conditional Edges: 
    # Use the 'router' function to decide where the 'orchestrator' sends the state.
    workflow.add_conditional_edges("orchestrator", router, {
        "methodology": "methodology",
        "rag": "rag",
        "pseudocode": "pseudocode",
        "codegen": "codegen",
        "execution": "execution",
        "merge": "merge",
        "evaluation": "evaluation",
        "orchestrator": "orchestrator",
        END: END
    })
    
    # 5. Define Subsequent Edges:
    workflow.add_conditional_edges("methodology", router, {
        "methodology": "methodology",
        "rag": "rag",
        "pseudocode": "pseudocode",
        "codegen": "codegen",
        "execution": "execution",
        "merge": "merge",
        "evaluation": "evaluation",
        "orchestrator": "orchestrator",
        END: END
    })
    
    workflow.add_conditional_edges("rag", router, {
        "methodology": "methodology",
        "rag": "rag",
        "pseudocode": "pseudocode",
        "codegen": "codegen",
        "execution": "execution",
        "merge": "merge",
        "evaluation": "evaluation",
        "orchestrator": "orchestrator",
        END: END
    })

    workflow.add_conditional_edges("pseudocode", router, {
        "methodology": "methodology",
        "pseudocode": "pseudocode",
        "codegen": "codegen",
        "execution": "execution",
        "merge": "merge",
        "evaluation": "evaluation",
        "orchestrator": "orchestrator",
        END: END
    })
    
    workflow.add_conditional_edges("codegen", router, {
        "methodology": "methodology",
        "pseudocode": "pseudocode",
        "codegen": "codegen",
        "execution": "execution",
        "merge": "merge",
        "evaluation": "evaluation",
        "orchestrator": "orchestrator",
        END: END
    })
    
    workflow.add_conditional_edges("execution", router, {
        "methodology": "methodology",
        "pseudocode": "pseudocode",
        "codegen": "codegen",
        "execution": "execution",
        "merge": "merge",
        "evaluation": "evaluation",
        "orchestrator": "orchestrator",
        END: END
    })

    workflow.add_conditional_edges("merge", router, {
        "evaluation": "evaluation",
        "orchestrator": "orchestrator",
        END: END
    })
    
    # Simple straight edge: Evaluation always ends the process
    workflow.add_edge("evaluation", END)
    
    # 6. Compile: Turn the definition into an executable Application
    return workflow.compile()



# Example Usage
if __name__ == "__main__":
    app = create_hyte_graph()

