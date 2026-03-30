import streamlit as st
import time
import os


def _sync_streamlit_secrets_to_env():
    """Copy selected Streamlit secrets into environment variables for downstream modules."""
    secret_keys = [
        "GEMINI_API_KEY",
        "GEMINI_API_KEYS",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "ENABLE_PHOENIX",
        "PHOENIX_COLLECTOR_ENDPOINT",
    ]
    for key in secret_keys:
        if key in os.environ:
            continue
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None
        if value:
            if isinstance(value, (list, tuple)):
                os.environ[key] = ",".join(str(v) for v in value if str(v).strip())
            else:
                os.environ[key] = str(value)


_sync_streamlit_secrets_to_env()

from hyte_graph import create_hyte_graph


# --- Page Config ---
st.set_page_config(page_title="HyTE - Hypothesis Testing Engine", layout="wide", page_icon="📊")

from observability import setup_observability

# --- Observability ---
if "phoenix_url" not in st.session_state:
    try:
        url = setup_observability()
        st.session_state.phoenix_url = url
    except Exception as e:
        print(f"Observability setup failed: {e}")
        st.session_state.phoenix_url = None

if st.session_state.phoenix_url:
    st.sidebar.markdown(f"[🚀 Arize Phoenix UI]({st.session_state.phoenix_url})")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    .st-emotion-cache-1kyxreq {
        justify-content: center;
    }
    .highlight {
        color: #2e7bcf;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def format_metadata_view(content):
    """Parses and displays metadata context whether it is JSON (Candidates) or Markdown (Context Table)."""
    import json
    import pandas as pd
    
    # Try parsing as JSON (Candidates List)
    try:
        if content.strip().startswith("{"):
            candidates_map = json.loads(content)
            # candidates_map is { "KPI": [ {Table, Rank, ...} ] }
            
            st.markdown("### 🔍 RAG Candidates (Top 5 per KPI)")
            
            for kpi, candidates in candidates_map.items():
                with st.expander(f"**{kpi}**", expanded=False):
                    # Convert to simple dataframe for display
                    df_data = []
                    for c in candidates:
                        df_data.append({
                            "Rank": c.get("Rank"),
                            "Table": c.get("Table"),
                            "Sim": f"{c.get('Similarity', 0):.2f}",
                            "Desc": c.get("Description", "")
                        })
                    st.table(pd.DataFrame(df_data))
            return
    except:
        pass # Not JSON or failed, treat as Markdown
        
    # Default: Markdown (The Final Context Table)
    st.markdown(content)

# --- Table Mapping for Artifact View ---
STEP_TO_STATE = {
    "strategy_generated": "initial_strategy",
    "methodology_generated": "methodology",
    "pseudocode_generated": "pseudocode",
    "code_generated": "python_code",
    "executed": "kpi_execution_results"
}

# --- Initialization ---
if "graph" not in st.session_state:
    st.session_state.graph = create_hyte_graph()

if "viewing_artifact" not in st.session_state:
    st.session_state.viewing_artifact = None

if "graph_state" not in st.session_state:
    st.session_state.graph_state = {
        "messages": [],
        "hypothesis": "",
        "current_step": "start",
        "retry_count": 0,
        "hypothesis_refinement_count": 0,
        "metadata_context": "",
        "methodology": "",
        "pseudocode": {},
        "python_code": {},
        "kpi_execution_results": {},
        "execution_results": "",
        "step_outputs": [],
        "evaluations": [],
        "user_feedback": []
    }

# --- Sidebar Progress ---
with st.sidebar:
    st.title("🛡️ HyTE Progress")
    if st.button("💬 SHOW CHAT / REFRESH", use_container_width=True, type="primary"):
        st.session_state.viewing_artifact = None
        st.rerun()
    st.markdown("---")
    
    steps = {
        "start": "Waiting for Hypothesis",
        "hypothesis_refinement": "Refining Hypothesis",
        "trigger_initial_strategy": "Designing Strategy...",
        "strategy_generated": "Initial Strategy",
        "trigger_final_methodology": "Creating Methodology...",
        "methodology_generated": "Final Methodology & Logic",
        "trigger_pseudocode": "Generating Pseudocode...",
        "pseudocode_generated": "Pseudocode (All KPIs)",
        "trigger_codegen": "Writing Code...",
        "code_generated": "Code Ready (All KPIs)",
        "trigger_execution": "Running Analysis...",
        "executed": "Analysis Complete",
        "trigger_evaluation": "Evaluating Quality...",
        "finalized": "Finalized"
    }
    
    current_step = st.session_state.graph_state["current_step"]
    step_keys = list(steps.keys())
    
    for step_id, step_name in steps.items():
        is_completed = step_keys.index(step_id) < step_keys.index(current_step) if current_step in step_keys else False
        is_active = (step_id == current_step)
        
        if is_completed or (is_active and step_id != "start" and st.session_state.graph_state.get(STEP_TO_STATE.get(step_id, ""))):
            # Clickable step
            if st.button(f"✅ {step_name}", key=f"btn_{step_id}", use_container_width=True):
                st.session_state.viewing_artifact = step_id
                st.rerun()
        elif is_active:
            st.markdown(f"**🔵 {step_name}**")
        else:
            st.markdown(f"⚪ {step_name}")
            
    st.markdown("---")
    if st.button("Reset Session", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    # --- Live Feedback Status ---
    if st.session_state.graph_state.get("latest_feedback"):
         st.markdown("### 🔄 Feedback Loop")
         st.info(f"**Latest Instruction:**\n\n{st.session_state.graph_state['latest_feedback']}")
         st.caption(f"Targeting: {st.session_state.graph_state['current_step']}")

# --- Main Layout ---
st.title("📊 HyTE: Hypothesis Testing Engine")
st.caption("Grounding Telecom Hypotheses in Enriched Metadata Dictionary")

if not os.getenv("GEMINI_API_KEY") and not os.getenv("GEMINI_API_KEYS"):
    st.warning(
        "Gemini API key is not configured. Add GEMINI_API_KEY or GEMINI_API_KEYS in Streamlit app secrets."
    )

# --- UI LOGIC: Artifact View vs Chat View ---
# --- UI LOGIC: Artifact View vs Chat View ---
if st.session_state.viewing_artifact:
    artifact_id = st.session_state.viewing_artifact
    st.markdown(f"## 📄 Reviewing: {steps.get(artifact_id, artifact_id)}")
    
    state_key = STEP_TO_STATE.get(artifact_id)
    content = st.session_state.graph_state.get(state_key, "No content available yet.")
    
    if artifact_id == "pseudocode_generated":
        # Show per-KPI pseudocode in expandable sections
        if isinstance(content, dict) and content:
            st.markdown(f"### 📝 Pseudocode for {len(content)} KPIs")
            for kpi_name, pseudo in content.items():
                with st.expander(f"🔹 {kpi_name}", expanded=False):
                    st.code(pseudo, language="text")
        else:
            st.info("No pseudocode generated yet.")
    
    elif artifact_id == "code_generated":
        # Show per-KPI Python code in expandable sections
        if isinstance(content, dict) and content:
            st.markdown(f"### 💻 Python Code for {len(content)} KPIs")
            for kpi_name, code in content.items():
                with st.expander(f"🔹 {kpi_name}", expanded=False):
                    st.code(code, language="python")
        elif isinstance(content, str):
            st.code(content, language="python")
        else:
            st.info("No code generated yet.")
    
    elif artifact_id == "executed":
        # Show per-KPI execution results
        if isinstance(content, dict) and content:
            st.markdown(f"### 🚀 Execution Results for {len(content)} KPIs")
            for kpi_name, result in content.items():
                status_icon = "✅" if "error" not in result.lower() else "❌"
                with st.expander(f"{status_icon} {kpi_name}", expanded=False):
                    st.text(result)
            # Show datasets
            kpi_datasets = st.session_state.graph_state.get("kpi_datasets", {})
            if kpi_datasets:
                st.markdown("### 📊 Generated Datasets")
                for kpi_name, csv_file in kpi_datasets.items():
                    st.markdown(f"- **{kpi_name}**: `{csv_file}`")
        else:
            st.info(str(content) if content else "No execution results yet.")
        
        art_path = st.session_state.graph_state.get("artifact_path")
        if art_path and os.path.exists(os.path.join(art_path, "result_plot.png")):
            st.image(os.path.join(art_path, "result_plot.png"), caption="Analysis Visualization")
    
    elif artifact_id == "methodology_generated":
        st.markdown(content)
        st.markdown("### 🔍 Selected Metadata")
        st.markdown(st.session_state.graph_state.get("metadata_context", ""))
    else:
        st.markdown(content if isinstance(content, str) else str(content))
        
    st.markdown("---")
    if st.button("Go Back to Chat"):
        st.session_state.viewing_artifact = None
        st.rerun()
else:
    # --- Step Progress Panel ---
    step_outputs = st.session_state.graph_state.get("step_outputs", [])
    if step_outputs:
        with st.expander(f"📊 Pipeline Progress ({len(step_outputs)} steps)", expanded=False):
            for so in step_outputs:
                status = so.get('status', '⚪')
                kpi = so.get('kpi', 'N/A')
                step = so.get('step', 'unknown')
                output = so.get('output', '')
                st.markdown(f"{status} **[{step}]** {kpi}")
                if output and status != '⏳':
                    with st.expander(f"Output: {kpi}", expanded=False):
                        st.text(output[:500])
    
    # --- Chat Interface ---
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.graph_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # --- Approval Buttons ---
    # Show buttons when waiting for approval at specific steps
    current_step = st.session_state.graph_state.get("current_step", "start")
    
    approval_steps = {
        "strategy_generated": "✅ Approve Strategy & Retrieve Data Tables",
        "methodology_generated": "✅ Approve Methodology & Generate Pseudocode for All KPIs",
        "pseudocode_generated": "✅ Approve Pseudocode & Generate Code for All KPIs",
        "code_generated": "🚀 Execute All KPI Scripts Against Datalake",
        "executed": "🔗 Merge Per-KPI Datasets"
    }
    
    if current_step in approval_steps:
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button(approval_steps[current_step], type="primary", use_container_width=True):
                # Auto-submit approval
                st.session_state.graph_state["messages"].append({"role": "user", "content": "approve"})
                
                with st.spinner("Processing..."):
                    result = st.session_state.graph.invoke(st.session_state.graph_state)
                    st.session_state.graph_state.update(result)
                    
                st.rerun()
        
        with col2:
            st.caption("_Or type feedback below_")

    # --- User Input ---
    if prompt := st.chat_input("Type your feedback or questions here..."):
        st.session_state.graph_state["messages"].append({"role": "user", "content": prompt})
        if st.session_state.graph_state["current_step"] == "start":
            st.session_state.graph_state["hypothesis"] = prompt
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.spinner("Orchestrator is thinking..."):
            result = st.session_state.graph.invoke(st.session_state.graph_state)
            st.session_state.graph_state.update(result)
            if result.get("messages"):
                new_msg = result["messages"][-1]
                with st.chat_message("assistant"):
                    st.markdown(new_msg["content"])
                    
        st.rerun()

