import pandas as pd
import sys
import os
import time
import ast
import json
import logging
import numpy as np
import re
from tqdm import tqdm
from neo4j import GraphDatabase
import chromadb
from google import genai
from google import genai
from state import GraphState
from observability import trace_tool, trace_node, log_decision

# Add Data_Dictionary to path for imports
sys.path.append(os.path.join(os.getcwd(), 'Data_Dictionary'))

try:
    from config import API_KEYS, MODELS
    import gemini_client
except ImportError:
    print("Warning: Could not import API_KEYS/gemini_client. Using placeholders.")
    API_KEYS = ["YOUR_API_KEY"]
    gemini_client = None

# --- Configuration (Adapted from Notebook) ---
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
CHROMA_DB_PATH = "./chroma_db"
CSV_PATH = os.path.join(os.getcwd(), 'Data_Dictionary', 'data_dictionary_enriched.csv')
DATALAKE_PATH = os.path.join(os.getcwd(), 'Data_Dictionary', 'Datalake')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class RAGRetriever:
    """
    Retrieves relevant metadata context using Neo4j Knowledge Graph and GraphRAG.
    Implements logic from KG_Construction_Complete.ipynb.
    """
    
    def __init__(self):
        self.driver = None
        self.dictionary_df = None

        if NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD:
            try:
                self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
                # Test connection
                self.driver.verify_connectivity()
            except Exception:
                self.driver = None

        if os.path.exists(CSV_PATH):
            try:
                self.dictionary_df = pd.read_csv(CSV_PATH)
            except Exception:
                self.dictionary_df = None

        try:
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        except Exception as e:

            self.chroma_client = None
        
        # Initialize Gemini Client for embeddings (Rotation Logic)
        self.api_keys = API_KEYS
        self.current_key_index = 0
        if self.api_keys and self.api_keys[0] != "YOUR_API_KEY":
            try:
                self.sdk_client = genai.Client(api_key=self.api_keys[self.current_key_index])
            except:
                self.sdk_client = None
        else:
             self.sdk_client = None
        
        # Ensure Graph is built
        if self.driver:
            self._ensure_graph_initialized()

    def close(self):
        if self.driver:
            self.driver.close()

    def _tokenize(self, text):
        return set(re.findall(r"[a-z0-9_]+", str(text).lower()))

    def _score_row_for_kpi(self, kpi_text, row):
        kpi_tokens = self._tokenize(kpi_text)
        if not kpi_tokens:
            return 0.0

        row_text = " ".join([
            str(row.get('Table Name', '')),
            str(row.get('Column Name', '')),
            str(row.get('Table Description', '')),
            str(row.get('Column Description', '')),
        ])
        row_tokens = self._tokenize(row_text)
        overlap = kpi_tokens.intersection(row_tokens)
        return len(overlap) / max(len(kpi_tokens), 1)

    def _get_table_full_schema_from_csv(self, table_name):
        if self.dictionary_df is None:
            return f"Table: {table_name}\n(No schema found)"

        tdf = self.dictionary_df[self.dictionary_df['Table Name'] == table_name]
        if tdf.empty:
            return f"Table: {table_name}\n(No schema found)"

        table_desc = str(tdf.iloc[0].get('Table Description', ''))
        schema = f"Table: {table_name}\nDesc: {table_desc}\nColumns:\n"
        for _, r in tdf.iterrows():
            schema += f" - {r.get('Column Name', '')}: {r.get('Column Description', '')}\n"
        return schema

    def _retrieve_candidates_from_csv(self, kpis, top_n=5):
        """Cloud fallback: retrieve candidates using lexical scoring over data_dictionary_enriched.csv."""
        candidates_map = {}
        if self.dictionary_df is None or self.dictionary_df.empty:
            return candidates_map

        required_cols = {
            'Table Name', 'Column Name', 'Table Description', 'Column Description'
        }
        if not required_cols.issubset(set(self.dictionary_df.columns)):
            return candidates_map

        for kpiname in kpis:
            kpi_key = kpiname.split(":")[0].strip()
            scores = []

            for _, row in self.dictionary_df.iterrows():
                score = self._score_row_for_kpi(kpiname, row)
                if score > 0:
                    scores.append((score, row))

            if not scores:
                candidates_map[kpi_key] = []
                continue

            scores.sort(key=lambda x: x[0], reverse=True)
            selected = []
            seen_tables = set()

            for rank, (score, row) in enumerate(scores, start=1):
                table_name = str(row['Table Name'])
                if table_name in seen_tables:
                    continue
                seen_tables.add(table_name)

                selected.append({
                    "Table": table_name,
                    "Similarity": float(score),
                    "Source": "CSV_Fallback",
                    "Rank": len(selected) + 1,
                    "Full_Schema": self._get_table_full_schema_from_csv(table_name)
                })

                if len(selected) >= top_n:
                    break

            candidates_map[kpi_key] = selected

        return candidates_map

    def _get_embedding_with_rotation(self, text):
        """Fetches embeddings with automatic API key rotation on quota limits."""
        if not self.sdk_client:
            # Fallback to gemini_client if SDK not ready or failover needed
            if gemini_client:
                return gemini_client.get_embedding(text)
            return []

        while self.current_key_index < len(self.api_keys):
            try:
                result = self.sdk_client.models.embed_content(
                    model="gemini-embedding-001", 
                    contents=text
                )
                return result.embeddings[0].values
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "limit" in err_str:
                    self.current_key_index += 1
                    if self.current_key_index < len(self.api_keys):

                        self.sdk_client = genai.Client(api_key=self.api_keys[self.current_key_index])
                        continue

                return [] 
        return []

    def _ensure_graph_initialized(self):
        """Checks if Neo4j graph exists, if not, runs construction pipeline."""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count").single()
                count = result["count"]
                
            if count > 0:

                return


            if not os.path.exists(CSV_PATH):

                return

            df = pd.read_csv(CSV_PATH)


            # 1. Base Graph Construction
            with self.driver.session() as session:
                session.execute_write(self._create_base_graph_tx, df)


            # 2. Semantic Matching & Verification (Simplified for runtime)
            self._run_semantic_matching_and_verification(df)

            # 3. Create Vector Index
            self._create_vector_index(df)

        except Exception as e:
            pass

    def _create_base_graph_tx(self, tx, df):
        # Tables
        unique_tables = df[['Table Name', 'Table Description']].drop_duplicates('Table Name')
        for _, row in unique_tables.iterrows():
            tx.run("MERGE (t:Table {table_name: $name}) SET t.table_desc = $desc", 
                   name=row['Table Name'], desc=row['Table Description'])
        
        # Columns
        for _, row in df.iterrows():
            tx.run("""
            MERGE (c:Column {table_name: $tname, column_name: $cname})
            SET c.col_desc = $cdesc, c.fill_rate = $fr, c.sample_vals = $sv
            WITH c
            MATCH (t:Table {table_name: $tname})
            MERGE (c)-[:`column within table`]->(t)
            """, tname=row['Table Name'], cname=row['Column Name'], 
                cdesc=row['Column Description'], fr=row['Fill Rate'], sv=row['Unique Values'])

    def _run_semantic_matching_and_verification(self, df):
        """Runs the matching logic to create relationships."""

        
        if not self.chroma_client:

            return

        # Chroma Setup
        desc_coll = self.chroma_client.get_or_create_collection("descriptions", metadata={"hnsw:space": "cosine"})
        samp_coll = self.chroma_client.get_or_create_collection("samples", metadata={"hnsw:space": "cosine"})
        
        # Embedding loop (optimized to skip existing)
        existing_ids = set(desc_coll.get()['ids'])
        items_to_embed = [row for _, row in df.iterrows() if f"{row['Table Name']}.{row['Column Name']}" not in existing_ids]
        
        if items_to_embed:

            # Batch this if needed, simple loop for now
            for _, row in tqdm(pd.DataFrame(items_to_embed).iterrows(), total=len(items_to_embed)):
                col_id = f"{row['Table Name']}.{row['Column Name']}"
                desc_text = f"Column: {row['Column Name']}. Description: {row['Column Description']}"
                samp_text = f"Column: {row['Column Name']}. Samples: {row['Unique Values']}"
                
                d_emb = self._get_embedding_with_rotation(desc_text)
                s_emb = self._get_embedding_with_rotation(samp_text)
                
                if d_emb: desc_coll.add(ids=[col_id], embeddings=[d_emb], metadatas=[{"table": row['Table Name']}])
                if s_emb: samp_coll.add(ids=[col_id], embeddings=[s_emb], metadatas=[{"table": row['Table Name']}])

        # Matching Logic
        verified_results = []
        processed_pairs = set()
        
        all_desc = desc_coll.get(include=['embeddings'])
        cached_desc = {id: emb for id, emb in zip(all_desc['ids'], all_desc['embeddings'])}

        # Simplified matching loop (checking top 5 neighbors)

        for _, row_a in tqdm(df.iterrows(), total=len(df)):
            id_a = f"{row_a['Table Name']}.{row_a['Column Name']}"
            if id_a not in cached_desc: continue

            res = desc_coll.query(query_embeddings=[cached_desc[id_a]], n_results=5)
            
            for i in range(len(res['ids'][0])):
                id_b = res['ids'][0][i]
                sim_desc = 1 - res['distances'][0][i]
                
                if id_a == id_b: continue
                pair = tuple(sorted([id_a, id_b]))
                if pair in processed_pairs: continue
                processed_pairs.add(pair)
                
                t1, c1 = id_a.split('.')
                t2, c2 = id_b.split('.')
                if t1 == t2: continue 

                if sim_desc > 0.8:
                    # Verification logic
                    is_valid, _ = self._verify_join(t1, c1, t2, c2)
                    if is_valid:
                        verified_results.append({"source": id_a, "target": id_b, "score": sim_desc * 100})

        # Ingest Relationships
        if verified_results:

            with self.driver.session() as session:
                session.run("""
                UNWIND $batch AS row
                MATCH (c1:Column {table_name: split(row.source, '.')[0], column_name: split(row.source, '.')[1]})
                MATCH (c2:Column {table_name: split(row.target, '.')[0], column_name: split(row.target, '.')[1]})
                MERGE (c1)-[r:potentially_same_column]-(c2)
                SET r.confidence_score = row.score
                """, batch=verified_results)

    def _verify_join(self, t1, col1, t2, col2):
        """Verifies if two columns can actually join by checking local CSV files."""
        file1 = os.path.join(DATALAKE_PATH, f"{t1}.csv")
        file2 = os.path.join(DATALAKE_PATH, f"{t2}.csv")
        
        if not os.path.exists(file1) or not os.path.exists(file2):
            return False, 0
        
        try: 
            df1 = pd.read_csv(file1, usecols=[col1])
            df2 = pd.read_csv(file2, usecols=[col2])
            common = set(df1[col1].dropna().unique()).intersection(set(df2[col2].dropna().unique()))
            return len(common) > 0, len(common)
        except: return False, 0

    def _create_vector_index(self, df):
        """Enriches nodes with embeddings and creates index."""

        with self.driver.session() as session:
            # 1. Enrich Nodes
            for _, r in tqdm(df.iterrows(), total=len(df)):
                content = f"Table: {r['Table Name']}. Column: {r['Column Name']}. Description: {r['Column Description']}. Samples: {r['Unique Values']}"
                emb = self._get_embedding_with_rotation(content)
                if emb:
                    session.run("MATCH (c:Column {table_name: $t, column_name: $n}) SET c.embedding = $emb", 
                                t=r['Table Name'], n=r['Column Name'], emb=emb)
            
            # 2. Create Index
            try: session.run("DROP INDEX column_vector_index")
            except: pass
            
            # Use dummy embedding for dimension
            dummy = self._get_embedding_with_rotation("test")
            if dummy:
                session.run("""
                CREATE VECTOR INDEX column_vector_index FOR (c:Column) ON (c.embedding)
                OPTIONS {indexConfig: {`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}
                """, dim=len(dummy))

    @trace_tool("RAG Candidates")
    def retrieve_candidates_for_kpis(self, kpis):
        """
        Retrieves top candidate tables using GraphRAG (Neo4j Vector Search + Graph Traversal).
        """

        candidates_map = {}

        if not self.driver:
            return self._retrieve_candidates_from_csv(kpis)

        for kpiname in kpis:
            kpi_key = kpiname.split(":")[0].strip()
            embedding = self._get_embedding_with_rotation(kpiname)
            if not embedding or len(embedding) == 0: continue
            
            # Neo4j Vector Query
            cypher_query = """
            CALL db.index.vector.queryNodes('column_vector_index', 10, $emb) 
            YIELD node AS col, score
            MATCH (col)-[:`column within table`]->(t:Table)
            OPTIONAL MATCH (col)-[r:potentially_same_column]-(other:Column)
            RETURN col.column_name as name, col.col_desc as desc, 
                   t.table_name as table, t.table_desc as table_desc, score
            """
            
            try:
                with self.driver.session() as session:
                    results = session.run(cypher_query, emb=embedding)
                    kpi_candidates = []
                    
                    # Deduplicate tables, keep highest score
                    seen_tables = set()
                    
                    for i, res in enumerate(results):
                        table_name = res['table']
                        # Check table description for basic filtering? No, assume all in DB are valid.
                        if table_name in seen_tables: continue
                        seen_tables.add(table_name)
                        
                        full_schema = self._get_table_full_schema(table_name)
                        
                        kpi_candidates.append({
                            "Table": table_name,
                            "Similarity": res['score'],
                            "Source": "GraphRAG",
                            "Rank": i + 1,
                            "Full_Schema": full_schema 
                        })
                    
                    candidates_map[kpi_key] = kpi_candidates
                    
                    if not kpi_candidates:
                         print(f"[WARN] No candidates found for KPI: {kpiname}")

            except Exception as e:
                print(f"[ERROR] RAG Query Failed for {kpiname}: {e}")
                candidates_map[kpi_key] = [{"Error": str(e)}]
                
        return candidates_map

    def _get_table_full_schema(self, table_name):
        """Queries Neo4j to reconstruct the full schema text for a table."""
        if not self.driver:
            return self._get_table_full_schema_from_csv(table_name)
        try:
            with self.driver.session() as session:
                res = session.run("""
                MATCH (t:Table {table_name: $name})<-[:`column within table`]-(c:Column)
                RETURN t.table_desc as tdesc, c.column_name as cname, c.col_desc as cdesc
                """, name=table_name)
                
                records = list(res)
                if not records: return f"Table: {table_name}\n(No schema found)"
                
                desc = records[0]['tdesc'] or ""
                schema = f"Table: {table_name}\nDesc: {desc}\nColumns:\n"
                for r in records:
                    schema += f" - {r['cname']}: {r['cdesc']}\n"
                return schema
        except:
            return f"Table: {table_name}\n(Schema retrieval error)"
                
    @trace_tool("Metadata Identification")
    def identify_required_metadata(self, hypothesis, methodology, kpi_candidates):
        """
        Phase 4.0: Selection of Master Metadata Table (Grounding).
        Uses the detailed prompt structure as requested.
        """

        
        # Flatten schemas for prompt
        seen_tables = set()
        schemas_text = ""
        
        if not kpi_candidates:
            pass
        
        for kpi, candidates in kpi_candidates.items():
            for c in candidates:
                if c['Table'] not in seen_tables:
                    seen_tables.add(c['Table'])
                    schemas_text += c['Full_Schema'] + "\n" + "-"*20 + "\n"

        # The Detailed Prompt (as requested)
        prompt = f"""
        You are a Principal Data Architect. Based on the Hypothesis and Methodology, identify the MINIMUM set of tables and columns required to calculated the KPIs.
        
        Hypothesis: "{hypothesis}"
        Methodology:
        {methodology}
        
        Candidate Table Schemas (Retrieved via GraphRAG):
        {schemas_text}
        
        Task:
        Return a JSON nested list representing the 'Master Metadata Table'.
        Format:
        [
            ["S.No", "Table", "KPIs", "Columns", "Reasoning"],
            [1, "TABLE_A", "Churn Rate", "cust_id, churn_flag", "Primary fact table for churn."],
            ...
        ]
        
        Constraints:
        - Identify the minimum necessary tables to avoid data overhead.
        - Ensure all columns exist in the provided schemas.
        - Return ONLY the JSON list.
        """
        
        if gemini_client:
            # Use gemini_client.call_gemini if available
            try:
                response = gemini_client.call_gemini(prompt)
                clean_json = response.replace("```json", "").replace("```", "").strip()
                # Attempt to fix common JSON issues if simple load fails?
                try:
                    metadata_list = json.loads(clean_json)
                    return metadata_list
                except:
                     # Try to eval if it's a list string
                     try:
                         metadata_list = ast.literal_eval(clean_json)
                         if isinstance(metadata_list, list): return metadata_list
                     except:
                         pass
                return []
            except Exception as e:

                return []
        return []

    @trace_tool("Initialize KG")
    def tool_initialize_kg(self):
        """Tool to ensure Knowledge Graph is built."""
        self._ensure_graph_initialized()
        return "Knowledge Graph and Vector Index are ready."

    @trace_tool("RAG Candidates")
    def tool_retrieve_candidates(self, kpis):
        """Tool to retrieve candidate tables for KPIs."""
        return self.retrieve_candidates_for_kpis(kpis)

    @trace_tool("Identify Metadata")
    def tool_identify_metadata(self, hypothesis, methodology, candidates):
        """Tool to identify metadata."""
        return self.identify_required_metadata(hypothesis, methodology, candidates)

    @trace_tool("Brainstorm Metadata")
    def tool_brainstorm_metadata(self, hypothesis, methodology, candidates, feedback):
        """Tool to brainstorm or refine metadata selection based on feedback."""
        prompt = f"""
        You are a Principal Data Architect. The user has feedback on the initial metadata selection.
        
        Hypothesis: "{hypothesis}"
        Methodology:
        {methodology}
        
        Current Candidate Tables:
        {json.dumps(candidates, indent=2) if candidates else "None"}
        
        User Feedback:
        "{feedback}"
        
        Task:
        1. Analyze the feedback.
        2. If the user suggests new tables not in the list, explain if they are missing from the KG or suggest semantic equivalents.
        3. If the user wants to narrow down, refine the list.
        4. Return a JSON nested list representing the 'Master Metadata Table'.
        
        Format:
        [
            ["S.No", "Table", "KPIs", "Columns", "Reasoning"],
            ...
        ]
        """
        if gemini_client:
            try:
                response = gemini_client.call_gemini(prompt)
                clean_json = response.replace("```json", "").replace("```", "").strip()
                try:
                    return json.loads(clean_json)
                except:
                    return ast.literal_eval(clean_json)
            except:
                return []
        return []

    @trace_tool("RAG Router")
    def _rag_router(self, state_context):
        """
        Internal Router to decide RAG actions.
        Decides between: initialize_kg, retrieve_candidates, identify_metadata, brainstorm_metadata.
        """
        prompt = f"""
        You are the RAG Router for the HyTE Deep Agent.
        Your goal is to ensure we have the correct logical metadata to support the hypothesis.
        
        Current Context:
        {json.dumps(state_context, indent=2)}
        
        Available Tools:
        - "initialize_kg": strict first step. Checks/builds the graph.
        - "retrieve_candidates": Use when we have KPIs but no candidates yet.
        - "identify_metadata": Use when we have candidates but need to select the final list.
        - "brainstorm_metadata": Use if user provides feedback or rejects the initial list.
        
        Logic:
        1. If 'kg_ready' is False, call 'initialize_kg'.
        2. If 'candidates' is empty, call 'retrieve_candidates'.
        3. If 'metadata' is empty, call 'identify_metadata'.
        4. If 'feedback' is present, call 'brainstorm_metadata'.
        
        Return ONLY a JSON object:
        {{
            "thought_process": "Reasoning...",
            "action": "call_tool",
            "tool": "tool_name",
            "status_message": "User facing update."
        }}
        """
        if gemini_client:
            try:
                res = gemini_client.call_gemini(prompt)
                return json.loads(res.replace("```json", "").replace("```", "").strip())
            except:
                # Fallback default logic
                return {"action": "call_tool", "tool": "retrieve_candidates", "status_message": "Retrieving data..."}
        return {"action": "call_tool", "tool": "retrieve_candidates", "status_message": "Retrieving data..."}

    @trace_node("RAG Agent")
    def run(self, state: GraphState):
        """Entry point for the RAG step."""
        hypothesis = state.get("hypothesis", "")
        methodology = state.get("initial_strategy", "")
        kpis = state.get("kpi_list", [])
        
        # Fallback 1: Extract KPIs from initial_strategy_dict["kpis"] (structured dict)
        if not kpis and state.get("initial_strategy_dict"):
            kpis_dict = state["initial_strategy_dict"].get("kpis", {})
            if isinstance(kpis_dict, str):
                try:
                    kpis_dict = json.loads(kpis_dict.replace("```json", "").replace("```", "").strip())
                except: kpis_dict = {}
            if isinstance(kpis_dict, dict) and kpis_dict:
                kpis = list(kpis_dict.keys())
                print(f"[INFO] RAG: Extracted {len(kpis)} KPIs from initial_strategy_dict['kpis']: {kpis}")
        
        # Fallback 2: Extract KPIs from initial_strategy_dict["meth_kpis"] (raw text from parse_sections)
        if not kpis and state.get("initial_strategy_dict"):
            meth_kpis_text = state["initial_strategy_dict"].get("meth_kpis", "")
            if meth_kpis_text:
                # Try parsing as JSON first
                try:
                    kpis_dict = json.loads(meth_kpis_text.replace("```json", "").replace("```", "").strip())
                    if isinstance(kpis_dict, dict):
                        kpis = list(kpis_dict.keys())
                except:
                    # Extract KPI names from bullet points like "- **KPI Name**: description"
                    import re
                    kpi_matches = re.findall(r'\*\*(.+?)\*\*', meth_kpis_text)
                    if kpi_matches:
                        kpis = [k.strip().rstrip(':') for k in kpi_matches]
                if kpis:
                    print(f"[INFO] RAG: Extracted {len(kpis)} KPIs from meth_kpis text: {kpis}")

        # Fallback 3: Try parsing KPIs from raw methodology text
        if not kpis and methodology:
            import re
            kpi_section = re.search(r'2\.\s*\*\*KPIs\*\*:(.*?)(?=3\.\s*\*\*|$)', methodology, re.DOTALL)
            if kpi_section:
                kpi_matches = re.findall(r'\*\*(.+?)\*\*', kpi_section.group(1))
                if kpi_matches:
                    kpis = [k.strip().rstrip(':') for k in kpi_matches]
                    print(f"[INFO] RAG: Extracted {len(kpis)} KPIs from raw methodology text: {kpis}")
        
        if kpis:
            print(f"[INFO] RAG: Processing {len(kpis)} KPIs: {kpis}")
        
        # Validation: KPIs must exist
        if not kpis:
            print("[ERROR] RAG: No KPIs found in state. Cannot retrieve candidates.")
            return {
                "messages": [{"role": "assistant", "content": "[ERROR] **RAG Error**: No KPIs were found in the strategy. I cannot retrieve data without valid KPIs."}],
                "current_step": "rag_failed"
            }

        # Internal State construction
        internal_memory = state.get("internal_memory", {})
        rag_memory = internal_memory.get("rag_memory", {})
        
        context = {
            "hypothesis": hypothesis,
            "kpis_count": len(kpis),
            "kg_ready": rag_memory.get("kg_initialized", False),
            "has_candidates": bool(rag_memory.get("candidates")),
            "has_metadata": bool(state.get("metadata_context")),
            "feedback": state.get("latest_feedback") if state.get("current_step") == "refine_rag" else None
        }
        
        # Router Decision
        decision = self._rag_router(context)
        tool = decision.get("tool")
        message = decision.get("status_message", "Processing data request...")
        
        log_decision("RAG Router", tool, {
            "thought": decision.get("thought_process", ""),
            "kpis_count": len(kpis),
            "kg_ready": context["kg_ready"],
            "has_candidates": context["has_candidates"]
        })
        
        updates = {
            "messages": [{"role": "assistant", "content": message}]
        }
        
        # Execute Tool
        if tool == "initialize_kg":
            self.tool_initialize_kg()
            rag_memory["kg_initialized"] = True
            updates["internal_memory"] = {**internal_memory, "rag_memory": rag_memory}
            # Recursively continue or return to let graph loop? 
            # For simplicity, we'll return and let the next loop handle it, 
            # BUT since we're in a node, we might want to chain. 
            # Let's simple chain logic here for efficiency:
            context["kg_ready"] = True
            tool = "retrieve_candidates" # Auto-proceed
            
        if tool == "retrieve_candidates":
            candidates = self.tool_retrieve_candidates(kpis)
            rag_memory["candidates"] = candidates
            updates["internal_memory"] = {**internal_memory, "rag_memory": rag_memory}
            
            # Validation: Did we find anything?
            has_results = any(len(v) > 0 for v in candidates.values())
            if not has_results:
                 print(f"[WARN] RAG: No candidates found for KPIs: {kpis}")
                 return {
                    "messages": [{"role": "assistant", "content": f"[WARNING] **RAG Warning**: I searched the Knowledge Graph for {kpis} but found **no relevant tables**. Please refine the KPIs or check if the data exists."}],
                    "current_step": "rag_failed"
                 }

            # Auto-proceed to identify
            context["has_candidates"] = True
            if not context.get("feedback"):
                tool = "identify_metadata"

        if tool == "identify_metadata":
            candidates = rag_memory.get("candidates", {})
            if not candidates:
                 print("[ERROR] RAG: No candidates available for metadata identification.")
                 return {
                    "messages": [{"role": "assistant", "content": "[ERROR] **RAG Error**: No candidates available to identify metadata."}],
                    "current_step": "rag_failed"
                 }
            metadata_list = self.tool_identify_metadata(hypothesis, methodology, candidates)
            
            # Format output
            def to_markdown(ctx_list):
                if not ctx_list: return "No metadata identified."
                val = "| S.No | Table | KPIs | Columns | Reasoning |\n|---|---|---|---|---|\n"
                for row in ctx_list:
                    if len(row) >= 5:
                        val += f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |\n"
                return val
            
            updates["metadata_context"] = to_markdown(metadata_list)
            updates["current_step"] = "rag_generated"
            updates["messages"] = [{"role": "assistant", "content": f"## 🔍 Data Discovery Complete\n\nI've identified the following relevant tables and columns based on the strategy:\n\n{updates['metadata_context']}\n\nProceed to design the data-aware methodology?"}]
            
        elif tool == "brainstorm_metadata":
            candidates = rag_memory.get("candidates", {})
            metadata_list = self.tool_brainstorm_metadata(hypothesis, methodology, candidates, context["feedback"])
             # Format output (duplicated logic, could be helper)
            def to_markdown(ctx_list):
                if not ctx_list: return "No metadata identified."
                val = "| S.No | Table | KPIs | Columns | Reasoning |\n|---|---|---|---|---|\n"
                for row in ctx_list:
                    if len(row) >= 5:
                        val += f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |\n"
                return val
            
            updates["metadata_context"] = to_markdown(metadata_list)
            updates["current_step"] = "rag_generated"
            updates["messages"] = [{"role": "assistant", "content": f"## 🔄 Metadata Refined\n\nBased on your feedback, I've updated the data selection:\n\n{updates['metadata_context']}"}]

        return updates

if __name__ == "__main__":
    retriever = RAGRetriever()

    kpis = ["Churn Rate"]
    candidates = retriever.retrieve_candidates_for_kpis(kpis)

