import pandas as pd
import os
import json
from state import GraphState
from observability import trace_node

class MergeAgent:
    """
    Agent responsible for intelligently merging per-KPI master datasets.
    Uses feasibility analysis to identify compatible datasets and merge them.
    """
    
    @trace_node("Merge")
    def run(self, state: GraphState):
        """Merge compatible KPI datasets based on granularity analysis."""
        granularity_analysis = state.get("granularity_analysis", {})
        kpi_datasets = state.get("kpi_datasets", {})
        

        
        if not granularity_analysis or not kpi_datasets:
            return {
                "merge_report": "No datasets to merge or no feasibility analysis available.",
                "merged_datasets": {},
                "current_step": "merge_completed",
                "messages": [{
                    "role": "assistant",
                    "content": "⚠️ Merge skipped: No datasets or feasibility analysis available."
                }]
            }
        
        mergeable_groups = granularity_analysis.get("mergeable_groups", [])
        joining_keys = granularity_analysis.get("joining_keys", {})
        
        merged_datasets = {}
        merge_report_lines = []
        
        merge_report_lines.append("# Dataset Merge Report\n")
        merge_report_lines.append(f"**Total KPI Groups**: {len(mergeable_groups)}\n\n")
        
        for group in mergeable_groups:
            group_id = group.get("group_id")
            kpis = group.get("kpis", [])
            granularity = group.get("granularity", "unknown")
            
            merge_report_lines.append(f"## Group {group_id}: {granularity}\n")
            merge_report_lines.append(f"**KPIs**: {', '.join(kpis)}\n\n")
            
            # If only one KPI in group, skip merge
            if len(kpis) <= 1:
                merge_report_lines.append(f"✅ **Action**: Kept separate (single KPI group)\n\n")
                continue
            
            # Get joining keys for this group
            group_keys = joining_keys.get(f"group_{group_id}", [])
            
            if not group_keys:
                merge_report_lines.append(f"⚠️ **Action**: Kept separate (no joining keys identified)\n\n")
                continue
            
            # Attempt to merge datasets
            try:
                merged_df = self._merge_kpi_datasets(kpis, kpi_datasets, group_keys)
                
                # Save merged dataset
                merged_filename = f"master_dataset_merged_group_{group_id}.csv"
                merged_df.to_csv(merged_filename, index=False)
                merged_datasets[f"group_{group_id}"] = merged_filename
                
                merge_report_lines.append(f"✅ **Action**: Merged successfully\n")
                merge_report_lines.append(f"- **Joining Keys**: {', '.join(group_keys)}\n")
                merge_report_lines.append(f"- **Output File**: `{merged_filename}`\n")
                merge_report_lines.append(f"- **Shape**: {merged_df.shape}\n\n")
                

                
            except Exception as e:
                merge_report_lines.append(f"❌ **Action**: Merge failed\n")
                merge_report_lines.append(f"- **Error**: {str(e)}\n")
                merge_report_lines.append(f"- Datasets kept separate\n\n")

        
        merge_report = "".join(merge_report_lines)
        
        return {
            "merge_report": merge_report,
            "merged_datasets": merged_datasets,
            "current_step": "merge_completed",
            "messages": [{
                "role": "assistant",
                "content": f"## 🔗 Dataset Merge Complete\n\n{merge_report}\n\n---\n\nProceed to analysis?"
            }]
        }
    
    def _merge_kpi_datasets(self, kpis, kpi_datasets, joining_keys):
        """
        Merge multiple KPI datasets using specified joining keys.
        """
        dataframes = []
        
        for kpi in kpis:
            csv_filename = kpi_datasets.get(kpi)
            if not csv_filename or not os.path.exists(csv_filename):
                raise FileNotFoundError(f"Dataset for {kpi} not found: {csv_filename}")
            
            df = pd.read_csv(csv_filename)
            df.columns = df.columns.str.strip()
            
            # Add KPI identifier column (useful for tracking source)
            df[f'source_kpi'] = kpi
            
            dataframes.append(df)
        
        if not dataframes:
            raise ValueError("No dataframes to merge")
        
        # Start with first dataframe
        merged_df = dataframes[0]
        
        # Sequentially merge remaining dataframes
        for df in dataframes[1:]:
            # Find common keys that exist in both dataframes
            common_keys = [key for key in joining_keys if key in merged_df.columns and key in df.columns]
            
            if not common_keys:
                # If no common keys, try outer concat (less ideal)

                merged_df = pd.concat([merged_df, df], axis=0, ignore_index=True)
            else:
                # Use proper merge
                merged_df = merged_df.merge(df, on=common_keys, how='outer', suffixes=('', '_dup'))
                
                # Remove duplicate columns (keep original)
                dup_cols = [col for col in merged_df.columns if col.endswith('_dup')]
                merged_df = merged_df.drop(columns=dup_cols)
        
        return merged_df
