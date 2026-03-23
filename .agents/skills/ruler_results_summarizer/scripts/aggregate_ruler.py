import os
import glob
import pandas as pd
import argparse
import re

def format_method_name(raw_name, model_filter=None):
    parts = raw_name.split('_')
    method = parts[0].upper()
    
    params = []
    models = []
    for p in parts[1:]:
        match = re.match(r'^([a-zA-Z]+)([\d\.]+)$', p)
        if match:
            k, v = match.groups()
            params.append(f"{k}={v}")
        else:
            models.append(p)
            
    param_str = ", ".join(params)
    model_str = "_".join(models)
    
    if param_str:
        formatted = f"{method}({param_str})"
    else:
        formatted = f"{method}"
        
    if model_filter and model_filter == model_str:
        pass  # Omit appending model name if it matches filter exactly
    elif model_str:
        formatted += f" [{model_str}]"
        
    return formatted, method, model_str

def main():
    parser = argparse.ArgumentParser(description="Aggregate RULER benchmark results")
    parser.add_argument("--root_dir", type=str, default="eval/RULER/scripts/benchmark_root", help="Root directory containing benchmark results")
    parser.add_argument("--output_path", type=str, default="results/ruler/ruler_summary.csv", help="Path to save the summary CSV")
    parser.add_argument("--model", type=str, default=None, help="Filter by exact model name (e.g. llama3.1-8b-nanovllm)")
    parser.add_argument("--method", type=str, default=None, help="Filter by exact method abbreviation (e.g. blasst, compass)")
    
    args = parser.parse_args()
    
    pattern = os.path.join(args.root_dir, "*", "synthetic", "*", "pred", "summary.csv")
    csv_files = glob.glob(pattern)
    
    data = []
    
    for file in csv_files:
        try:
            parts = file.split(os.sep)
            pred_idx = parts.index("pred")
            length = parts[pred_idx - 1]
            raw_method_model = parts[pred_idx - 3]
        except ValueError:
            print(f"Skipping {file}: unable to parse path structure.")
            continue
            
        formatted_name, method_base, model_base = format_method_name(raw_method_model, args.model)
        
        # Apply filters
        if args.model and args.model != model_base:
            continue
        if args.method and args.method.lower() != method_base.lower():
            continue
            
        try:
            df = pd.read_csv(file, header=None)
            score_row_idx = df[df[0] == "Score"].index
            if len(score_row_idx) > 0:
                score_row = df.iloc[score_row_idx[0], 1:]
                scores = pd.to_numeric(score_row, errors='coerce')
                avg_score = scores.mean()
                
                if not pd.isna(avg_score):
                    data.append({
                        "Method_Model": formatted_name,
                        "Length": length,
                        "Avg_Score": avg_score
                    })
            else:
                print(f"Skipping {file}: 'Score' row not found.")
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    if not data:
        print("No valid data found after filtering.")
        return
        
    df_results = pd.DataFrame(data)
    
    def format_len(x):
        try:
            val = int(x)
            if val >= 1024:
                return f"{val//1024}k"
            return str(val)
        except:
            return str(x)
            
    pivot_df = df_results.pivot_table(index="Method_Model", columns="Length", values="Avg_Score", aggfunc='mean')
    
    cols_sorted_by_len = sorted(list(pivot_df.columns), key=lambda x: int(x) if str(x).isdigit() else float('inf'))
    pivot_df = pivot_df[cols_sorted_by_len]
    
    pivot_df["Avg."] = pivot_df.mean(axis=1)
    
    cols_formatted = [format_len(c) if c != "Avg." else c for c in pivot_df.columns]
    pivot_df.columns = cols_formatted
    
    formatted_df = pivot_df.map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    formatted_df.to_csv(args.output_path)
    print(f"Summary saved to {args.output_path}")
    
    print("\n--- RULER Results Summary ---")
    try:
        print(formatted_df.to_markdown())
    except ImportError:
        print(formatted_df)

if __name__ == "__main__":
    main()
