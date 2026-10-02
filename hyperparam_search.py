import subprocess
import json
import os
import time

def run_experiment(mode, lambda_ditto, mcfl_lr, ppfl_split):
    cmd = [
        "python", "main_phase2.py",
        "--mode", mode,
        "--lambda_ditto", str(lambda_ditto),
        "--mcfl_lr", str(mcfl_lr),
        "--rounds", "10",  # 10 rounds for faster search
        "--epochs", "2"
    ]
    if ppfl_split:
        cmd.append("--ppfl_split")
        
    print(f"Running: {' '.join(cmd)}")
    
    # Run and wait
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        print(f"Error running experiment: {result.stderr}")
        return None
        
    # Read the generated summary
    summary_path = os.path.join("results", "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            return json.load(f)
    return None

def main():
    print("Starting Hyperparameter Search...")
    print("Running for 10 rounds to quickly compare effects.\n")
    
    experiments = [
        # Baseline (No Personalization)
        {"mode": "standard", "lambda_ditto": 0.0, "mcfl_lr": 0.1, "ppfl_split": False, "name": "Baseline (Standard)"},
        
        # Tuning lambda_ditto (Proximal fine-tuning strength)
        {"mode": "full", "lambda_ditto": 0.01, "mcfl_lr": 0.1, "ppfl_split": True, "name": "Full (lambda=0.01, lr=0.1)"},
        {"mode": "full", "lambda_ditto": 0.1,  "mcfl_lr": 0.1, "ppfl_split": True, "name": "Full (lambda=0.1, lr=0.1)"},
        {"mode": "full", "lambda_ditto": 1.0,  "mcfl_lr": 0.1, "ppfl_split": True, "name": "Full (lambda=1.0, lr=0.1)"},
        
        # Tuning mcfl_lr (Mixture learning rate)
        {"mode": "full", "lambda_ditto": 0.1,  "mcfl_lr": 0.5, "ppfl_split": True, "name": "Full (lambda=0.1, lr=0.5)"}
    ]
    
    results_list = []
    
    for exp in experiments:
        t0 = time.time()
        res = run_experiment(exp["mode"], exp["lambda_ditto"], exp["mcfl_lr"], exp["ppfl_split"])
        dur = time.time() - t0
        if res:
            res["name"] = exp["name"]
            res["time"] = f"{dur:.1f}s"
            results_list.append(res)
            
    # Print Markdown Table
    print("\n" + "="*90)
    print("HYPERPARAMETER SEARCH RESULTS (10 Rounds)")
    print("="*90)
    print(f"| {'Experiment / Hyperparams':<30} | {'Global AUC':<10} | {'Mean AUROC':<10} | {'Worst AUROC':<11} | {'Time':<6} |")
    print(f"| {'-'*30} | {'-'*10} | {'-'*10} | {'-'*11} | {'-'*6} |")
    
    for r in results_list:
        name = r["name"]
        g_auc = f"{r['final_global_auc']:.4f}"
        m_auc = f"{r['final_avg_local_auc']:.4f}"
        w_auc = f"{r.get('final_worst_auc', 0.0):.4f}" # Will need to extract worst auc somehow
        # Actually, summary.json doesn't have worst_auc right now, let's update main_phase2 to save it!
        
        # Fallback if worst_auc isn't in summary
        print(f"| {name:<30} | {g_auc:<10} | {m_auc:<10} | {w_auc:<11} | {r['time']:<6} |")
        
if __name__ == "__main__":
    main()
