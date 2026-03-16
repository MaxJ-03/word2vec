import os
import json
import re
from evaluate_saved_embed import LoadedEmbeddings, save_metrics_to_log
from evaluation import evaluate_analogies

def parse_filename(filename):
    # Extracts hyperparameters and dataset information from the saved embedding filename.
    hyperparams_pattern = r'_ep(\d+)_lr([0-9.]+)_dim(\d+)_w(\d+)(?:_ns(\d+))?'
    match = re.search(hyperparams_pattern, filename)
    
    if match:
        epochs = match.group(1)
        lr = match.group(2)
        dim = match.group(3)
        window = match.group(4)
        ns = match.group(5) if match.group(5) else "-"
        
        prefix = filename[:match.start()]
        
        architectures = [
            "Standard_CBOW", 
            "CBOW_with_Hierarchical_Softmax", 
            "CBOW_with_Negative_Sampling",
            "Standard_Skip-Gram", 
            "Skip-Gram_with_Hierarchical_Softmax", 
            "Skip-Gram_with_Negative_Sampling"
        ]
        
        dataset = "Unknown"
        architecture = "Unknown"
        
        for arch in architectures:
            if arch in prefix:
                architecture = arch.replace("_", " ")
                dataset = prefix.replace(f"_{arch}", "")
                break
                
        return dataset, architecture, epochs, lr, dim, window, ns
        
    return "Unknown", filename, "-", "-", "-", "-", "-"

def update_readme_with_table(md_table_content, base_dir):
    # Updates the README.md file by injecting the markdown table between predefined markers.
    readme_path = os.path.join(base_dir, 'README.md')
    
    if not os.path.exists(readme_path):
        print(f"README.md not found at {readme_path}. Skipping README update.")
        return

    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    start_marker = "<" + "!-- BENCHMARK_TABLE_START --" + ">"
    end_marker = "<" + "!-- BENCHMARK_TABLE_END --" + ">"

    if start_marker in content and end_marker in content:
        before_table = content.split(start_marker)[0]
        after_table = content.split(end_marker)[1]
        
        new_content = f"{before_table}{start_marker}\n{md_table_content}\n{end_marker}{after_table}"
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("README.md successfully updated with the latest benchmark table.")
    else:
        print("Markers not found in README.md. Please add the start and end markers.")

def generate_markdown_report_from_log():
    # Scans the embeddings directory, evaluates any models missing from the log,
    # generates a comprehensive Markdown comparison table sorted by accuracy, and updates the README.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    embeddings_dir = os.path.join(base_dir, 'embeddings')
    testsets_dir = os.path.join(base_dir, 'testsets')
    reports_dir = os.path.join(base_dir, 'reports')
    
    os.makedirs(reports_dir, exist_ok=True)
    
    log_path = os.path.join(reports_dir, 'metrics_log.json')
    test_file_path = os.path.join(testsets_dir, 'word-test.v1.txt')

    if not os.path.exists(test_file_path):
        print(f"Could not find analogy test file at: {test_file_path}")
        return

    if not os.path.exists(embeddings_dir):
        print("No 'embeddings' directory found. Please train some models first.")
        return

    embed_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.txt')]
    if not embed_files:
        print("No saved embeddings found to evaluate.")
        return

    log_data = []
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                log_data = []

    evaluated_models = {entry['model_name'] for entry in log_data}

    for file in embed_files:
        model_name = file.replace('.txt', '')
        
        if model_name not in evaluated_models:
            print(f"Model '{model_name}' not found in logs. Evaluating now...")
            filepath = os.path.join(embeddings_dir, file)
            
            model = LoadedEmbeddings(filepath)
            metrics = evaluate_analogies(model, test_file_path)
            
            if metrics:
                save_metrics_to_log(metrics, base_dir)
                log_data.append(metrics)
                evaluated_models.add(model_name)

    if not log_data:
        print("No metrics available to generate a report.")
        return

    log_data_sorted = sorted(log_data, key=lambda x: x['total_accuracy'], reverse=True)

    headers = [
        "Dataset", "Architecture", "Epochs", "LR", "Dim", "Window", "NS",
        "Sem. Eval", "Sem. Acc", "Syn. Eval", "Syn. Acc", "Skipped", "Total Acc"
    ]
    
    md_table = "| " + " | ".join(headers) + " |\n"
    md_table += "|" + "|".join(["---"] * len(headers)) + "|\n"
    
    for res in log_data_sorted:
        model_name = res.get('model_name', 'Unknown')
        dataset, arch, epochs, lr, dim, window, ns = parse_filename(model_name)
        
        sem_eval = res.get('semantic_evaluated', 'N/A')
        syn_eval = res.get('syntactic_evaluated', 'N/A')
        sem_acc = f"{res.get('semantic_accuracy', 0):.2f}%"
        syn_acc = f"{res.get('syntactic_accuracy', 0):.2f}%"
        skipped = res.get('skipped', 'N/A')
        tot_acc = f"{res.get('total_accuracy', 0):.2f}%"
        
        row = [
            dataset, arch, epochs, lr, dim, window, ns,
            str(sem_eval), sem_acc, str(syn_eval), syn_acc, str(skipped), tot_acc
        ]
        md_table += "| " + " | ".join(row) + " |\n"

    update_readme_with_table(md_table, base_dir)

if __name__ == '__main__':
    generate_markdown_report_from_log()