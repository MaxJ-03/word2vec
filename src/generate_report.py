import os
import json
from evaluate_saved_embed import LoadedEmbeddings, save_metrics_to_log
from evaluation import evaluate_analogies

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
    # Scans the embeddings directory, automatically evaluates any models missing from the log,
    # generates a comprehensive Markdown comparison table sorted by accuracy, and updates the README.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    embeddings_dir = os.path.join(base_dir, 'embeddings')
    testsets_dir = os.path.join(base_dir, 'testsets')
    reports_dir = os.path.join(base_dir, 'reports')
    
    os.makedirs(reports_dir, exist_ok=True)
    
    log_path = os.path.join(reports_dir, 'metrics_log.json')
    report_path = os.path.join(reports_dir, 'benchmark_comparison.md')
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

    md_table = "| Model Architecture | Semantic Accuracy | Syntactic Accuracy | Total Accuracy |\n"
    md_table += "|---|---|---|---|\n"
    
    for res in log_data_sorted:
        md_table += f"| `{res['model_name']}` | {res['semantic_accuracy']:.2f}% | {res['syntactic_accuracy']:.2f}% | {res['total_accuracy']:.2f}% |\n"
        
    md_content = "# Word2Vec Architecture Comparison\n\n" + md_table
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    print(f"Markdown report generated successfully at: {report_path}")

    update_readme_with_table(md_table, base_dir)

if __name__ == '__main__':
    generate_markdown_report_from_log()