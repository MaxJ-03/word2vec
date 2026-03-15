import os
import json

def generate_markdown_report_from_log():
    """
    Reads the stored metrics log and generates a comprehensive Markdown comparison table.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(base_dir, 'reports', 'metrics_log.json')
    report_path = os.path.join(base_dir, 'reports', 'benchmark_comparison.md')

    if not os.path.exists(log_path):
        print("\n[!] No metrics log found. Please run evaluations first.")
        return

    with open(log_path, 'r', encoding='utf-8') as f:
        try:
            log_data = json.load(f)
        except json.JSONDecodeError:
            print("\n[!] Metrics log is corrupted or empty.")
            return

    if not log_data:
        print("\n[!] Metrics log is empty.")
        return

    # Construct the Markdown table headers and populate the rows.
    md_content = "# Word2Vec Architecture Comparison\n\n"
    md_content += "| Model Architecture | Semantic Accuracy | Syntactic Accuracy | Total Accuracy |\n"
    md_content += "|---|---|---|---|\n"
    
    for res in log_data:
        md_content += f"| `{res['model_name']}` | {res['semantic_accuracy']:.2f}% | {res['syntactic_accuracy']:.2f}% | {res['total_accuracy']:.2f}% |\n"
        
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    print(f"\nMarkdown report generated successfully at: {report_path}")

if __name__ == '__main__':
    generate_markdown_report_from_log()