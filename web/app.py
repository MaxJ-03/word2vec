import os
import sys
import threading
import re
import json
import numpy as np
from flask import Flask, render_template, jsonify, request

# Add the src directory to the system path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, 'src')
sys.path.append(src_dir)

# Import core modules
from evaluate_saved_embed import LoadedEmbeddings

# Update 'run_training' and 'train_model' to match your actual filenames
try:
    from run_training import train_model 
except ImportError:
    # Fallback for local development or different naming conventions
    def train_model(*args, **kwargs):
        print("Training module not found. Check imports in app.py.")

app = Flask(__name__)

EMBEDDINGS_DIR = os.path.join(base_dir, 'embeddings')
TRAINSETS_DIR = os.path.join(base_dir, 'trainsets')

current_model = None
training_status = {"active": False, "message": "Idle"}

def parse_filename(filename):
    hyperparams_pattern = r'_ep(\d+)_lr([0-9.]+)_dim(\d+)_w(\d+)(?:_ns(\d+))?'
    match = re.search(hyperparams_pattern, filename)
    if match:
        epochs, lr, dim, window = match.groups()[:4]
        ns = match.group(5) if match.group(5) else "-"
        prefix = filename[:match.start()]
        architectures = [
            "Standard_CBOW", "CBOW_with_Hierarchical_Softmax", "CBOW_with_Negative_Sampling",
            "Standard_Skip-Gram", "Skip-Gram_with_Hierarchical_Softmax", "Skip-Gram_with_Negative_Sampling"
        ]
        dataset, architecture = "Unknown", "Unknown"
        for arch in architectures:
            if arch in prefix:
                architecture = arch.replace("_", " ")
                dataset = prefix.replace(f"_{arch}", "")
                break
        return f"{dataset} | {architecture} | Ep:{epochs} | LR:{lr} | Dim:{dim} | Win:{window} | NS:{ns}"
    return filename 

def perform_pca(vectors, n_components=2):
    mean = np.mean(vectors, axis=0)
    centered_data = vectors - mean
    U, S, Vh = np.linalg.svd(centered_data, full_matrices=False)
    return centered_data @ Vh[:n_components].T

def background_training(params):
    global training_status
    training_status["active"] = True
    training_status["message"] = f"Training on {params['dataset']}..."
    try:
        # Executes the specific training logic from src/
        train_model(
            dataset_path=os.path.join(TRAINSETS_DIR, params['dataset']),
            architectures=params['architectures'],
            epochs=int(params['epochs']),
            lr=float(params['lr']),
            dim=int(params['dim']),
            window=int(params['window']),
            ns=int(params['ns'])
        )
        training_status["message"] = "Training Complete!"
    except Exception as e:
        training_status["message"] = f"Error: {str(e)}"
    finally:
        training_status["active"] = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models')
def get_models():
    if not os.path.exists(EMBEDDINGS_DIR): return jsonify([])
    return jsonify([{"filename": f, "display_name": parse_filename(f)} 
                    for f in os.listdir(EMBEDDINGS_DIR) if f.endswith('.txt')])

@app.route('/api/datasets')
def get_datasets():
    if not os.path.exists(TRAINSETS_DIR): return jsonify([])
    return jsonify([f for f in os.listdir(TRAINSETS_DIR) if f.endswith('.txt')])

@app.route('/api/leaderboard')
def get_leaderboard():
    log_path = os.path.join(base_dir, 'reports', 'metrics_log.json')
    if not os.path.exists(log_path): return jsonify([])
    try:
        with open(log_path, 'r') as f:
            data = json.load(f)
        for row in data:
            row['display_name'] = parse_filename(row.get('model_name', ''))
        return jsonify(sorted(data, key=lambda x: x.get('total_accuracy', 0), reverse=True))
    except: return jsonify([])

@app.route('/api/load', methods=['POST'])
def load_model():
    global current_model
    data = request.json
    try:
        current_model = LoadedEmbeddings(os.path.join(EMBEDDINGS_DIR, data['filename']))
        W1 = current_model.W1
        current_model.W1_normalized = W1 / (np.linalg.norm(W1, axis=1, keepdims=True) + 1e-9)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/similar', methods=['POST'])
def get_similar():
    if not current_model: return jsonify({"error": "No model loaded"}), 400
    word = request.json.get('word', '').lower()
    vocab = current_model.vocab if hasattr(current_model, 'vocab') else current_model.data_processing.vocabulary
    if word not in vocab: return jsonify({"error": "Word not found"}), 404
    w2id = current_model.word_to_id if hasattr(current_model, 'word_to_id') else current_model.data_processing.word_to_id
    id2w = current_model.id_to_word if hasattr(current_model, 'id_to_word') else current_model.data_processing.id_to_word
    idx = w2id[word]
    sims = np.dot(current_model.W1_normalized, current_model.W1_normalized[idx])
    best = np.argsort(sims)[::-1][1:6]
    return jsonify({"neighbors": [{"word": id2w[i]} for i in best]})

@app.route('/api/analogy', methods=['POST'])
def calculate_analogy():
    if not current_model: return jsonify({"error": "No model loaded"}), 400
    d = request.json
    words = [d['word_a'].lower(), d['word_b'].lower(), d['word_c'].lower()]
    vocab = current_model.vocab if hasattr(current_model, 'vocab') else current_model.data_processing.vocabulary
    w2id = current_model.word_to_id if hasattr(current_model, 'word_to_id') else current_model.data_processing.word_to_id
    id2w = current_model.id_to_word if hasattr(current_model, 'id_to_word') else current_model.data_processing.id_to_word
    if any(w not in vocab for w in words): return jsonify({"error": "Word(s) missing"}), 404
    vecs = [current_model.W1_normalized[w2id[w]] for w in words]
    target = vecs[1] - vecs[0] + vecs[2]
    sims = np.dot(current_model.W1_normalized, target / (np.linalg.norm(target) + 1e-9))
    for w in words: sims[w2id[w]] = -np.inf
    pred_id = np.argmax(sims)
    pred_word = id2w[pred_id]
    all_vecs = np.array([vecs[0], vecs[1], vecs[2], current_model.W1_normalized[pred_id]])
    return jsonify({"predicted_word": pred_word, "plot_data": {"labels": [*words, pred_word], "coords": perform_pca(all_vecs).tolist()}})

@app.route('/api/train', methods=['POST'])
def start_train():
    if training_status["active"]: return jsonify({"error": "Training in progress"}), 400
    threading.Thread(target=background_training, args=(request.json,)).start()
    return jsonify({"success": True})

@app.route('/api/status')
def get_status():
    return jsonify(training_status)

if __name__ == '__main__':
    app.run(debug=True, port=5000)