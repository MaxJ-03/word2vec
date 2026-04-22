function switchTab(t) {
    document.querySelectorAll('.tab-content').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('nav button').forEach(x => x.classList.remove('active'));
    document.getElementById(t + '-section').classList.add('active');
    document.getElementById('tab-' + t).classList.add('active');
}

let loadedModelFilename = null;

function setModelLoadMessage(message, level = "neutral") {
    const messageBox = document.getElementById('model-load-message');
    messageBox.textContent = message || "";
    messageBox.classList.remove('error', 'success');
    if (level === "error") messageBox.classList.add('error');
    if (level === "success") messageBox.classList.add('success');
}

function setLoadButtonState(state) {
    const button = document.getElementById('load-model-btn');

    button.classList.remove('is-loaded');
    button.disabled = false;

    if (state === 'loading') {
        button.textContent = 'Loading...';
        button.disabled = true;
        return;
    }

    if (state === 'loaded') {
        button.textContent = 'Loaded!';
        button.classList.add('is-loaded');
        return;
    }

    button.textContent = 'Load into Memory';
}

function parseModelName(filename) {
    return filename.endsWith('.txt') ? filename.slice(0, -4) : filename;
}

function populateModelSelect(models, leaderboardRows, selectedValue = null) {
    const modelSelect = document.getElementById('model-select');
    modelSelect.innerHTML = '';

    const scoreByModel = new Map(
        leaderboardRows
            .filter(row => typeof row.model_name === 'string')
            .map(row => [parseModelName(row.model_name), Number(row.total_accuracy) || 0])
    );

    const sortedModels = [...models].sort((a, b) => {
        const scoreA = scoreByModel.get(parseModelName(a.filename));
        const scoreB = scoreByModel.get(parseModelName(b.filename));

        if (scoreA !== undefined && scoreB !== undefined) {
            return scoreB - scoreA;
        }

        if (scoreA !== undefined) return -1;
        if (scoreB !== undefined) return 1;

        return a.display_name.localeCompare(b.display_name);
    });


    if (!sortedModels.length) {
        const emptyOption = document.createElement('option');
        emptyOption.value = '';
        emptyOption.textContent = 'No models found in embeddings/';
        modelSelect.appendChild(emptyOption);
        return;
    }

    sortedModels.forEach(model => {
        const option = document.createElement('option');
        option.value = model.filename;
        option.textContent = model.display_name;
        modelSelect.appendChild(option);
    });

    if (selectedValue) {
        const matching = sortedModels.find(model => model.filename === selectedValue);
        if (matching) {
            modelSelect.value = selectedValue;
        }
    }
}

async function refreshModelOrdering(selectedValue = null) {
    const [modelsResponse, leaderboardResponse] = await Promise.all([
        fetch('/api/models'),
        fetch('/api/leaderboard')
    ]);

    const [models, leaderboard] = await Promise.all([
        modelsResponse.json(),
        leaderboardResponse.json()
    ]);

    populateModelSelect(models, leaderboard, selectedValue);
    return leaderboard;
}

document.addEventListener("DOMContentLoaded", async () => {
    try {
        const [datasetsResponse, leaderboard] = await Promise.all([
            fetch('/api/datasets'),
            refreshModelOrdering()
        ]);

        const datasets = await datasetsResponse.json();
        const datasetSelect = document.getElementById('dataset-select');

        datasets.forEach(dataset => {
            const option = document.createElement('option');
            option.value = dataset;
            option.textContent = dataset;
            datasetSelect.appendChild(option);
        });

        fetchLeaderboard(leaderboard);
    } catch (error) {
        setModelLoadMessage('Failed to fetch initial dashboard data.', 'error');
    }

    setInterval(checkStatus, 2000);
});

function fetchLeaderboard(existingData = null) {
    const renderLeaderboard = (rows) => {
        const body = document.getElementById('leaderboard-body');
        body.innerHTML = rows.map(row => `<tr><td>${row.display_name}</td><td>${row.semantic_accuracy?.toFixed(1)}%</td><td>${row.syntactic_accuracy?.toFixed(1)}%</td><td><b>${row.total_accuracy?.toFixed(1)}%</b></td></tr>`).join('');
    };

    if (existingData) {
        renderLeaderboard(existingData);
        return;
    }

    fetch('/api/leaderboard')
        .then(response => response.json())
        .then(data => renderLeaderboard(data));
}

function checkStatus() {
    fetch('/api/status').then(r => r.json()).then(async data => {
        const span = document.getElementById('status-text');
        span.textContent = data.message;
        span.style.color = data.active ? "#3498db" : "#27ae60";

        if (!data.active && data.message === "Training Complete!") {
            const selectedFilename = document.getElementById('model-select').value;
            const leaderboard = await refreshModelOrdering(selectedFilename || loadedModelFilename);
            fetchLeaderboard(leaderboard);
        }
    });
}

document.getElementById('model-select').addEventListener('change', function () {
    if (loadedModelFilename && this.value !== loadedModelFilename) {
        setLoadButtonState('default');
        setModelLoadMessage('Selected model changed. Click "Load into Memory" to use it.');
        document.getElementById('explorer-tools').style.display = 'none';
    }
});

document.getElementById('load-model-btn').onclick = async function () {
    const selectedFilename = document.getElementById('model-select').value;
    if (!selectedFilename) {
        setModelLoadMessage('No model is selected.', 'error');
        return;
    }

    setLoadButtonState('loading');
    setModelLoadMessage('');

    try {
        const response = await fetch('/api/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: selectedFilename })
        });

        const payload = await response.json();

        if (!response.ok || payload.error) {
            throw new Error(payload.error || 'Failed to load model.');
        }

        loadedModelFilename = selectedFilename;
        document.getElementById('explorer-tools').style.display = 'block';
        setLoadButtonState('loaded');
        setModelLoadMessage(`Loaded ${selectedFilename}.`, 'success');
    } catch (error) {
        loadedModelFilename = null;
        document.getElementById('explorer-tools').style.display = 'none';
        setLoadButtonState('default');
        setModelLoadMessage(error.message, 'error');
    }
};

document.getElementById('search-word-btn').onclick = function () {
    const w = document.getElementById('target-word').value;
    fetch('/api/similar', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ word: w }) })
        .then(r => r.json()).then(data => {
            const l = document.getElementById('neighbors-list');
            l.innerHTML = data.error ? `<li>${data.error}</li>` : data.neighbors.map(n => `<li>${n.word}</li>`).join('');
        });
};

document.getElementById('calculate-analogy-btn').onclick = function () {
    const p = { word_a: document.getElementById('word-a').value, word_b: document.getElementById('word-b').value, word_c: document.getElementById('word-c').value };
    fetch('/api/analogy', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(p) })
        .then(r => r.json()).then(data => {
            const res = document.getElementById('analogy-result');
            if (data.error) res.textContent = data.error;
            else {
                res.textContent = "Predicted: " + data.predicted_word;
                drawPlot(data.plot_data);
            }
        });
};

document.getElementById('start-training-btn').onclick = function () {
    const arches = Array.from(document.querySelectorAll('.arch-cb:checked')).map(cb => cb.value);
    if (!arches.length) return alert("Select at least one architecture.");

    const payload = {
        dataset: document.getElementById('dataset-select').value,
        epochs: document.getElementById('train-ep').value,
        lr: document.getElementById('train-lr').value,
        dim: document.getElementById('train-dim').value,
        window: document.getElementById('train-win').value,
        ns: document.getElementById('train-ns').value,
        architectures: arches
    };

    fetch('/api/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
        .then(r => r.json()).then(data => {
            if (data.error) alert(data.error);
            else document.getElementById('status-text').textContent = "Starting...";
        });
};

function drawPlot(data) {
    const c = document.getElementById('analogy-canvas');
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    const pad = 40;
    const xs = data.coords.map(c => c[0]), ys = data.coords.map(c => c[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const scale = (v, min, max, s) => pad + (v - min) * (s - 2 * pad) / (max - min || 1);
    const pts = data.coords.map(co => ({ x: scale(co[0], minX, maxX, c.width), y: c.height - scale(co[1], minY, maxY, c.height) }));
    ctx.setLineDash([5, 5]); ctx.strokeStyle = "#ccc";
    ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y); ctx.lineTo(pts[1].x, pts[1].y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pts[2].x, pts[2].y); ctx.lineTo(pts[3].x, pts[3].y); ctx.stroke();
    data.labels.forEach((l, i) => {
        ctx.fillStyle = (i === 3) ? "#18bc9c" : "#34495e";
        ctx.beginPath(); ctx.arc(pts[i].x, pts[i].y, 5, 0, 7); ctx.fill();
        ctx.fillText(l, pts[i].x + 10, pts[i].y);
    });
}