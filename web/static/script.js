function switchTab(t) {
    document.querySelectorAll('.tab-content').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('nav button').forEach(x => x.classList.remove('active'));
    document.getElementById(t + '-section').classList.add('active');
    document.getElementById('tab-' + t).classList.add('active');
}

document.addEventListener("DOMContentLoaded", () => {
    fetch('/api/models').then(r => r.json()).then(data => {
        const s = document.getElementById('model-select');
        data.forEach(m => {
            let o = document.createElement('option');
            o.value = m.filename; o.textContent = m.display_name;
            s.appendChild(o);
        });
    });

    fetch('/api/datasets').then(r => r.json()).then(data => {
        const s = document.getElementById('dataset-select');
        data.forEach(d => {
            let o = document.createElement('option');
            o.value = d; o.textContent = d;
            s.appendChild(o);
        });
    });

    fetchLeaderboard();
    setInterval(checkStatus, 2000);
});

function fetchLeaderboard() {
    fetch('/api/leaderboard').then(r => r.json()).then(data => {
        const b = document.getElementById('leaderboard-body');
        b.innerHTML = data.map(r => `<tr><td>${r.display_name}</td><td>${r.semantic_accuracy?.toFixed(1)}%</td><td>${r.syntactic_accuracy?.toFixed(1)}%</td><td><b>${r.total_accuracy?.toFixed(1)}%</b></td></tr>`).join('');
    });
}

function checkStatus() {
    fetch('/api/status').then(r => r.json()).then(data => {
        const span = document.getElementById('status-text');
        span.textContent = data.message;
        span.style.color = data.active ? "#3498db" : "#27ae60";
        if (!data.active && data.message === "Training Complete!") fetchLeaderboard();
    });
}

document.getElementById('load-model-btn').onclick = function () {
    const f = document.getElementById('model-select').value;
    if (!f) return;
    this.textContent = "Loading...";
    fetch('/api/load', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ filename: f }) })
        .then(r => r.json()).then(() => {
            document.getElementById('explorer-tools').style.display = 'block';
            this.textContent = "Loaded!";
            this.style.background = "#18bc9c";
        });
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