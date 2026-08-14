/**
 * ViPym Studio — Interactive Frontend Logic
 */

let currentTab = 'dashboard';
let experimentsCache = [];
let dagStages = [];

document.addEventListener('DOMContentLoaded', () => {
    setupNavigation();
    loadDashboard();
    renderMoEGrid(896, 16);
});

function setupNavigation() {
    const navButtons = document.querySelectorAll('.nav-item');
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            switchTab(tabId);
        });
    });
}

function switchTab(tabId) {
    currentTab = tabId;
    
    // Update nav buttons
    document.querySelectorAll('.nav-item').forEach(b => {
        b.classList.toggle('active', b.getAttribute('data-tab') === tabId);
    });

    // Update tab panes
    document.querySelectorAll('.tab-pane').forEach(p => {
        p.classList.toggle('active', p.id === `pane-${tabId}`);
    });

    // Load tab-specific data
    if (tabId === 'dashboard') loadDashboard();
    if (tabId === 'pareto') loadParetoSelect();
    if (tabId === 'recipes') loadRecipes();
    if (tabId === 'doctor') loadDoctor();
}

function refreshData() {
    if (currentTab === 'dashboard') loadDashboard();
    if (currentTab === 'pareto') loadParetoSelect();
    if (currentTab === 'recipes') loadRecipes();
    if (currentTab === 'doctor') loadDoctor();
}

async function loadDashboard() {
    try {
        const res = await fetch('/api/experiments');
        experimentsCache = await res.json();

        document.getElementById('stat-total-experiments').innerText = experimentsCache.length;
        
        let totalParetoPoints = 0;
        experimentsCache.forEach(e => totalParetoPoints += (e.pareto_points_count || 0));
        document.getElementById('stat-pareto-points').innerText = totalParetoPoints;

        const tbody = document.getElementById('tbody-experiments');
        if (!experimentsCache || experimentsCache.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" class="text-center py-6 text-muted">No experiments found in ./artifacts. Run <code>vipym run -c configs/experiments/smoke_test.yaml</code> to launch one.</td></tr>`;
            return;
        }

        tbody.innerHTML = experimentsCache.map(exp => `
            <tr>
                <td><strong>${escapeHtml(exp.experiment_id)}</strong></td>
                <td><span class="badge ${exp.state === 'REPORT_COMPLETED' ? 'badge-green' : 'badge-cyan'}">${exp.state}</span></td>
                <td>${exp.timestamp ? new Date(exp.timestamp).toLocaleString() : 'N/A'}</td>
                <td>${(exp.duration_sec || 0).toFixed(2)}s</td>
                <td>$${(exp.cost_usd || 0).toFixed(4)}</td>
                <td><strong>${exp.pareto_points_count || 0}</strong> points</td>
                <td>
                    <button class="btn btn-secondary text-sm" onclick="inspectExperiment('${escapeHtml(exp.experiment_id)}')">Inspect</button>
                </td>
            </tr>
        `).join('');
    } catch (e) {
        console.error("Failed to load experiments", e);
    }
}

async function loadParetoSelect() {
    const select = document.getElementById('select-experiment-pareto');
    if (!experimentsCache || experimentsCache.length === 0) {
        const res = await fetch('/api/experiments');
        experimentsCache = await res.json();
    }

    select.innerHTML = experimentsCache.map(e => `
        <option value="${escapeHtml(e.experiment_id)}">${escapeHtml(e.experiment_id)}</option>
    `).join('');

    if (experimentsCache.length > 0) {
        renderParetoPlot();
    } else {
        renderMockParetoPlot();
    }
}

async function renderParetoPlot() {
    const select = document.getElementById('select-experiment-pareto');
    const expId = select.value;
    if (!expId) {
        renderMockParetoPlot();
        return;
    }

    try {
        const res = await fetch(`/api/experiments/${expId}`);
        const data = await res.json();
        const points = data.results || [];

        if (points.length === 0) {
            renderMockParetoPlot();
            return;
        }

        const labels = points.map(p => p.config_id || 'Point');
        const vram = points.map(p => p.peak_vram_gb || 0);
        const pass1 = points.map(p => (p.pass_at_1 || 0) * 100);
        const cost = points.map(p => (p.estimated_cost_usd_per_1m_tokens || 0));

        const trace = {
            x: vram,
            y: pass1,
            text: labels,
            mode: 'markers+text',
            textposition: 'top center',
            marker: {
                size: 16,
                color: pass1,
                colorscale: 'Viridis',
                showscale: true,
                colorbar: { title: 'Pass@1 (%)' }
            },
            type: 'scatter'
        };

        const layout = {
            title: `Pareto Frontier: ${expId}`,
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'rgba(0,0,0,0.2)',
            font: { color: '#f8fafc', family: 'Outfit, sans-serif' },
            xaxis: { title: 'Peak GPU VRAM (GB)', gridcolor: 'rgba(255,255,255,0.08)' },
            yaxis: { title: 'HumanEval Pass@1 (%)', gridcolor: 'rgba(255,255,255,0.08)' },
            margin: { t: 50, b: 50, l: 60, r: 60 }
        };

        Plotly.newPlot('plotly-pareto-div', [trace], layout, { responsive: true });
    } catch (e) {
        renderMockParetoPlot();
    }
}

function renderMockParetoPlot() {
    // Demonstration frontier
    const mockData = [
        { label: 'Baseline (BF16)', vram: 80.0, pass1: 85.2, color: '#f59e0b' },
        { label: 'QuaRot+AWQ W4A16', vram: 24.2, pass1: 84.1, color: '#00f2fe' },
        { label: 'AutoRound W4A16', vram: 24.5, pass1: 83.4, color: '#38bdf8' },
        { label: 'OCP MXFP4 + 2:4 Sparse', vram: 14.8, pass1: 79.8, color: '#c084fc' },
        { label: 'RTN Int4 (Aggressive)', vram: 18.0, pass1: 62.0, color: '#f87171' },
    ];

    const trace = {
        x: mockData.map(d => d.vram),
        y: mockData.map(d => d.pass1),
        text: mockData.map(d => d.label),
        mode: 'markers+text',
        textposition: 'top center',
        marker: {
            size: 18,
            color: mockData.map(d => d.pass1),
            colorscale: 'Bluered',
            showscale: true,
            colorbar: { title: 'Pass@1 (%)' }
        },
        type: 'scatter'
    };

    const layout = {
        title: 'Reference Demonstration: Pareto Optimization Frontier (Quality vs Footprint)',
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'rgba(0,0,0,0.2)',
        font: { color: '#f8fafc', family: 'Outfit, sans-serif' },
        xaxis: { title: 'Peak GPU VRAM Footprint (GB)', gridcolor: 'rgba(255,255,255,0.08)' },
        yaxis: { title: 'Code Intelligence Score: Pass@1 (%)', gridcolor: 'rgba(255,255,255,0.08)' },
        margin: { t: 50, b: 50, l: 60, r: 60 }
    };

    Plotly.newPlot('plotly-pareto-div', [trace], layout, { responsive: true });
}

/* Visual DAG Builder */
function addDagStage(method, scheme) {
    const idx = dagStages.length + 1;
    const stageId = `stage_${String(idx).padStart(2, '0')}_${method}`;
    const deps = idx > 1 ? [dagStages[idx - 2].stage_id] : [];

    dagStages.push({
        stage_id: stageId,
        method: method,
        scheme: scheme,
        dependencies: deps
    });

    renderDagCanvas();
    generateYamlFromDag();
}

function clearDag() {
    dagStages = [];
    renderDagCanvas();
    generateYamlFromDag();
}

function renderDagCanvas() {
    const container = document.getElementById('dag-nodes-canvas');
    if (dagStages.length === 0) {
        container.innerHTML = `<span class="text-muted text-sm">DAG is empty. Add stages from the left palette.</span>`;
        return;
    }

    container.innerHTML = dagStages.map((s, i) => `
        <div class="dag-node-card">
            <div>
                <span class="badge badge-cyan">${s.scheme}</span>
                <strong class="d-block mt-2">${s.stage_id}</strong>
                <p class="text-sm text-muted">Method: ${s.method} | Depends on: ${s.dependencies.join(', ') || 'root'}</p>
            </div>
            ${i < dagStages.length - 1 ? '<span style="font-size: 1.5rem; color: #38bdf8;">➔</span>' : ''}
        </div>
    `).join('');
}

function generateYamlFromDag() {
    if (dagStages.length === 0) {
        document.getElementById('dag-yaml-output').innerText = '# Add stages from the left palette to generate YAML...';
        return;
    }

    const yamlObj = {
        experiment_id: 'custom-interactive-dag-001',
        seed: 42,
        model: {
            id: 'moonshotai/Kimi-K3',
            trust_remote_code: true
        },
        compression_pipeline: dagStages,
        serving: {
            backend: 'vllm',
            tensor_parallel_size: 8
        },
        evaluation: {
            suites: ['humaneval', 'mbpp']
        }
    };

    let yamlStr = `experiment_id: ${yamlObj.experiment_id}\nseed: 42\n\nmodel:\n  id: "${yamlObj.model.id}"\n  trust_remote_code: true\n\ncompression_pipeline:\n`;
    
    dagStages.forEach(s => {
        yamlStr += `  - stage_id: "${s.stage_id}"\n    method: "${s.method}"\n    scheme: "${s.scheme}"\n    dependencies: [${s.dependencies.map(d => `"${d}"`).join(', ')}]\n`;
    });

    yamlStr += `\nserving:\n  backend: "vllm"\n  tensor_parallel_size: 8\n\nevaluation:\n  suites:\n    - "humaneval"\n    - "mbpp"\n`;

    document.getElementById('dag-yaml-output').innerText = yamlStr;
}

/* MoE Expert Lattice */
function renderMoEGrid(totalExperts, activePerToken) {
    const grid = document.getElementById('moe-grid-canvas');
    if (!grid) return;

    let html = '';
    // Shared experts
    html += `<div class="expert-chip shared-expert" title="Shared Expert S1">S1</div>`;
    html += `<div class="expert-chip shared-expert" title="Shared Expert S2">S2</div>`;

    for (let i = 1; i <= totalExperts; i++) {
        const isActive = i <= activePerToken;
        html += `<div class="expert-chip ${isActive ? 'active-expert' : ''}" title="Routed Expert E${i} ${isActive ? '(Active per-token)' : ''}">E${i}</div>`;
    }

    grid.innerHTML = html;
}

async function inspectModel() {
    const modelId = document.getElementById('input-model-id').value;
    try {
        const res = await fetch(`/api/models/inspect?model_id=${encodeURIComponent(modelId)}`);
        const data = await res.json();

        document.getElementById('moe-total-params').innerText = `${data.total_parameters_b} B`;
        document.getElementById('moe-active-params').innerText = `${data.active_parameters_b} B`;
        document.getElementById('moe-routed-experts').innerText = data.num_experts || 'Dense (1)';
        document.getElementById('moe-context-len').innerText = (data.context_window_length || 2048).toLocaleString();

        renderMoEGrid(data.num_experts || 1, data.num_selected_experts || 1);
    } catch (e) {
        console.error("Failed to inspect model", e);
    }
}

/* Recipe Hub */
async function loadRecipes() {
    const grid = document.getElementById('recipes-catalog-grid');
    try {
        const res = await fetch('/api/recipes');
        const recipes = await res.json();

        grid.innerHTML = recipes.map(r => `
            <div class="recipe-card">
                <div>
                    <div class="flex-between mb-4">
                        <span class="badge badge-cyan">${r.domain}</span>
                        <span class="badge badge-gold">${r.compression_ratio}x Compression</span>
                    </div>
                    <h3>${escapeHtml(r.recipe_id)}</h3>
                    <p class="text-sm text-muted mt-2">${escapeHtml(r.description)}</p>
                    <div class="mt-4">
                        <p class="text-sm"><strong>Target Model:</strong> <span class="text-gradient-cyan">${escapeHtml(r.model)}</span></p>
                        <p class="text-sm"><strong>Expected Retention:</strong> <span class="text-gradient-gold">${escapeHtml(r.quality_retention)}</span></p>
                        <p class="text-sm"><strong>Hardware Target:</strong> ${escapeHtml(r.hardware)}</p>
                    </div>
                </div>
                <div class="mt-6 flex-between">
                    <button class="btn btn-secondary text-sm" onclick="copyRecipeCommand('${escapeHtml(r.recipe_id)}')">📋 Copy CLI Command</button>
                </div>
            </div>
        `).join('');
    } catch (e) {
        console.error("Failed to load recipes", e);
    }
}

function copyRecipeCommand(recipeId) {
    const cmd = `vipym recipe run ${recipeId}`;
    navigator.clipboard.writeText(cmd);
    alert(`Copied command to clipboard:\n${cmd}`);
}

/* Doctor */
async function loadDoctor() {
    const container = document.getElementById('doctor-container');
    try {
        const res = await fetch('/api/doctor');
        const data = await res.json();

        container.innerHTML = `
            <div class="stat-card glassmorphism">
                <span class="stat-label">Python Environment</span>
                <span class="stat-value text-gradient-cyan">${data.python_version}</span>
                <span class="stat-desc">CPython runtime</span>
            </div>
            <div class="stat-card glassmorphism">
                <span class="stat-label">CUDA Acceleration</span>
                <span class="stat-value ${data.cuda_available ? 'text-gradient-green' : 'text-gradient-gold'}">${data.cuda_available ? 'Available' : 'CPU Mode'}</span>
                <span class="stat-desc">${data.gpu_count} GPUs detected</span>
            </div>
            <div class="stat-card glassmorphism">
                <span class="stat-label">Disk Storage Free</span>
                <span class="stat-value text-gradient-cyan">${data.disk_free_gb} GB</span>
                <span class="stat-desc">Working drive headroom</span>
            </div>
            <div class="stat-card glassmorphism">
                <span class="stat-label">Docker MicroVM Sandboxing</span>
                <span class="stat-value ${data.docker_available ? 'text-gradient-green' : 'text-gradient-gold'}">${data.docker_available ? 'Ready' : 'Subprocess Fallback'}</span>
                <span class="stat-desc">gVisor user-space kernel isolation</span>
            </div>
        `;
    } catch (e) {
        console.error("Failed to load doctor", e);
    }
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
