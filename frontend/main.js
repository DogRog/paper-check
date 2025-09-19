const API_BASE = localStorage.getItem('API_BASE') || 'http://127.0.0.1:8000';
const btnUpload = document.getElementById('btn-upload');
const btnAnalyze = document.getElementById('btn-analyze');
const btnSettings = document.getElementById('btn-settings');
const fileInput = document.getElementById('file-input');
const filenameEl = document.getElementById('filename');
const progressEl = document.getElementById('progress');
const placeholder = document.getElementById('placeholder');
const viewerContainer = document.getElementById('viewerContainer');
const pdfViewer = document.getElementById('pdfViewer');
const filtersContainer = document.getElementById('filters');
const msgOk = document.getElementById('msg-ok');
const msgError = document.getElementById('msg-error');

// Settings modal elements
const settingsModal = document.getElementById('settings-modal');
const btnCloseSettings = document.getElementById('btn-close-settings');
const btnNewAgent = document.getElementById('btn-new-agent');
const agentsUl = document.getElementById('agents-ul');
const agentIdEl = document.getElementById('agent-id');
const agentNameEl = document.getElementById('agent-name');
const agentCategoryEl = document.getElementById('agent-category');
const agentPromptEl = document.getElementById('agent-prompt');
const btnSaveAgent = document.getElementById('btn-save-agent');
const btnDeleteAgent = document.getElementById('btn-delete-agent');
const settingsMsg = document.getElementById('settings-msg');

function show(el, html) { el.innerHTML = html; el.hidden = false; }
function hide(el) { el.hidden = true; el.innerHTML = ''; }

btnUpload.addEventListener('click', () => fileInput.click());

let currentPdf = null;
let pdfEventBus = null;
let highlights = [];
let activeFilters = new Set();
let cachedAgents = [];

function clearHighlights() { for (const h of highlights) h.el.remove(); highlights = []; }
function applyFilterVisibility() {
  for (const h of highlights) {
    h.el.style.display = activeFilters.has(h.category) ? 'block' : 'none';
  }
}

function renderFilters() {
  filtersContainer.innerHTML = '';
  for (const a of cachedAgents) {
    const id = `filter-${a.name.replace(/\s+/g,'-').toLowerCase()}`;
    const label = document.createElement('label');
    label.className = 'check';
    label.innerHTML = `<input id="${id}" type="checkbox" ${activeFilters.has(a.name) ? 'checked' : ''}/> <span>${a.name}</span>`;
    const input = label.querySelector('input');
    input.addEventListener('change', () => {
      if (input.checked) activeFilters.add(a.name); else activeFilters.delete(a.name);
      applyFilterVisibility();
    });
    filtersContainer.appendChild(label);
  }
}

fileInput.addEventListener('change', async () => {
  hide(msgOk); hide(msgError); clearHighlights();
  const file = fileInput.files?.[0];
  if (!file) { filenameEl.textContent = 'No file selected'; viewerContainer.hidden = true; placeholder.hidden = false; btnAnalyze.disabled = true; return; }
  filenameEl.textContent = file.name;

  const pdfjsLib = window['pdfjsLib'];
  pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
  const arrayBuf = await file.arrayBuffer();
  const loadingTask = pdfjsLib.getDocument({ data: arrayBuf });
  currentPdf = await loadingTask.promise;
  pdfViewer.innerHTML = '';
  viewerContainer.hidden = false; placeholder.hidden = true; btnAnalyze.disabled = false;
  pdfEventBus = new window['pdfjsViewer'].EventBus();
  const viewer = new window['pdfjsViewer'].PDFViewer({ container: viewerContainer, viewer: pdfViewer, textLayerMode: 2, annotationMode: 2, eventBus: pdfEventBus });
  viewer.setDocument(currentPdf);
});

function colorForAgent(agent, category) {
  // Stable but varied colors; map by category string
  const key = (category || '').toLowerCase();
  if (key.includes('coherence')) return 'rgba(16,185,129,0.35)';
  if (key.includes('structure')) return 'rgba(59,130,246,0.35)';
  if (key.includes('tone') || key.includes('style')) return 'rgba(234,88,12,0.35)';
  return 'rgba(217,119,6,0.35)';
}
function ensureOverlay(pageView) { let overlay = pageView.querySelector('.overlay-layer'); if (!overlay) { overlay = document.createElement('div'); overlay.className = 'overlay-layer'; overlay.style.position = 'absolute'; overlay.style.left = overlay.style.top = overlay.style.right = overlay.style.bottom = '0'; overlay.style.pointerEvents = 'auto'; overlay.style.zIndex = '30'; pageView.appendChild(overlay);} return overlay; }
async function waitForTextLayer(pageNum, attempts=30, delay=100){ for(let i=0;i<attempts;i++){ const pageView = pdfViewer.querySelector(`.page[data-page-number="${pageNum}"]`); const textLayer = pageView && pageView.querySelector('.textLayer'); if (textLayer && textLayer.querySelector('span')) return true; await new Promise(r=>setTimeout(r,delay)); } return false; }

btnAnalyze.addEventListener('click', async () => {
  hide(msgOk); hide(msgError); clearHighlights();
  const file = fileInput.files?.[0]; if (!file) { show(msgError, 'Please choose a PDF first.'); return; }
  try {
    // Disable controls and show progress
    btnAnalyze.disabled = true; btnUpload.disabled = true; show(progressEl, 'Analyzing…');
    const formData = new FormData(); formData.append('file', file, file.name);
    const res = await fetch(`${API_BASE}/api/analyze_pdf`, { method: 'POST', body: formData });
    if (!res.ok) { const text = await res.text(); let data; try { data = JSON.parse(text); } catch { data = { detail: text }; } throw new Error(data.detail || data.message || `Analyze failed (${res.status})`); }
    const data = await res.json(); const annotations = data.annotations || []; const pageSizes = new Map((data.page_sizes || []).map(p => [p.page, { width: p.width, height: p.height }]));

    const num = currentPdf.numPages; for (let i=1;i<=num;i++){ await waitForTextLayer(i); }

  const byPage = new Map(); for (const a of annotations){ if(!byPage.has(a.page)) byPage.set(a.page,[]); byPage.get(a.page).push(a); }
    for (const [pageNum, list] of byPage.entries()) {
      const pageView = pdfViewer.querySelector(`.page[data-page-number="${pageNum}"]`); if (!pageView) continue;
      const overlay = ensureOverlay(pageView); const pageRect = pageView.getBoundingClientRect(); const size = pageSizes.get(pageNum);
      let sx=1, sy=1; if (size && size.width>0 && size.height>0) { sx = pageRect.width/size.width; sy = pageRect.height/size.height; }
      for (const a of list) {
        const [x1,y1,x2,y2] = a.rect || [0,0,0,0]; const left = x1*sx; const top = y1*sy; const width = (x2-x1)*sx; const height = (y2-y1)*sy;
        const category = a.category || a.agent || 'Other';
        const el = document.createElement('div'); el.className='hl'; Object.assign(el.style,{position:'absolute',left:`${left}px`,top:`${top}px`,width:`${width}px`,height:`${height}px`,background: colorForAgent(a.agent, category),pointerEvents:'auto',cursor:'help'});
        const tooltip = document.createElement('div'); tooltip.innerHTML = `<strong>${a.agent}</strong><br/><em>${a.quote||''}</em><br/>${a.comment||''}`; tippy(el,{content:tooltip,allowHTML:true,theme:'light-border',maxWidth:420});
        overlay.appendChild(el);
        highlights.push({ el, category });
      }
    }
    applyFilterVisibility(); show(msgOk, `${annotations.length} highlights added`);
  } catch (err) { show(msgError, err.message || String(err)); }
  finally {
    // Re-enable controls and hide progress
    btnAnalyze.disabled = false; btnUpload.disabled = false; hide(progressEl);
  }
});

// -------- Settings UI / Agents CRUD --------
function openSettings() { settingsModal.hidden = false; loadAgents(); }
function closeSettings() { settingsModal.hidden = true; clearAgentForm(); agentsUl.innerHTML = ''; hide(settingsMsg); }
btnSettings?.addEventListener('click', openSettings);
btnCloseSettings?.addEventListener('click', closeSettings);
settingsModal?.addEventListener('click', (e) => { if (e.target.classList.contains('modal-backdrop')) closeSettings(); });

function showSettingsMsg(text, ok=false) { settingsMsg.textContent = text; settingsMsg.hidden = false; settingsMsg.classList.toggle('ok', ok); settingsMsg.classList.toggle('error', !ok); }
function clearAgentForm() {
  agentIdEl.value = '';
  agentNameEl.value = '';
  // category is derived from name now
  agentPromptEl.value = '';
  btnDeleteAgent.disabled = true;
}

btnNewAgent?.addEventListener('click', () => { clearAgentForm(); agentNameEl.focus(); });

async function loadAgents() {
  try {
    const res = await fetch(`${API_BASE}/api/agents`);
    if (!res.ok) throw new Error(`Failed to load agents (${res.status})`);
    const data = await res.json();
    const agents = (data.agents || []).map(a => ({...a, category: a.name})); // enforce name==category in UI
    cachedAgents = agents;
    if (activeFilters.size === 0) {
      // default all on
      for (const a of agents) activeFilters.add(a.name);
    } else {
      // prune filters for removed agents
      for (const key of Array.from(activeFilters)) if (!agents.find(a=>a.name===key)) activeFilters.delete(key);
    }
    renderFilters();
    agentsUl.innerHTML = '';
    for (const a of agents) {
      const li = document.createElement('li');
      li.className = 'li';
      li.innerHTML = `<div class="li-title">${a.name}</div><div class="li-sub">${a.name}</div>`;
      li.addEventListener('click', () => selectAgent(a));
      agentsUl.appendChild(li);
    }
  } catch (err) {
    showSettingsMsg(err.message || String(err));
  }
}

function selectAgent(a) {
  agentIdEl.value = a.id;
  agentNameEl.value = a.name || '';
  // category derived from name
  agentPromptEl.value = a.prompt || '';
  btnDeleteAgent.disabled = false;
}

btnSaveAgent?.addEventListener('click', async () => {
  try {
    hide(settingsMsg);
    const payload = {
      name: agentNameEl.value.trim(),
      category: agentNameEl.value.trim(),
      prompt: agentPromptEl.value.trim(),
    };
    if (!payload.name) { showSettingsMsg('Name is required'); return; }
    const id = agentIdEl.value.trim();
    const method = id ? 'PUT' : 'POST';
    const url = id ? `${API_BASE}/api/agents/${encodeURIComponent(id)}` : `${API_BASE}/api/agents`;
    const res = await fetch(url, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    if (!res.ok) throw new Error(`Save failed (${res.status})`);
    showSettingsMsg('Saved', true);
    await loadAgents();
    // After CRUD, re-apply filters to highlights
    applyFilterVisibility();
  } catch (err) {
    showSettingsMsg(err.message || String(err));
  }
});

btnDeleteAgent?.addEventListener('click', async () => {
  const id = agentIdEl.value.trim(); if (!id) return;
  try {
    const res = await fetch(`${API_BASE}/api/agents/${encodeURIComponent(id)}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(`Delete failed (${res.status})`);
    showSettingsMsg('Deleted', true);
    clearAgentForm();
    await loadAgents();
    applyFilterVisibility();
  } catch (err) {
    showSettingsMsg(err.message || String(err));
  }
});
