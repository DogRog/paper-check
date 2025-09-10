/* global pdfjsLib, tippy */
const API_URL = 'http://127.0.0.1:8000';

const uploadForm = document.getElementById('uploadForm');
const pdfInput = document.getElementById('pdfFile');
const pdfViewer = document.getElementById('pdfViewer');
const overlay = document.getElementById('overlay');
const filters = document.getElementById('filters');

let pdfDoc = null;
let BASE_SCALE = 1.5; // used for coordinate scaling

async function renderPdf(arrayBuffer) {
  pdfViewer.innerHTML = '';
  overlay.innerHTML = '';
  pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
  const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
  pdfDoc = await loadingTask.promise;

  for (let pageNum = 1; pageNum <= pdfDoc.numPages; pageNum++) {
    const page = await pdfDoc.getPage(pageNum);
    const viewport = page.getViewport({ scale: BASE_SCALE });
    const outputScale = window.devicePixelRatio || 1;

    const canvas = document.createElement('canvas');
    canvas.className = 'pageCanvas';
    // Set internal pixel size for crisp rendering on HiDPI displays
    canvas.width = Math.floor(viewport.width * outputScale);
    canvas.height = Math.floor(viewport.height * outputScale);
    // CSS size matches viewport so layout positions remain correct
    canvas.style.width = `${viewport.width}px`;
    canvas.style.height = `${viewport.height}px`;
    const ctx = canvas.getContext('2d');

    const renderContext = {
      canvasContext: ctx,
      viewport,
      transform: [outputScale, 0, 0, outputScale, 0, 0],
    };
    await page.render(renderContext).promise;
    pdfViewer.appendChild(canvas);
  }

  overlay.style.width = pdfViewer.clientWidth + 'px';
  overlay.style.height = pdfViewer.clientHeight + 'px';
  overlay.style.top = pdfViewer.offsetTop + 'px';
  overlay.style.left = pdfViewer.offsetLeft + 'px';
}

function addHighlights(annotations) {
  // Clear old
  overlay.innerHTML = '';

  // Build page offsets map
  const pageCanvases = Array.from(document.querySelectorAll('.pageCanvas'));
  const pageOffsets = pageCanvases.map((c) => ({
    top: c.offsetTop,
    left: c.offsetLeft,
    width: c.width,
    height: c.height,
  }));

  annotations.forEach((a) => {
    const { page, rect, agent, quote, comment } = a;
    // Filter by agent per active checkboxes
    if (!isAgentEnabled(agent)) return;
    const canvas = pageCanvases[page];
    if (!canvas) return;

    // PDF coordinates are in points; our canvas is scaled at 1.5 default viewport scale.
    // When we rendered, viewport.scale=1.5, so the coordinates from backend (from original PDF points)
    // need to be scaled by the same factor. We don't know the exact DPI used on backend, so assume 1.0 page rect
    // and match by canvas/page size ratio using first page as reference.
    // We'll compute scale from page bbox to canvas size using PDF.js page.getViewport(1.0) size API if needed later.

  // For now, assume doc[0].rect mapped to canvas size at BASE_SCALE on backend. We'll try basic placement.
    const [x1, y1, x2, y2] = rect;

    const highlight = document.createElement('div');
  highlight.className = 'highlight';
  highlight.dataset.agent = agent;

    // Place relative to page canvas
    const pageTop = pageOffsets[page].top;
    const pageLeft = pageOffsets[page].left;

    // We need the size of original page points. Let's infer from canvas and a typical 72dpi baseline.
    // PyMuPDF returns points coordinates; PDF.js viewport at scale s maps points*s to pixels.
    // We can estimate scale as canvas.width / page.view[2] (but we don't have view here). We'll approximate with
    // width ratio using the first canvas and stick to it for all pages (assuming same scale used).

  // Safer: use the same base scale we used to render the CSS-sized canvases
  const SCALE = BASE_SCALE;
  const left = pageLeft + x1 * SCALE;
  const top = pageTop + y1 * SCALE;
  const width = (x2 - x1) * SCALE;
  const height = (y2 - y1) * SCALE;

    Object.assign(highlight.style, {
      left: left + 'px',
      top: top + 'px',
      width: width + 'px',
      height: height + 'px',
    });

    overlay.appendChild(highlight);

    tippy(highlight, {
      content: `<strong>${agent}</strong><br/><em>${quote}</em><br/>${comment}`,
      allowHTML: true,
      theme: 'light',
      placement: 'top',
      interactive: true,
    });
  });
}

function isAgentEnabled(agent) {
  const inputs = filters.querySelectorAll('input[type="checkbox"]');
  for (const input of inputs) {
    if (input.dataset.agent === agent) {
      return input.checked;
    }
  }
  return true;
}

filters.addEventListener('change', () => {
  const nodes = overlay.querySelectorAll('.highlight');
  nodes.forEach(node => {
    const agent = node.dataset.agent || '';
    node.style.display = isAgentEnabled(agent) ? 'block' : 'none';
  });
});

uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = pdfInput.files[0];
  if (!file) return;

  const arrayBuffer = await file.arrayBuffer();
  await renderPdf(arrayBuffer);

  const formData = new FormData();
  // Send the original File to preserve metadata and avoid any browser quirks
  formData.append('file', file);

  const res = await fetch(API_URL + '/analyze', { method: 'POST', body: formData });
  const data = await res.json();
  addHighlights(data.annotations || []);
});
