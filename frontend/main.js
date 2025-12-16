// Frontend logic to preview a PDF, send it to backend for analysis, and render annotations
// Requirements:
// - Backend: FastAPI endpoint POST /api/analyze_pdf (multipart file)
// - Libraries loaded from index.html: pdf.js, pdf_viewer.js, tippy.js

; (function () {
  'use strict'

  // Elements
  const btnUpload = document.getElementById('btn-upload')
  const btnAnalyze = document.getElementById('btn-analyze')
  const fileInput = document.getElementById('file-input')
  const filenameEl = document.getElementById('filename')
  const progressEl = document.getElementById('progress')
  const msgOk = document.getElementById('msg-ok')
  const msgError = document.getElementById('msg-error')
  const placeholder = document.getElementById('placeholder')
  const viewerContainer = document.getElementById('viewerContainer')
  const pdfViewerRoot = document.getElementById('pdfViewer')
  const resultsPanel = document.getElementById('results-panel')
  const finalScoreEl = document.getElementById('final-score')
  const scoringSummaryEl = document.getElementById('scoring-summary')

  // State
  let currentFile = null /** @type {File|null} */
  let pdfDoc = null /** @type {import('pdfjs-dist').PDFDocumentProxy|null} */
  let pageSizes = [] /** @type {Array<{page:number,width:number,height:number}>} */
  let annotations = [] /** @type {Array<any>} */
  const overlayLayers = new Map() /** pageNum -> HTMLElement */

  // Filters map UI id -> agent category substrings
  const FILTERS = {
    'chk-tone': ['Stylist', 'Tone'],
    'chk-structure': ['Structure Reviewer', 'Structure'],
    'chk-coherence': ['Coherence Analyst', 'Coherence'],
    'chk-citation': ['Citation Analyst', 'Citation']
  }

  // Configure pdf.js worker (use CDN shipped one if available)
  const pdfjsLib = window['pdfjsLib']
  if (pdfjsLib) {
    // Try to auto-detect worker from same CDN version; fallback to default
    if (!pdfjsLib.GlobalWorkerOptions.workerSrc) {
      pdfjsLib.GlobalWorkerOptions.workerSrc =
        'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js'
    }
  }

  // Setup PDF.js viewer
  const eventBus = new pdfjsViewer.EventBus()
  const pdfLinkService = new pdfjsViewer.PDFLinkService({ eventBus })
  const pdfViewer = new pdfjsViewer.PDFViewer({
    container: viewerContainer,
    viewer: pdfViewerRoot,
    eventBus,
    linkService: pdfLinkService,
    removePageBorders: true
  })
  pdfLinkService.setViewer(pdfViewer)

  // UI helpers
  function show(el) {
    el.removeAttribute('hidden')
  }
  function hide(el) {
    el.setAttribute('hidden', 'true')
  }
  function toastOk(text) {
    msgOk.textContent = text
    show(msgOk)
    setTimeout(() => hide(msgOk), 3000)
  }
  function toastError(text) {
    msgError.textContent = text
    show(msgError)
    setTimeout(() => hide(msgError), 5000)
  }
  function setBusy(busy) {
    if (busy) {
      btnAnalyze.setAttribute('disabled', 'true')
      btnUpload.setAttribute('disabled', 'true')
      show(progressEl)
    } else {
      btnAnalyze.removeAttribute('disabled')
      btnUpload.removeAttribute('disabled')
      hide(progressEl)
    }
  }

  // File selection
  btnUpload.addEventListener('click', () => fileInput.click())
  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files && e.target.files[0]
    if (!file) return
    if (file.type !== 'application/pdf') {
      toastError('Please select a PDF file.')
      return
    }
    currentFile = file
    filenameEl.textContent = file.name
    btnAnalyze.removeAttribute('disabled')
    try {
      await loadLocalPdf(file)
      toastOk('Loaded preview')
    } catch (err) {
      console.error(err)
      toastError('Failed to load PDF preview.')
    }
  })

  // Model selection logic
  const scoringAgentSelect = document.getElementById('scoring-agent-select')
  const apiModelContainer = document.getElementById('api-model-container')
  const modelSelect = document.getElementById('model-select')
  
  function updateModelOptions() {
    if (scoringAgentSelect.value === 'api' || scoringAgentSelect.value === 'local-finetuned' || scoringAgentSelect.value === 'local-base') {
      show(apiModelContainer)
    } else {
      hide(apiModelContainer)
    }
  }
  scoringAgentSelect.addEventListener('change', updateModelOptions)
  updateModelOptions()

  // Analyze button
  btnAnalyze.addEventListener('click', async () => {
    if (!currentFile) return
    annotations = []
    pageSizes = []
    clearAllOverlays()
    setBusy(true)
    try {
      const res = await callAnalyze(currentFile)
      annotations = res.annotations || []
      pageSizes = res.page_sizes || []
      
      // Show results
      if (res.final_score !== null && res.final_score !== undefined) {
        finalScoreEl.textContent = res.final_score + '/10'
        scoringSummaryEl.textContent = res.scoring_summary || 'No summary available.'
        show(resultsPanel)
      } else {
        hide(resultsPanel)
      }

      if (!annotations.length) {
        toastOk('No issues found.')
      } else {
        toastOk(`Found ${annotations.length} items`)
      }
      renderAllOverlays()
      scrollToFirstAnnotation()
    } catch (err) {
      console.error(err)
      toastError('Analysis failed. Check backend logs.')
    } finally {
      setBusy(false)
    }
  })

  // Load local PDF into viewer
  async function loadLocalPdf(file) {
    const buf = await file.arrayBuffer()
    const loadingTask = pdfjsLib.getDocument({ data: buf })
    pdfDoc = await loadingTask.promise
    pdfViewer.setDocument(pdfDoc)
    pdfLinkService.setDocument(pdfDoc)
    hide(placeholder)
    show(viewerContainer)
    // Clear previous overlays when loading a new doc
    clearAllOverlays()
    // After first rendering, build overlay roots per page
    // We rely on a render event to size overlays when pages render
  }

  // Backend call
  async function callAnalyze(file) {
    const fd = new FormData()
    fd.append('file', file)

    // Collect active filters
    const activeFilters = []
    if (document.getElementById('chk-tone').checked) activeFilters.push('tone')
    if (document.getElementById('chk-structure').checked) activeFilters.push('structure')
    if (document.getElementById('chk-coherence').checked) activeFilters.push('coherence')
    if (document.getElementById('chk-citation').checked) activeFilters.push('citation')

    fd.append('agents', JSON.stringify(activeFilters))

    // Determine model parameters based on scoring agent selection
    const scoringAgent = document.getElementById('scoring-agent-select').value
    
    let model = ''
    let useLocal = false
    let scoringModel = 'api'

    if (scoringAgent === 'api') {
      model = document.getElementById('model-select').value
      useLocal = false
      scoringModel = 'api'
    } else if (scoringAgent === 'local-base') {
      model = document.getElementById('model-select').value
      useLocal = false
      scoringModel = 'base'
    } else if (scoringAgent === 'local-finetuned') {
      model = document.getElementById('model-select').value
      useLocal = false
      scoringModel = 'finetuned'
    }

    fd.append('model', model)
    fd.append('use_local_model', useLocal)
    fd.append('scoring_model', scoringModel)

    const resp = await fetch('/api/analyze_pdf', {
      method: 'POST',
      body: fd
    })
    if (!resp.ok) {
      const t = await safeText(resp)
      throw new Error(`HTTP ${resp.status}: ${t}`)
    }
    return resp.json()
  }

  async function safeText(resp) {
    try {
      return await resp.text()
    } catch {
      return ''
    }
  }

  // Build and manage overlays
  function ensureOverlayForPage(pageNum) {
    let layer = overlayLayers.get(pageNum)
    if (layer) return layer
    // Find the page DOM element created by pdf.js
    const pageView = pdfViewer._pages?.[pageNum - 1]
    if (!pageView) return null
    const div = pageView.div
    let overlay = div.querySelector(':scope > .annotationOverlay')
    if (!overlay) {
      overlay = document.createElement('div')
      overlay.className = 'annotationOverlay'
      overlay.style.position = 'absolute'
      overlay.style.left = '0'
      overlay.style.top = '0'
      overlay.style.right = '0'
      overlay.style.bottom = '0'
      overlay.style.pointerEvents = 'none'
      overlay.style.zIndex = '20'
      div.appendChild(overlay)
    }
    overlayLayers.set(pageNum, overlay)
    return overlay
  }

  function clearAllOverlays() {
    overlayLayers.forEach((layer) => layer.remove())
    overlayLayers.clear()
  }

  function activeCategoryFilter(item) {
    // item.category or item.agent may contain our keywords
    const checks = Object.keys(FILTERS)
    for (const id of checks) {
      const input = document.getElementById(id)
      if (!input) continue
      const on = input.checked
      const keys = FILTERS[id]
      const hay = `${item.category || ''} ${item.agent || ''}`
      const match = keys.some((k) => hay.toLowerCase().includes(k.toLowerCase()))
      if (match && !on) return false
    }
    return true
  }

  function rectForPageSpace(item, pageNum) {
    // Backend returns rect in PDF points for page index (1-based already adjusted)
    // We need to scale to the rendered viewport size
    const ps = pageSizes.find((p) => p.page === pageNum)
    if (!ps) return null
    const pageView = pdfViewer._pages?.[pageNum - 1]
    if (!pageView) return null
    const viewport = pageView.viewport
    const scaleX = viewport.width / ps.width
    const scaleY = viewport.height / ps.height
    const [x0, y0, x1, y1] = item.rect
    return {
      left: x0 * scaleX,
      top: y0 * scaleY,
      width: (x1 - x0) * scaleX,
      height: (y1 - y0) * scaleY
    }
  }

  function renderAllOverlays() {
    if (!annotations || !annotations.length) return
    // Group by page
    const byPage = new Map()
    for (const a of annotations) {
      const p = a.page // already 1-based per backend
      if (!byPage.has(p)) byPage.set(p, [])
      byPage.get(p).push(a)
    }
    // For each page, draw boxes
    byPage.forEach((items, pageNum) => {
      const layer = ensureOverlayForPage(pageNum)
      if (!layer) return
      // Clear layer
      layer.replaceChildren()
      for (const item of items) {
        if (!activeCategoryFilter(item)) continue
        const rects = item.rects || (item.rect ? [item.rect] : [])

        for (const rectData of rects) {
          const tempItem = { rect: rectData }
          const r = rectForPageSpace(tempItem, pageNum)
          if (!r) continue
          const el = document.createElement('div')
          el.className = 'anno'
          el.style.position = 'absolute'
          el.style.left = `${r.left}px`
          el.style.top = `${r.top}px`
          el.style.width = `${r.width}px`
          el.style.height = `${r.height}px`
          el.style.background = overlayColor(item)
          el.style.opacity = '0.8'
          el.style.border = `1px solid ${overlayStroke(item)}`
          el.style.borderRadius = '2px'
          el.style.pointerEvents = 'auto'
          el.style.cursor = 'help'
          el.dataset.tooltip = tooltipHtml(item)
          layer.appendChild(el)
          // Attach tooltip
          tippy(el, {
            allowHTML: true,
            content: el.dataset.tooltip,
            theme: 'light-border',
            delay: [100, 0],
            maxWidth: 420,
            interactive: true
          })
        }
      }
    })
  }

  function overlayColor(item) {
    const cat = `${item.category || ''}`.toLowerCase()
    if (cat.includes('structure')) return 'rgba(255, 159, 64, 0.35)'
    if (cat.includes('coherence')) return 'rgba(54, 162, 235, 0.35)'
    if (cat.includes('citation')) return 'rgba(153, 102, 255, 0.35)'
    // stylist/tone default
    return 'rgba(255, 99, 132, 0.35)'
  }
  function overlayStroke(item) {
    const cat = `${item.category || ''}`.toLowerCase()
    if (cat.includes('structure')) return 'rgb(255, 159, 64)'
    if (cat.includes('coherence')) return 'rgb(54, 162, 235)'
    if (cat.includes('citation')) return 'rgb(153, 102, 255)'
    return 'rgb(255, 99, 132)'
  }

  function tooltipHtml(item) {
    const esc = (s) => String(s || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
    return `
			<div class="tip">
				<div class="tip-h">
					<span class="tip-cat">${esc(item.category || 'Other')}</span>
					<span class="tip-agent">${esc(item.agent || '')}</span>
				</div>
				<div class="tip-quote">“${esc(item.quote || '')}”</div>
				<div class="tip-comment">${esc(item.comment || '')}</div>
			</div>
		`
  }

  function scrollToFirstAnnotation() {
    if (!annotations || !annotations.length) return
    const first = annotations[0]
    const pageNum = first.page
    const layer = ensureOverlayForPage(pageNum)
    if (!layer) return
    // find the element roughly near the first annotation
    const el = layer.querySelector('.anno')
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  // Re-render overlays when pages render (e.g., after zoom)
  eventBus.on('pagesinit', () => {
    pdfViewer.currentScaleValue = 'page-width'
  })
  eventBus.on('pagerendered', () => {
    // pagerendered fires per page; we can rerender overlays for visible pages
    renderAllOverlays()
  })
  window.addEventListener('resize', () => {
    renderAllOverlays()
  })

  // Filters listeners
  Object.keys(FILTERS).forEach((id) => {
    const el = document.getElementById(id)
    if (el) el.addEventListener('change', () => renderAllOverlays())
  })
})()

