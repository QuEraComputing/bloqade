(function () {
  const plot = document.getElementById('{plot_id}');
  if (!plot || plot.__archVisualizerHoverInstalled) return;
  plot.__archVisualizerHoverInstalled = true;

  const busIndices = (plot.layout.meta || {}).archVisualizerBusTraceIndices || [];
  const busIndexSet = new Set(busIndices);
  const siteTraceIndex = (plot.layout.meta || {}).archVisualizerSiteTraceIndex;
  const siteLabels = (plot.layout.meta || {}).archVisualizerSiteLabels || [];
  const siteLanePaths = (plot.layout.meta || {}).archVisualizerSiteLanePaths || [];
  const siteLanePathRefs =
    (plot.layout.meta || {}).archVisualizerSiteLanePathRefs || {};
  const busControls = (plot.layout.meta || {}).archVisualizerBusControls || [];
  const siteLanePreviewMode =
    (plot.layout.meta || {}).archVisualizerSiteLanePreviewMode || 'hover';
  const debuggerMeta = (plot.layout.meta || {}).bloqadePlotlyDebugger || {};
  const debuggerFrameNames = debuggerMeta.frameNames || [];
  const debuggerSlider = (plot.layout.sliders || [])[0];
  const interactiveDebugger =
    debuggerSlider !== undefined && debuggerFrameNames.length >= 2;
  let currentDebuggerStep = interactiveDebugger &&
    Number.isInteger(debuggerSlider.active)
    ? Math.min(Math.max(debuggerSlider.active, 0), debuggerFrameNames.length - 1)
    : 0;
  // A jump updates the slider directly; its matching frame event is redundant.
  let pendingDebuggerJump = null;
  let currentPathStyle =
    (plot.layout.meta || {}).archVisualizerPathStyle || 'exact';
  let highlightPath = null;
  let siteLaneOverlays = [];
  let movePathHoverOverlays = [];
  let activeSiteKey = null;
  const selectedSites = new Map();
  const busCheckboxes = new Map();
  const busVisibility = busControls.map(
    (control) => plot.data[control.traceIndex].visible === true
  );
  let previewHistory = [];
  let previewHistoryIndex = -1;
  let previewMutationVersion = 0;
  let undoPreviewButton = null;
  let redoPreviewButton = null;
  let lanePreviewTooltip = null;
  let lanePreviewConnector = null;
  let atomTooltipActive = false;
  let customPathTooltipActive = false;

  function getPathOverlayLayer() {
    let layer = plot.querySelector('[data-arch-visualizer-path-overlays]');
    if (layer !== null && layer.isConnected) return layer;

    // Plotly owns and replaces ``.hoverlayer`` as the pointer moves between
    // points. Keep our paths in the stable upper SVG instead so drawing one
    // site's lanes cannot make subsequent atom/site hover targets disappear.
    const upperSvg = [...plot.querySelectorAll('.main-svg')].find(
      (svg) => svg.querySelector('.infolayer') !== null
    );
    if (upperSvg === undefined) return null;

    layer = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    layer.setAttribute('data-arch-visualizer-path-overlays', '');
    const infoLayer = upperSvg.querySelector('.infolayer');
    upperSvg.insertBefore(layer, infoLayer);
    return layer;
  }

  function clearHighlight() {
    if (highlightPath !== null) highlightPath.remove();
    highlightPath = null;
  }

  function clearSiteLaneOverlays() {
    siteLaneOverlays.forEach((element) => element.remove());
    siteLaneOverlays = [];
    if (!atomTooltipActive) hideLanePreviewTooltip();
  }

  function clearMovePathHoverOverlays() {
    movePathHoverOverlays.forEach((element) => element.remove());
    movePathHoverOverlays = [];
  }

  function getLanePreviewTooltip() {
    if (lanePreviewTooltip !== null && lanePreviewTooltip.isConnected) {
      return lanePreviewTooltip;
    }

    lanePreviewTooltip = document.createElement('div');
    lanePreviewTooltip.setAttribute('data-arch-visualizer-lane-tooltip', '');
    lanePreviewTooltip.style.position = 'absolute';
    lanePreviewTooltip.style.display = 'none';
    lanePreviewTooltip.style.pointerEvents = 'none';
    lanePreviewTooltip.style.zIndex = '1002';
    lanePreviewTooltip.style.padding = '7px 9px';
    lanePreviewTooltip.style.borderRadius = '4px';
    lanePreviewTooltip.style.border = '1px solid rgba(100, 116, 139, 0.45)';
    lanePreviewTooltip.style.background = plot.layout.paper_bgcolor;
    lanePreviewTooltip.style.color = plot.layout.font.color;
    lanePreviewTooltip.style.fontFamily =
      plot.layout.font.family || 'Arial, sans-serif';
    lanePreviewTooltip.style.fontSize = '12px';
    lanePreviewTooltip.style.lineHeight = '1.35';
    lanePreviewTooltip.style.whiteSpace = 'pre-line';
    plot.appendChild(lanePreviewTooltip);
    return lanePreviewTooltip;
  }

  function getLanePreviewConnector() {
    if (lanePreviewConnector !== null && lanePreviewConnector.isConnected) {
      return lanePreviewConnector;
    }
    lanePreviewConnector = document.createElement('div');
    lanePreviewConnector.setAttribute(
      'data-arch-visualizer-lane-tooltip-connector',
      ''
    );
    lanePreviewConnector.style.position = 'absolute';
    lanePreviewConnector.style.display = 'none';
    lanePreviewConnector.style.pointerEvents = 'none';
    lanePreviewConnector.style.zIndex = '1001';
    lanePreviewConnector.style.width = '2px';
    plot.appendChild(lanePreviewConnector);
    return lanePreviewConnector;
  }

  function showLanePreviewTooltip(event, lanePath) {
    atomTooltipActive = false;
    customPathTooltipActive = true;
    getLanePreviewConnector().style.display = 'none';
    const tooltip = getLanePreviewTooltip();
    tooltip.style.borderColor = 'rgba(100, 116, 139, 0.45)';
    tooltip.textContent =
      `${lanePath.previewLabel}\n` +
      `source: ${lanePath.source}\n` +
      `destination: ${lanePath.destination}`;
    tooltip.style.display = 'block';

    const plotRect = plot.getBoundingClientRect();
    const left = Math.min(event.clientX - plotRect.left + 12, plotRect.width - 250);
    const top = Math.min(event.clientY - plotRect.top + 12, plotRect.height - 120);
    tooltip.style.left = `${Math.max(0, left)}px`;
    tooltip.style.top = `${Math.max(0, top)}px`;
  }

  function showMovePathTooltip(segment) {
    atomTooltipActive = false;
    customPathTooltipActive = true;
    const tooltip = getLanePreviewTooltip();
    const connector = getLanePreviewConnector();
    tooltip.textContent =
      `Atom ${segment.atomId} move path\n` +
      `${segment.busLabel}\n` +
      `source: ${segment.source}\n` +
      `destination: ${segment.destination}`;
    tooltip.style.borderColor = segment.color;
    tooltip.style.display = 'block';

    const xAxis = plot._fullLayout.xaxis;
    const yAxis = plot._fullLayout.yaxis;
    if (!xAxis || !yAxis) return;
    const centerX = xAxis._offset + xAxis.l2p(
      (segment.start[0] + segment.end[0]) / 2
    );
    const centerY = yAxis._offset + yAxis.l2p(
      (segment.start[1] + segment.end[1]) / 2
    );
    const connectorHeight = 14;
    const left = centerX - tooltip.offsetWidth / 2;
    const top = centerY - tooltip.offsetHeight - connectorHeight;
    tooltip.style.left = `${Math.max(0, left)}px`;
    tooltip.style.top = `${Math.max(0, top)}px`;
    connector.style.display = 'block';
    connector.style.background = segment.color;
    connector.style.height = `${connectorHeight}px`;
    connector.style.left = `${centerX - 1}px`;
    connector.style.top = `${centerY - connectorHeight}px`;
  }

  function showAtomTooltip(atomData, x, y) {
    atomTooltipActive = true;
    customPathTooltipActive = false;
    getLanePreviewConnector().style.display = 'none';
    const tooltip = getLanePreviewTooltip();
    tooltip.style.borderColor = 'rgba(100, 116, 139, 0.45)';
    tooltip.innerHTML =
      `<b>atom ${atomData[0]}</b><br>` +
      `(zone, word, site): (${atomData[1]}, ${atomData[2]}, ${atomData[3]})<br>` +
      `grid (x, y): (${atomData[4]}, ${atomData[5]})<br>` +
      `position (x, y): (${Number(x).toFixed(3)}, ${Number(y).toFixed(3)}) ` +
      `µm${atomData[6] || ''}`;
    tooltip.style.display = 'block';

    const xAxis = plot._fullLayout.xaxis;
    const yAxis = plot._fullLayout.yaxis;
    if (!xAxis || !yAxis) return;
    const markerX = xAxis._offset + xAxis.l2p(x);
    const markerY = yAxis._offset + yAxis.l2p(y);
    const left = Math.min(markerX + 14, plot.clientWidth - tooltip.offsetWidth - 8);
    const top = Math.min(markerY + 14, plot.clientHeight - tooltip.offsetHeight - 8);
    tooltip.style.left = `${Math.max(0, left)}px`;
    tooltip.style.top = `${Math.max(0, top)}px`;
  }

  function hideLanePreviewTooltip() {
    customPathTooltipActive = false;
    if (lanePreviewTooltip !== null) {
      lanePreviewTooltip.style.display = 'none';
    }
    if (lanePreviewConnector !== null) {
      lanePreviewConnector.style.display = 'none';
    }
  }

  function coordinatePath(xValues, yValues, reverse) {
    const xAxis = plot._fullLayout.xaxis;
    const yAxis = plot._fullLayout.yaxis;
    if (!xAxis || !yAxis) return {pathData: '', pixels: []};

    const pixels = [];
    const start = reverse ? xValues.length - 1 : 0;
    const stop = reverse ? -1 : xValues.length;
    const step = reverse ? -1 : 1;
    let pathData = '';
    for (let index = start; index !== stop; index += step) {
      const x = xValues[index];
      const y = yValues[index];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const pixel = [
        xAxis._offset + xAxis.l2p(x),
        yAxis._offset + yAxis.l2p(y)
      ];
      pathData += `${pixels.length ? 'L' : 'M'}${pixel[0]},${pixel[1]}`;
      pixels.push(pixel);
    }
    return {pathData, pixels};
  }

  function pathDataFromPixels(pixels) {
    return pixels.map(
      (point, index) => `${index ? 'L' : 'M'}${point[0]},${point[1]}`
    ).join('');
  }

  function insetPathEndpoints(pixels, inset) {
    const trimmed = pixels.map((point) => [...point]);
    const totalLength = trimmed.slice(1).reduce((length, point, index) => {
      const previous = trimmed[index];
      return length + Math.hypot(
        point[0] - previous[0],
        point[1] - previous[1]
      );
    }, 0);
    const boundedInset = Math.min(inset, totalLength * 0.2);

    function trimStart(points) {
      let remaining = boundedInset;
      while (points.length >= 2 && remaining > 0) {
        const first = points[0];
        const second = points[1];
        const dx = second[0] - first[0];
        const dy = second[1] - first[1];
        const length = Math.hypot(dx, dy);
        if (length > remaining) {
          first[0] += dx * remaining / length;
          first[1] += dy * remaining / length;
          return;
        }
        remaining -= length;
        points.shift();
      }
    }

    trimStart(trimmed);
    trimmed.reverse();
    trimStart(trimmed);
    trimmed.reverse();
    return trimmed;
  }

  function dashPattern(dash) {
    return {
      dot: '2,5',
      dash: '9,6',
      longdash: '14,7',
      dashdot: '9,5,2,5',
      longdashdot: '14,6,2,6'
    }[dash] || null;
  }

  function drawHighlight(index) {
    clearHighlight();

    const trace = plot.data[index];
    const xAxis = plot._fullLayout.xaxis;
    const yAxis = plot._fullLayout.yaxis;
    const overlayLayer = getPathOverlayLayer();
    if (!trace || !xAxis || !yAxis || !overlayLayer) return;

    let pathData = '';
    let drawing = false;
    for (let pointIndex = 0; pointIndex < trace.x.length; pointIndex++) {
      const x = trace.x[pointIndex];
      const y = trace.y[pointIndex];
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        drawing = false;
        continue;
      }

      const pixelX = xAxis._offset + xAxis.l2p(x);
      const pixelY = yAxis._offset + yAxis.l2p(y);
      pathData += `${drawing ? 'L' : 'M'}${pixelX},${pixelY}`;
      drawing = true;
    }
    if (!pathData) return;

    const line = trace.line || {};
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', pathData);
    path.setAttribute('data-arch-visualizer-bus-highlight', '');
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', line.color || '#f59e0b');
    path.setAttribute(
      'stroke-width',
      String(Math.max(6, Number(line.width || 2.25) * 2.6))
    );
    path.setAttribute('stroke-opacity', '1');
    path.setAttribute('stroke-linecap', 'round');
    path.setAttribute('stroke-linejoin', 'round');
    path.setAttribute('pointer-events', 'none');
    const strokeDasharray = dashPattern(line.dash);
    if (strokeDasharray !== null) {
      path.setAttribute('stroke-dasharray', strokeDasharray);
    }

    // Keep the highlight above every WebGL trace, including exactly
    // overlapping buses, but below Plotly's labels and tooltip text.
    overlayLayer.appendChild(path);
    highlightPath = path;
  }

  function drawSiteLaneOverlays(customdataValues) {
    clearSiteLaneOverlays();
    const overlayLayer = getPathOverlayLayer();
    if (!overlayLayer) return;
    const drawnPathIndices = new Set();

    customdataValues.forEach((customdata) => {
      if (!customdata || customdata.length < 3) return;
      const key = `${customdata[0]},${customdata[1]},${customdata[2]}`;
      const refs = siteLanePathRefs[key] || [];

      refs.forEach(([pathIndex, reverse]) => {
        // The same lane can touch two selected sites. Draw it once so its
        // opacity does not depend on how many selected atoms share it.
        if (drawnPathIndices.has(pathIndex)) return;
        drawnPathIndices.add(pathIndex);

        const lanePath = siteLanePaths[pathIndex];
        if (!lanePath) return;
        const xValues = currentPathStyle === 'cartoon'
          ? lanePath.cartoonX
          : lanePath.exactX;
        const yValues = currentPathStyle === 'cartoon'
          ? lanePath.cartoonY
          : lanePath.exactY;
        const {pathData, pixels} = coordinatePath(
          xValues,
          yValues,
          reverse
        );
        if (!pathData || pixels.length < 2) return;

        const path = document.createElementNS(
          'http://www.w3.org/2000/svg',
          'path'
        );
        path.setAttribute('d', pathData);
        path.setAttribute('data-arch-visualizer-site-lane', '');
        path.setAttribute('data-arch-visualizer-lane-path-index', pathIndex);
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', lanePath.color);
        path.setAttribute('stroke-width', '4');
        path.setAttribute('stroke-opacity', '0.42');
        path.setAttribute('stroke-linecap', 'round');
        path.setAttribute('stroke-linejoin', 'round');
        path.setAttribute('pointer-events', 'none');

        // Keep the colored preview visually connected to its sites, but trim
        // its transparent hover target away from the markers so atom/site
        // hover remains unambiguous at either endpoint.
        const hitPixels = insetPathEndpoints(pixels, 12);
        const hitPath = document.createElementNS(
          'http://www.w3.org/2000/svg',
          'path'
        );
        hitPath.setAttribute('d', pathDataFromPixels(hitPixels));
        hitPath.setAttribute('data-arch-visualizer-site-lane-hit-target', '');
        hitPath.setAttribute('fill', 'none');
        hitPath.setAttribute('stroke', 'transparent');
        hitPath.setAttribute('stroke-width', '14');
        hitPath.setAttribute('pointer-events', 'stroke');
        hitPath.style.cursor = 'default';
        hitPath.addEventListener('mouseenter', (event) => {
          showLanePreviewTooltip(event, lanePath);
        });
        hitPath.addEventListener('mousemove', (event) => {
          showLanePreviewTooltip(event, lanePath);
        });
        hitPath.addEventListener('mouseleave', hideLanePreviewTooltip);
        const strokeDasharray = dashPattern(lanePath.dash);
        if (strokeDasharray !== null) {
          path.setAttribute('stroke-dasharray', strokeDasharray);
        }
        overlayLayer.appendChild(path);
        overlayLayer.appendChild(hitPath);
        siteLaneOverlays.push(path, hitPath);
      });
    });
  }

  function drawMovePathHoverOverlays() {
    clearMovePathHoverOverlays();
    const overlayLayer = getPathOverlayLayer();
    if (!overlayLayer) return;

    plot.data.forEach((trace) => {
      const meta = trace.meta || {};
      if (
        meta.bloqadeTraceKind !== 'movePath' ||
        trace.visible === false ||
        trace.visible === 'legendonly'
      ) return;

      (meta.segments || []).forEach((segment) => {
        const {pixels} = coordinatePath(
          [segment.start[0], segment.end[0]],
          [segment.start[1], segment.end[1]],
          false
        );
        if (pixels.length < 2) return;

        const hitPath = document.createElementNS(
          'http://www.w3.org/2000/svg',
          'path'
        );
        hitPath.setAttribute(
          'd',
          pathDataFromPixels(insetPathEndpoints(pixels, 12))
        );
        hitPath.setAttribute('data-bloqade-move-path-hover-target', '');
        hitPath.setAttribute('fill', 'none');
        hitPath.setAttribute('stroke', 'rgba(0, 0, 0, 0.001)');
        hitPath.setAttribute('stroke-width', '16');
        hitPath.setAttribute('stroke-linecap', 'round');
        hitPath.setAttribute('pointer-events', 'stroke');
        hitPath.style.cursor = 'default';
        hitPath.addEventListener('mouseenter', () => {
          showMovePathTooltip(segment);
        });
        hitPath.addEventListener('mousemove', () => {
          showMovePathTooltip(segment);
        });
        hitPath.addEventListener('mouseleave', hideLanePreviewTooltip);
        overlayLayer.appendChild(hitPath);
        movePathHoverOverlays.push(hitPath);
      });
    });
  }

  function siteCustomdataAt(x, y) {
    const siteTrace = plot.data[siteTraceIndex];
    if (!siteTrace || !Number.isFinite(x) || !Number.isFinite(y)) return null;

    const coordinateTolerance = 1e-9;
    for (let pointIndex = 0; pointIndex < siteTrace.x.length; pointIndex++) {
      if (
        Math.abs(siteTrace.x[pointIndex] - x) <= coordinateTolerance &&
        Math.abs(siteTrace.y[pointIndex] - y) <= coordinateTolerance
      ) {
        return siteTrace.customdata[pointIndex];
      }
    }
    return null;
  }

  function siteCustomdataNearPointer(event) {
    const siteTrace = plot.data[siteTraceIndex];
    const xAxis = plot._fullLayout.xaxis;
    const yAxis = plot._fullLayout.yaxis;
    if (!siteTrace || !xAxis || !yAxis) return null;

    const plotRect = plot.getBoundingClientRect();
    const pointerX = event.clientX - plotRect.left;
    const pointerY = event.clientY - plotRect.top;
    const hoverRadiusSquared = 14 * 14;
    let nearestCustomdata = null;
    let nearestDistanceSquared = hoverRadiusSquared;

    for (let pointIndex = 0; pointIndex < siteTrace.x.length; pointIndex++) {
      const x = siteTrace.x[pointIndex];
      const y = siteTrace.y[pointIndex];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;

      const siteX = xAxis._offset + xAxis.l2p(x);
      const siteY = yAxis._offset + yAxis.l2p(y);
      const dx = siteX - pointerX;
      const dy = siteY - pointerY;
      const distanceSquared = dx * dx + dy * dy;
      if (distanceSquared <= nearestDistanceSquared) {
        nearestDistanceSquared = distanceSquared;
        nearestCustomdata = siteTrace.customdata[pointIndex];
      }
    }
    return nearestCustomdata;
  }

  function siteCustomdataFromPlotlyEvent(event) {
    const points = event.points || [];
    const sitePoint = points.find(
      (item) => item.curveNumber === siteTraceIndex
    );
    if (sitePoint) return sitePoint.customdata;

    const atomPoint = points.find((item) => {
      const trace = plot.data[item.curveNumber];
      return trace && trace.meta && trace.meta.bloqadeTraceKind === 'atom';
    });
    if (atomPoint && atomPoint.customdata && atomPoint.customdata.length >= 6) {
      // Atom data starts with the atom ID; the remaining values match the
      // architecture site's (zone, word, site, grid-x, grid-y) data.
      return atomPoint.customdata.slice(1);
    }

    // Animation can leave a gate region or move-path point as Plotly's click
    // winner even when the pointer is visibly over an atom/site. Resolve the
    // physical pointer position before relying on that winning trace's data.
    const pointerEvent = event.event;
    if (pointerEvent) {
      const nearbySite = siteCustomdataNearPointer(pointerEvent);
      if (nearbySite) return nearbySite;
    }

    return points
      .filter((item) => !isCircuitPoint(item))
      .map((item) => siteCustomdataAt(item.x, item.y))
      .find((customdata) => customdata !== null);
  }

  // Executed-circuit traces live on their own axis pair above the
  // architecture. Their integer column/row coordinates can coincide with
  // site positions, so never resolve them as architecture sites.
  function isCircuitPoint(point) {
    const trace = plot.data[point.curveNumber];
    return Boolean(trace && trace.xaxis && trace.xaxis !== 'x');
  }

  function jumpToDebuggerStep(stepIndex) {
    const frameName = debuggerFrameNames[stepIndex];
    if (frameName === undefined) return;
    currentDebuggerStep = stepIndex;
    pendingDebuggerJump = frameName;
    setDebuggerSlider(stepIndex);
    // Plotly rejects the previous animation's promise when a new immediate
    // animation interrupts it; that is expected when clicking quickly.
    window.Plotly.animate(plot, [frameName], {
      mode: 'immediate',
      frame: {duration: 0, redraw: true},
      transition: {duration: 0}
    }).catch(function () {}).finally(function () {
      if (pendingDebuggerJump === frameName) pendingDebuggerJump = null;
    });
  }

  function atomHoverTargetForSite(siteCustomdata) {
    if (!siteCustomdata) return null;
    for (let traceIndex = 0; traceIndex < plot.data.length; traceIndex++) {
      const trace = plot.data[traceIndex];
      if (!trace.meta || trace.meta.bloqadeTraceKind !== 'atom') continue;
      const customdataValues = trace.customdata || [];
      for (let pointIndex = 0; pointIndex < customdataValues.length; pointIndex++) {
        const atomData = customdataValues[pointIndex];
        if (
          atomData &&
          atomData[1] === siteCustomdata[0] &&
          atomData[2] === siteCustomdata[1] &&
          atomData[3] === siteCustomdata[2]
        ) {
          return {
            curveNumber: traceIndex,
            pointNumber: pointIndex,
            customdata: atomData
          };
        }
      }
    }
    return null;
  }

  function activateSiteLaneOverlays(customdata) {
    const key = customdata
      ? `${customdata[0]},${customdata[1]},${customdata[2]}`
      : null;
    const overlaysAreConnected = siteLaneOverlays.some(
      (element) => element.isConnected
    );
    if (key === activeSiteKey && (key === null || overlaysAreConnected)) return;

    activeSiteKey = key;
    if (customdata) {
      drawSiteLaneOverlays([customdata]);
    } else {
      clearSiteLaneOverlays();
    }
  }

  function redrawSelectedSiteLaneOverlays() {
    drawSiteLaneOverlays([...selectedSites.values()]);
    // Executed move paths are more specific than available-lane previews.
    // Reappend their transparent hit targets above coincident preview lanes.
    drawMovePathHoverOverlays();
  }

  function toggleSelectedSite(customdata) {
    previewMutationVersion += 1;
    const key = `${customdata[0]},${customdata[1]},${customdata[2]}`;
    if (selectedSites.has(key)) {
      selectedSites.delete(key);
    } else {
      selectedSites.set(key, customdata);
    }
    redrawSelectedSiteLaneOverlays();
    commitPreviewState();
  }

  function capturePreviewState() {
    return {
      buses: [...busVisibility],
      sites: [...selectedSites.values()].map((customdata) => [...customdata])
    };
  }

  function previewStatesMatch(left, right) {
    return JSON.stringify(left) === JSON.stringify(right);
  }

  function updatePreviewHistoryButtons() {
    if (undoPreviewButton !== null) {
      undoPreviewButton.disabled = previewHistoryIndex <= 0;
    }
    if (redoPreviewButton !== null) {
      redoPreviewButton.disabled =
        previewHistoryIndex < 0 ||
        previewHistoryIndex >= previewHistory.length - 1;
    }
  }

  function commitPreviewState(state = capturePreviewState()) {
    const currentState = previewHistory[previewHistoryIndex];
    if (currentState && previewStatesMatch(state, currentState)) return;

    previewHistory.splice(previewHistoryIndex + 1);
    previewHistory.push(state);
    previewHistoryIndex = previewHistory.length - 1;
    updatePreviewHistoryButtons();
  }

  function syncBusCheckboxes() {
    busControls.forEach((control, index) => {
      const checkbox = busCheckboxes.get(control.traceIndex);
      if (checkbox) checkbox.checked = busVisibility[index];
    });
  }

  function setDebuggerSlider(frameIndex, force = false) {
    if (!plot.layout.sliders || !plot.layout.sliders.length) {
      return Promise.resolve();
    }
    // During Play, Plotly can change the layout value before this event runs.
    // Still request the slider redraw for that frame.
    if (!force && plot.layout.sliders[0].active === frameIndex) {
      return Promise.resolve();
    }
    return window.Plotly.relayout(
      plot,
      {'sliders[0].active': frameIndex}
    );
  }

  function restorePreviewState(historyIndex) {
    if (historyIndex < 0 || historyIndex >= previewHistory.length) return;
    const state = previewHistory[historyIndex];
    const mutationVersion = ++previewMutationVersion;
    previewHistoryIndex = historyIndex;

    selectedSites.clear();
    state.sites.forEach((customdata) => {
      const key = `${customdata[0]},${customdata[1]},${customdata[2]}`;
      selectedSites.set(key, customdata);
    });
    busVisibility.splice(0, busVisibility.length, ...state.buses);

    const traceIndices = busControls.map((control) => control.traceIndex);
    const visible = busVisibility.map((isVisible) =>
      isVisible ? true : 'legendonly'
    );
    syncBusCheckboxes();
    redrawSelectedSiteLaneOverlays();
    updatePreviewHistoryButtons();
    const restyle = traceIndices.length
      ? window.Plotly.restyle(plot, {visible}, traceIndices)
      : Promise.resolve();
    restyle.then(() => {
      if (mutationVersion !== previewMutationVersion) return;
      syncBusCheckboxes();
      redrawSelectedSiteLaneOverlays();
      updatePreviewHistoryButtons();
    });
  }

  function setAllBusPreviews(visible, clearSites) {
    const mutationVersion = ++previewMutationVersion;
    if (clearSites) {
      selectedSites.clear();
      activeSiteKey = null;
      clearSiteLaneOverlays();
      clearHighlight();
    }

    busVisibility.fill(visible);
    const traceIndices = busControls.map((control) => control.traceIndex);
    const visibility = traceIndices.map(() =>
      visible ? true : 'legendonly'
    );
    syncBusCheckboxes();
    commitPreviewState();
    const restyle = traceIndices.length
      ? window.Plotly.restyle(
          plot,
          {visible: visibility},
          traceIndices
        )
      : Promise.resolve();
    restyle.then(() => {
      if (mutationVersion !== previewMutationVersion) return;
      syncBusCheckboxes();
      redrawSelectedSiteLaneOverlays();
    });
  }

  function installBusMultiselectors() {
    if (plot.querySelector('[data-arch-visualizer-bus-selectors]')) return;

    plot.style.position = 'relative';
    const panel = document.createElement('div');
    panel.setAttribute('data-arch-visualizer-bus-selectors', '');
    panel.style.position = 'absolute';
    panel.style.top = '92px';
    panel.style.right = '12px';
    panel.style.width = '236px';
    panel.style.zIndex = '1001';
    panel.style.color = plot.layout.font.color;
    panel.style.fontFamily = plot.layout.font.family || 'Arial, sans-serif';
    panel.style.fontSize = '12px';

    const groups = [
      ['site', 'Site Buses'],
      ['word', 'Word Buses'],
      ['zone', 'Zone Buses']
    ];

    const historyControls = document.createElement('div');
    historyControls.style.display = 'grid';
    historyControls.style.gridTemplateColumns = '1fr 1fr';
    historyControls.style.gap = '6px';
    historyControls.style.marginBottom = '8px';

    function historyButton(label, title, onClick) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = label;
      button.title = title;
      button.setAttribute('aria-label', title);
      button.style.padding = '6px 8px';
      button.style.cursor = 'pointer';
      button.style.color = plot.layout.font.color;
      button.style.background = plot.layout.paper_bgcolor;
      button.style.border = '1px solid rgba(100, 116, 139, 0.45)';
      button.style.borderRadius = '4px';
      button.addEventListener('click', onClick);
      historyControls.appendChild(button);
      return button;
    }

    undoPreviewButton = historyButton(
      'Undo',
      'Undo bus or lane visibility change',
      () => restorePreviewState(previewHistoryIndex - 1)
    );
    redoPreviewButton = historyButton(
      'Redo',
      'Redo bus or lane visibility change',
      () => restorePreviewState(previewHistoryIndex + 1)
    );
    panel.appendChild(historyControls);

    groups.forEach(([kind, heading]) => {
      const controls = busControls.filter((control) => control.kind === kind);
      const details = document.createElement('details');
      details.style.marginBottom = '8px';

      const summary = document.createElement('summary');
      summary.textContent = heading;
      summary.style.cursor = 'pointer';
      summary.style.padding = '7px 9px';
      summary.style.background = plot.layout.paper_bgcolor;
      summary.style.border = '1px solid rgba(100, 116, 139, 0.45)';
      summary.style.borderRadius = '4px';
      summary.style.userSelect = 'none';
      details.appendChild(summary);

      const options = document.createElement('div');
      options.style.maxHeight = '235px';
      options.style.overflowY = 'auto';
      options.style.padding = '6px 7px';
      options.style.background = plot.layout.paper_bgcolor;
      options.style.border = '1px solid rgba(100, 116, 139, 0.35)';
      options.style.borderTop = '0';

      if (!controls.length) {
        const empty = document.createElement('div');
        empty.textContent = `No ${heading.toLowerCase()}`;
        empty.style.opacity = '0.72';
        empty.style.padding = '4px';
        options.appendChild(empty);
      }

      controls.forEach((control) => {
        const label = document.createElement('label');
        label.style.display = 'flex';
        label.style.alignItems = 'center';
        label.style.gap = '7px';
        label.style.padding = '4px 2px';
        label.style.cursor = 'pointer';
        label.title = `${heading.slice(0, -2)} ID ${control.busId}`;
        label.addEventListener('mouseenter', () => drawHighlight(control.traceIndex));
        label.addEventListener('mouseleave', clearHighlight);

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        const controlIndex = busControls.findIndex(
          (candidate) => candidate.traceIndex === control.traceIndex
        );
        checkbox.checked = busVisibility[controlIndex];
        checkbox.setAttribute(
          'aria-label',
          `${heading.slice(0, -2)} ID ${control.busId}`
        );
        checkbox.addEventListener('change', () => {
          const mutationVersion = ++previewMutationVersion;
          if (controlIndex >= 0) {
            busVisibility[controlIndex] = checkbox.checked;
          }
          commitPreviewState();
          window.Plotly.restyle(
            plot,
            {visible: checkbox.checked ? true : 'legendonly'},
            [control.traceIndex]
          ).then(() => {
            if (mutationVersion !== previewMutationVersion) return;
            syncBusCheckboxes();
            if (siteLanePreviewMode === 'click' && selectedSites.size) {
              redrawSelectedSiteLaneOverlays();
            }
          });
        });
        busCheckboxes.set(control.traceIndex, checkbox);

        const swatch = document.createElement('span');
        swatch.setAttribute('data-arch-visualizer-bus-color', control.color);
        swatch.style.display = 'inline-block';
        swatch.style.width = '12px';
        swatch.style.height = '12px';
        swatch.style.flex = '0 0 12px';
        swatch.style.borderRadius = '50%';
        swatch.style.background = control.color;

        const text = document.createElement('span');
        text.textContent = `ID ${control.busId} · ${control.label}`;

        label.appendChild(checkbox);
        label.appendChild(swatch);
        label.appendChild(text);
        options.appendChild(label);
      });

      details.appendChild(options);
      panel.appendChild(details);
    });

    plot.appendChild(panel);
    plot.on('plotly_restyle', function () {
      const mutationVersion = previewMutationVersion;
      window.setTimeout(() => {
        if (mutationVersion !== previewMutationVersion) return;
        syncBusCheckboxes();
        if (siteLanePreviewMode === 'click' && selectedSites.size) {
          redrawSelectedSiteLaneOverlays();
        }
      }, 0);
    });
  }

  installBusMultiselectors();
  previewHistory = [capturePreviewState()];
  previewHistoryIndex = 0;
  updatePreviewHistoryButtons();
  window.setTimeout(drawMovePathHoverOverlays, 0);

  // Keep direct Plotly animation calls synchronized too (for example, a
  // caller using ``Plotly.animate`` outside the built-in controls).
  if (debuggerFrameNames.length) {
    plot.on('plotly_animatingframe', function (event) {
      const frameName = event.name || (event.frame && event.frame.name);
      const frameIndex = debuggerFrameNames.indexOf(frameName);
      if (frameIndex >= 0) {
        currentDebuggerStep = frameIndex;
        if (frameName !== pendingDebuggerJump) {
          setDebuggerSlider(frameIndex, true);
        }
        pendingDebuggerJump = null;
      }
    });
    // Clicking a gate in the executed-circuit panel jumps the debugger to
    // the step that applies it and updates the slider directly.
    plot.on('plotly_click', function (event) {
      const gatePoint = (event.points || []).find((item) => {
        const trace = plot.data[item.curveNumber];
        return trace && trace.meta && trace.meta.bloqadeTraceKind === 'circuitGate';
      });
      if (!gatePoint || !gatePoint.customdata) return;
      jumpToDebuggerStep(gatePoint.customdata[0]);
    });
  }

  if (interactiveDebugger) {
    plot.setAttribute('tabindex', '0');
    plot.addEventListener('click', function (event) {
      const target = event.target;
      if (target && target.closest && target.closest(
        'input, button, select, textarea, [contenteditable], ' +
        '[data-arch-visualizer-bus-selectors]'
      )) return;
      plot.focus({preventScroll: true});
    });
    plot.addEventListener('keydown', function (event) {
      if (event.target !== plot) return;
      const stepDelta = event.key === 'ArrowLeft' || event.key === 'ArrowUp'
        ? -1
        : event.key === 'ArrowRight' || event.key === 'ArrowDown'
          ? 1
          : 0;
      if (stepDelta === 0) return;

      event.preventDefault();
      const nextStep = Math.min(
        Math.max(currentDebuggerStep + stepDelta, 0),
        debuggerFrameNames.length - 1
      );
      jumpToDebuggerStep(nextStep);
    });
  }

  // Reserve wheel zoom for trackpad pinch, which browsers report as a wheel
  // event with ``ctrlKey`` set. A plain scroll wheel or two-finger scroll
  // would otherwise zoom the plot as a notebook or page scrolls past it.
  // Stopping propagation in the capture phase keeps the event away from
  // Plotly's drag-layer handler without preventing the page scroll.
  plot.addEventListener('wheel', function (event) {
    if (!event.ctrlKey) event.stopPropagation();
  }, true);

  // Resolve site previews from pointer position instead of relying solely on
  // Plotly's hover winner. A visible or recently restyled WebGL bus can win
  // hover arbitration over a coincident SVG site marker, even after that bus
  // is hidden again. Direct proximity keeps site previews independent of bus
  // selection state and also lets them coexist with selected buses.
  if (siteLanePreviewMode === 'hover') {
    plot.addEventListener('mousemove', function (event) {
      activateSiteLaneOverlays(siteCustomdataNearPointer(event));
    });
    plot.addEventListener('mouseleave', function () {
      activateSiteLaneOverlays(null);
    });
  } else {
    // Plotly replaces its SVG nodes during restyle operations such as
    // undo/redo. Its transparent move-path hover targets can consequently sit
    // above an atom marker and prevent Plotly from emitting ``plotly_click``.
    // Resolve ordinary DOM clicks in the capture phase so site selection is
    // independent of whichever regenerated SVG node wins pointer dispatch.
    plot.addEventListener('click', function (event) {
      const siteCustomdata = siteCustomdataNearPointer(event);
      if (!siteCustomdata) return;

      toggleSelectedSite(siteCustomdata);
    }, true);
  }

  plot.addEventListener('mousemove', function (event) {
    if (!atomTooltipActive) return;
    const nearbySite = siteCustomdataNearPointer(event);
    if (atomHoverTargetForSite(nearbySite)) return;

    atomTooltipActive = false;
    hideLanePreviewTooltip();
    const plotlyHoverLayer = plot.querySelector('.hoverlayer');
    if (plotlyHoverLayer) plotlyHoverLayer.style.display = '';
  });
  plot.addEventListener('mouseleave', function () {
    atomTooltipActive = false;
    hideLanePreviewTooltip();
  });

  // Handle custom overlay state from the native pointer click. Plotly's
  // ``plotly_buttonclicked`` event is not emitted consistently after every
  // animation/restyle, while the SVG update-menu button always receives this
  // DOM event. Plotly's configured button method still updates trace data.
  plot.addEventListener('click', function (event) {
    const target = event.target;
    const button = target && target.closest
      ? target.closest('g.updatemenu-button')
      : null;
    if (!button) return;

    const label = (button.textContent || '').trim();
    if (label === 'Clear all buses') {
      setAllBusPreviews(false, true);
    } else if (label === 'Show all buses') {
      setAllBusPreviews(true, true);
    } else if (label === 'Exact paths' || label === 'Cartoon paths') {
      currentPathStyle = label === 'Cartoon paths' ? 'cartoon' : 'exact';
      // A path-style update replaces Plotly's upper SVG. Redraw previews only
      // after that update finishes so they land in the replacement layer.
      clearSiteLaneOverlays();
      if (siteLanePreviewMode === 'click' && selectedSites.size) {
        window.setTimeout(redrawSelectedSiteLaneOverlays, 0);
      }
    }
  }, true);

  plot.on('plotly_afterplot', function () {
    window.setTimeout(() => {
      if (siteLanePreviewMode === 'click' && selectedSites.size) {
        redrawSelectedSiteLaneOverlays();
      } else {
        drawMovePathHoverOverlays();
      }
    }, 0);
  });

  plot.on('plotly_hover', function (event) {
    clearHighlight();
    const plotlyHoverLayer = plot.querySelector('.hoverlayer');
    if (customPathTooltipActive) {
      if (plotlyHoverLayer) plotlyHoverLayer.style.display = 'none';
      return;
    }
    if (plotlyHoverLayer) plotlyHoverLayer.style.display = '';
    hideLanePreviewTooltip();
    // Resolve previews before the atom-tooltip branches return. In the
    // debugger, either the atom or its coincident SLM site can win the hover;
    // both must expose the site's available lanes as well as the atom label.
    const siteCustomdata = siteCustomdataFromPlotlyEvent(event);
    if (siteLanePreviewMode === 'hover' && siteCustomdata) {
      activateSiteLaneOverlays(siteCustomdata);
    }
    const directSitePoint = event.points.find(
      (item) => item.curveNumber === siteTraceIndex
    );
    const directAtomPoint = event.points.find((item) => {
      const trace = plot.data[item.curveNumber];
      return trace && trace.meta && trace.meta.bloqadeTraceKind === 'atom';
    });
    if (directAtomPoint) {
      // Plotly can return both the animated atom and the static SLM site for
      // coincident points, rendering one native label for each. Replace that
      // whole native hover layer with the atom-only tooltip.
      if (plotlyHoverLayer) plotlyHoverLayer.style.display = 'none';
      showAtomTooltip(
        directAtomPoint.customdata,
        directAtomPoint.x,
        directAtomPoint.y
      );
      return;
    }
    if (directSitePoint) {
      const atomTarget = atomHoverTargetForSite(directSitePoint.customdata);
      if (atomTarget) {
        // Plotly breaks coincident-point hover ties in favor of the static
        // site trace. Hide that tooltip and show the animated atom's data
        // directly so an occupied site has exactly one tooltip: the atom's.
        if (plotlyHoverLayer) plotlyHoverLayer.style.display = 'none';
        showAtomTooltip(
          atomTarget.customdata,
          directSitePoint.x,
          directSitePoint.y
        );
        return;
      }
    }
    const busPoint = event.points.find(
      (item) => busIndexSet.has(item.curveNumber)
    );
    // Once a bus is visible, Plotly may report its endpoint instead of the
    // coincident site marker. Treat an endpoint at a site coordinate as a site
    // hover so lane previews are independent of bus visibility.
    if (!(siteLanePreviewMode === 'hover' && siteCustomdata) && busPoint) {
      drawHighlight(busPoint.curveNumber);
    }
    if (
      siteLanePreviewMode === 'click' &&
      selectedSites.size &&
      siteLaneOverlays.some((element) => !element.isConnected)
    ) {
      redrawSelectedSiteLaneOverlays();
    }
  });

  plot.on('plotly_unhover', function () {
    clearHighlight();
    const plotlyHoverLayer = plot.querySelector('.hoverlayer');
    if (!atomTooltipActive) {
      if (plotlyHoverLayer) plotlyHoverLayer.style.display = '';
      hideLanePreviewTooltip();
    }
    if (siteLanePreviewMode === 'click' && selectedSites.size) {
      window.setTimeout(redrawSelectedSiteLaneOverlays, 0);
    }
  });
  plot.on('plotly_relayout', function () {
    const layoutPathStyle =
      (plot.layout.meta || {}).archVisualizerPathStyle;
    if (layoutPathStyle === 'exact' || layoutPathStyle === 'cartoon') {
      currentPathStyle = layoutPathStyle;
    }
    clearHighlight();
    clearSiteLaneOverlays();
    clearMovePathHoverOverlays();
    if (siteLanePreviewMode === 'hover') {
      activeSiteKey = null;
    } else if (selectedSites.size) {
      window.setTimeout(redrawSelectedSiteLaneOverlays, 0);
    }
    window.setTimeout(drawMovePathHoverOverlays, 0);
  });
  plot.on('plotly_animated', function () {
    window.setTimeout(() => {
      if (siteLanePreviewMode === 'click' && selectedSites.size) {
        redrawSelectedSiteLaneOverlays();
      } else {
        drawMovePathHoverOverlays();
      }
    }, 0);
  });
})();
