// Execute the packaged controller with a minimal DOM/Plotly event fixture.
// Browser rendering remains a separate integration check.
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const path = require('node:path');
const {test} = require('node:test');
const vm = require('node:vm');

const source = readFileSync(path.join(
  __dirname, '../../../src/bloqade/gemini/visualize/_arch_interactive.js'
), 'utf8');

function fixture(occupied, debuggerOptions = {}) {
  const {
    frameNames = ['step-0'],
    hasSlider = false,
    activeStep = 0
  } = debuggerOptions;
  class Element {
    constructor() {
      this.style = {};
      this.attributes = {};
      this.children = [];
      this.isConnected = true;
      this.offsetWidth = 100;
      this.offsetHeight = 40;
    }
    setAttribute(name, value) { this.attributes[name] = value; }
    addEventListener() {}
    appendChild(child) { this.children.push(child); }
    remove() { this.isConnected = false; }
  }
  const overlay = new Element();
  const hover = new Element();
  const events = {};
  const domEvents = {};
  const siteData = [0, 0, 0, 0, 0];
  const atomData = [7, ...siteData, ''];
  const plot = new Element();
  plot.focus = () => { plot.focused = true; };
  plot.clientWidth = 600;
  plot.clientHeight = 400;
  plot.layout = {font: {color: 'black'}, meta: {
    archVisualizerSiteTraceIndex: 0,
    archVisualizerSiteLanePreviewMode: 'hover',
    archVisualizerSiteLanePathRefs: {'0,0,0': [[0, false]]},
    archVisualizerSiteLanePaths: [{
      exactX: [0, 1], exactY: [0, 0], color: 'red', dash: 'dash'
    }],
    bloqadePlotlyDebugger: {frameNames}
  }};
  if (hasSlider) {
    plot.layout.sliders = [{active: activeStep}];
  }
  plot.data = [{x: [0], y: [0], customdata: [siteData]}];
  if (occupied) {
    plot.data.push({meta: {bloqadeTraceKind: 'atom'}, customdata: [atomData]});
  }
  plot._fullLayout = {
    xaxis: {_offset: 10, l2p: (x) => 100 * x},
    yaxis: {_offset: 20, l2p: (y) => 100 * y}
  };
  plot.querySelector = (selector) => {
    if (selector === '[data-arch-visualizer-path-overlays]') return overlay;
    if (selector === '.hoverlayer') return hover;
    // Bus panel construction is unrelated to the hover event under test.
    if (selector === '[data-arch-visualizer-bus-selectors]') return new Element();
    return null;
  };
  plot.on = (name, handler) => { (events[name] ??= []).push(handler); };
  const domOptions = {};
  plot.addEventListener = (name, handler, options) => {
    (domEvents[name] ??= []).push(handler);
    (domOptions[name] ??= []).push(options);
  };
  plot.getBoundingClientRect = () => ({left: 0, top: 0});
  const animateCalls = [];
  const relayoutCalls = [];
  vm.runInNewContext(source, {
    document: {
      getElementById: () => plot,
      createElement: () => new Element(),
      createElementNS: () => new Element()
    },
    window: {
      setTimeout: (callback) => callback(),
      Plotly: {
        animate: (_plot, frames, options) => {
          animateCalls.push({frames, options});
          return Promise.resolve();
        },
        relayout: (_plot, update) => {
          relayoutCalls.push(update);
          if (plot.layout.sliders && 'sliders[0].active' in update) {
            plot.layout.sliders[0].active = update['sliders[0].active'];
          }
          return Promise.resolve();
        }
      }
    }
  });
  return {
    plot, siteData, atomData, domEvents, domOptions, animateCalls, relayoutCalls,
    emit: (name, event) => events[name].forEach((handler) => handler(event)),
    move: (event) => domEvents.mousemove.forEach((handler) => handler(event)),
    leave: () => domEvents.mouseleave.forEach((handler) => handler()),
    click: (target) => domEvents.click.forEach((handler) => handler({target})),
    keydown: (event) => domEvents.keydown.forEach((handler) => handler(event)),
    lanes: () => overlay.children.filter((element) =>
      element.isConnected && 'data-arch-visualizer-site-lane' in element.attributes
    )
  };
}

function keyEvent(key, target) {
  return {
    key,
    target,
    prevented: false,
    preventDefault() { this.prevented = true; }
  };
}

test('interactive multi-frame debugger makes only the plot focusable', () => {
  const f = fixture(false, {
    frameNames: ['step-0', 'step-1'], hasSlider: true
  });

  assert.equal(f.plot.attributes.tabindex, '0');
  assert.equal(f.domEvents.keydown.length, 1);
});

test('clicking the debugger plot gives arrow keys figure focus', () => {
  const f = fixture(false, {
    frameNames: ['step-0', 'step-1'], hasSlider: true
  });
  f.click({closest: () => null});

  assert.equal(f.plot.focused, true);
});

test('clicking a nested bus control does not take its keyboard focus', () => {
  const f = fixture(false, {
    frameNames: ['step-0', 'step-1'], hasSlider: true
  });
  f.click({closest: (selector) => selector.includes('input') ? {} : null});

  assert.equal(f.plot.focused, undefined);
});

test('debugger arrow keys navigate every previous and next mapping', () => {
  const f = fixture(false, {
    frameNames: ['step-0', 'step-1', 'step-2'], hasSlider: true, activeStep: 1
  });
  const expectedFrameForKey = {
    ArrowLeft: 'step-0', ArrowUp: 'step-0',
    ArrowRight: 'step-2', ArrowDown: 'step-2'
  };

  for (const [key, frameName] of Object.entries(expectedFrameForKey)) {
    f.animateCalls.length = 0;
    f.plot.layout.sliders[0].active = 1;
    f.emit('plotly_animatingframe', {name: 'step-1'});
    const event = keyEvent(key, f.plot);
    f.keydown(event);
    assert.equal(f.animateCalls.length, 1, key);
    assert.deepEqual(plain(f.animateCalls[0].frames), [frameName], key);
    assert.equal(event.prevented, true, key);
  }
});

test('debugger arrow navigation starts at the active slider step', () => {
  const f = fixture(false, {
    frameNames: ['step-0', 'step-1', 'step-2'], hasSlider: true, activeStep: 1
  });
  const event = keyEvent('ArrowRight', f.plot);

  f.keydown(event);

  assert.deepEqual(plain(f.animateCalls[0].frames), ['step-2']);
  assert.deepEqual(plain(f.relayoutCalls), [{'sliders[0].active': 2}]);
});

test('debugger arrow navigation clamps at both ends and prevents scrolling', () => {
  const f = fixture(false, {
    frameNames: ['step-0', 'step-1'], hasSlider: true
  });

  const first = keyEvent('ArrowLeft', f.plot);
  f.keydown(first);
  const last = keyEvent('ArrowRight', f.plot);
  f.keydown(last);
  const beyondLast = keyEvent('ArrowDown', f.plot);
  f.keydown(beyondLast);

  assert.deepEqual(
    f.animateCalls.map((call) => plain(call.frames)),
    [['step-0'], ['step-1'], ['step-1']]
  );
  assert.equal(first.prevented, true);
  assert.equal(last.prevented, true);
  assert.equal(beyondLast.prevented, true);
});

test('debugger arrow navigation ignores nested controls without preventing default', () => {
  const f = fixture(false, {
    frameNames: ['step-0', 'step-1'], hasSlider: true
  });
  const event = keyEvent('ArrowRight', {});

  f.keydown(event);

  assert.equal(f.animateCalls.length, 0);
  assert.equal(event.prevented, false);
});

test('single-frame and autoplay debuggers remain unfocusable and inactive', () => {
  const singleFrame = fixture(false, {
    frameNames: ['step-0'], hasSlider: true
  });
  const autoplay = fixture(false, {
    frameNames: ['step-0', 'step-1']
  });

  for (const f of [singleFrame, autoplay]) {
    assert.equal(f.plot.attributes.tabindex, undefined);
    assert.equal(f.domEvents.keydown, undefined);
  }
});

test('direct frame animation synchronizes the debugger arrow navigation state', () => {
  const f = fixture(false, {
    frameNames: ['step-0', 'step-1', 'step-2'], hasSlider: true
  });
  f.emit('plotly_animatingframe', {name: 'step-2'});
  const event = keyEvent('ArrowUp', f.plot);

  f.keydown(event);

  assert.deepEqual(plain(f.animateCalls[0].frames), ['step-1']);
  assert.deepEqual(plain(f.relayoutCalls[0]), {'sliders[0].active': 2});
});

test('Play steps refreshes the slider after Plotly updates its active value', () => {
  const f = fixture(false, {
    frameNames: ['step-0', 'step-1', 'step-2'], hasSlider: true
  });

  // Plotly can update the layout value before the external frame listener runs.
  // The visible slider still needs the relayout request for this frame.
  f.plot.layout.sliders[0].active = 1;
  f.emit('plotly_animatingframe', {name: 'step-1'});

  assert.deepEqual(plain(f.relayoutCalls), [{'sliders[0].active': 1}]);
});

test('Play steps refreshes after a jump whose frame event did not arrive', async () => {
  const f = fixture(false, {
    frameNames: ['step-0', 'step-1', 'step-2'], hasSlider: true
  });
  f.keydown(keyEvent('ArrowRight', f.plot));
  await Promise.resolve();
  await Promise.resolve();

  f.plot.layout.sliders[0].active = 0;
  f.emit('plotly_animatingframe', {name: 'step-1'});

  assert.deepEqual(plain(f.relayoutCalls), [
    {'sliders[0].active': 1},
    {'sliders[0].active': 1}
  ]);
});

test('click-to-jump updates the debugger arrow navigation state', () => {
  const f = fixture(false, {
    frameNames: ['step-0', 'step-1', 'step-2', 'step-3'], hasSlider: true
  });
  f.plot.data.push({meta: {bloqadeTraceKind: 'circuitGate'}});
  f.emit('plotly_click', {points: [{curveNumber: 1, customdata: [2]}]});
  const event = keyEvent('ArrowRight', f.plot);

  f.keydown(event);

  assert.deepEqual(
    f.animateCalls.map((call) => plain(call.frames)),
    [['step-2'], ['step-3']]
  );
});

test('pointer proximity previews a site even when another overlay captures hover', () => {
  const f = fixture(true);
  f.move({clientX: 10, clientY: 20});
  assert.equal(f.lanes().length, 1);
  f.move({clientX: 300, clientY: 300});
  assert.equal(f.lanes().length, 0);
});

test('only pinch (ctrl+wheel) reaches Plotly wheel zoom', () => {
  const f = fixture(false);
  // Plotly's wheel handler sits on a descendant drag layer. Only a capture
  // listener on the plot runs before it; a bubbling one would run too late.
  assert.deepEqual(f.domOptions.wheel, [true]);
  const wheel = (ctrlKey) => {
    const event = {ctrlKey, stopped: false,
      stopPropagation() { this.stopped = true; }};
    f.domEvents.wheel.forEach((handler) => handler(event));
    return event.stopped;
  };
  assert.equal(wheel(false), true);
  assert.equal(wheel(true), false);
});

for (const kind of ['empty site', 'occupied site', 'atom']) {
  test(`debugger previews lanes when Plotly hovers an ${kind}`, () => {
    const f = fixture(kind !== 'empty site');
    f.emit('plotly_hover', {points: [{
      curveNumber: kind === 'atom' ? 1 : 0,
      customdata: kind === 'atom' ? f.atomData : f.siteData,
      x: 0, y: 0
    }]});
    assert.equal(f.lanes().length, 1);
    assert.equal(f.lanes()[0].attributes.d, 'M10,20L110,20');
    assert.equal(f.lanes()[0].attributes['stroke-dasharray'], '9,6');
    f.leave();
    assert.equal(f.lanes().length, 0);
  });
}

// ── Bus multiselector panel ──
//
// The fixture above stubs the panel selector out, so `installBusMultiselectors`
// returns before building anything and the checkboxes, undo/redo and the
// update-menu label dispatch never run. This fixture lets it build, so those
// paths are exercised rather than assumed.

// Arguments the controller hands back come from the `vm` realm, so their
// prototypes are not this file's. `deepStrictEqual` compares prototypes and
// would reject them as "same structure but not reference-equal"; a JSON round
// trip rebuilds them here.
function plain(value) {
  return JSON.parse(JSON.stringify(value));
}

function panelFixture(previewMode) {
  class Element {
    constructor(tag) {
      this.tag = tag;
      this.style = {};
      this.attributes = {};
      this.children = [];
      this.listeners = {};
      this.isConnected = true;
      this.offsetWidth = 100;
      this.offsetHeight = 40;
    }
    setAttribute(name, value) { this.attributes[name] = value; }
    addEventListener(name, handler) {
      (this.listeners[name] ??= []).push(handler);
    }
    appendChild(child) { this.children.push(child); return child; }
    remove() { this.isConnected = false; }
    dispatch(name, event) {
      (this.listeners[name] || []).forEach((handler) => handler(event));
    }
    // Depth-first walk, so a test can find a control without a real DOM query.
    descendants() {
      return this.children.flatMap((child) => [child, ...child.descendants()]);
    }
  }

  const overlay = new Element('g');
  const events = {};
  const domEvents = {};
  const restyleCalls = [];
  const plot = new Element('div');
  plot.clientWidth = 600;
  plot.clientHeight = 400;
  plot.data = [
    {visible: 'legendonly'},
    {visible: true},
    {visible: 'legendonly'},
    {x: [0], y: [0], customdata: [[0, 0, 0, 0, 0]]}
  ];
  plot.layout = {
    font: {color: 'black'},
    paper_bgcolor: 'white',
    meta: {
      archVisualizerSiteTraceIndex: 3,
      archVisualizerSiteLanePreviewMode: previewMode,
      archVisualizerSiteLanePathRefs: {},
      archVisualizerSiteLanePaths: [],
      archVisualizerBusTraceIndices: [0, 1, 2],
      archVisualizerBusControls: [
        {traceIndex: 0, kind: 'site', busId: 0, label: 'zone 0 · site bus 0',
         color: 'hsl(0, 68%, 40%)'},
        {traceIndex: 1, kind: 'word', busId: 1, label: 'zone 0 · word bus 1',
         color: 'hsl(138, 68%, 40%)'},
        {traceIndex: 2, kind: 'zone', busId: 2, label: 'zone bus 2',
         color: 'hsl(275, 68%, 40%)'}
      ]
    }
  };
  plot._fullLayout = {
    xaxis: {_offset: 10, l2p: (x) => 100 * x},
    yaxis: {_offset: 20, l2p: (y) => 100 * y}
  };
  plot.querySelector = (selector) => {
    if (selector === '[data-arch-visualizer-path-overlays]') return overlay;
    // Null, unlike the fixture above: let the panel actually be built.
    return null;
  };
  plot.querySelectorAll = () => [];
  plot.on = (name, handler) => { (events[name] ??= []).push(handler); };
  plot.addEventListener = (name, handler) => {
    (domEvents[name] ??= []).push(handler);
  };
  plot.getBoundingClientRect = () => ({left: 0, top: 0, width: 600, height: 400});

  vm.runInNewContext(source, {
    document: {
      getElementById: () => plot,
      createElement: (tag) => new Element(tag),
      createElementNS: (ns, tag) => new Element(tag)
    },
    window: {
      setTimeout: (callback) => callback(),
      Plotly: {
        restyle: (_plot, update, indices) => {
          restyleCalls.push({update, indices});
          return Promise.resolve();
        },
        relayout: () => Promise.resolve()
      }
    }
  });

  const panel = plot.children.find(
    (child) => 'data-arch-visualizer-bus-selectors' in child.attributes
  );
  const nodes = panel === undefined ? [] : panel.descendants();
  return {
    plot, panel, restyleCalls,
    checkboxes: nodes.filter((node) => node.type === 'checkbox'),
    buttons: nodes.filter((node) => node.tag === 'button'),
    swatches: nodes.filter(
      (node) => 'data-arch-visualizer-bus-color' in node.attributes
    ),
    // Click an update-menu button the way the capture-phase handler sees it.
    clickMenuButton: (label) => domEvents.click.forEach((handler) => handler({
      target: {closest: (selector) =>
        selector === 'g.updatemenu-button' ? {textContent: label} : null}
    }))
  };
}

test('the panel lists every bus with its colour and current visibility', () => {
  const f = panelFixture('hover');

  assert.notEqual(f.panel, undefined, 'the panel should be built');
  assert.equal(f.checkboxes.length, 3, 'one checkbox per bus');
  // Initial state is read off the traces: only trace 1 starts visible.
  assert.deepEqual(f.checkboxes.map((box) => box.checked), [false, true, false]);
  assert.deepEqual(
    f.swatches.map((swatch) => swatch.attributes['data-arch-visualizer-bus-color']),
    ['hsl(0, 68%, 40%)', 'hsl(138, 68%, 40%)', 'hsl(275, 68%, 40%)']
  );
  // Undo/redo, then the three group headings.
  assert.equal(f.buttons.length, 2);
  assert.deepEqual(f.buttons.map((button) => button.textContent), ['Undo', 'Redo']);
  // Nothing to undo yet from the initial state.
  assert.equal(f.buttons[0].disabled, true);
  assert.equal(f.buttons[1].disabled, true);
});

test('toggling a checkbox restyles only that bus', () => {
  const f = panelFixture('hover');

  f.checkboxes[0].checked = true;
  f.checkboxes[0].dispatch('change');

  assert.deepEqual(plain(f.restyleCalls), [{update: {visible: true}, indices: [0]}]);
  // The change is now undoable.
  assert.equal(f.buttons[0].disabled, false);
});

test('undo and redo walk the visibility history', () => {
  const f = panelFixture('hover');
  const [undo, redo] = f.buttons;

  f.checkboxes[0].checked = true;
  f.checkboxes[0].dispatch('change');
  assert.equal(undo.disabled, false);
  assert.equal(redo.disabled, true);

  undo.dispatch('click');
  assert.deepEqual(f.checkboxes.map((box) => box.checked), [false, true, false]);
  assert.equal(undo.disabled, true, 'back at the initial state');
  assert.equal(redo.disabled, false);

  redo.dispatch('click');
  assert.deepEqual(f.checkboxes.map((box) => box.checked), [true, true, false]);
  assert.equal(redo.disabled, true);
});

test('the clear and show-all menu buttons drive every bus', () => {
  const f = panelFixture('hover');

  f.clickMenuButton('Show all buses');
  assert.deepEqual(f.checkboxes.map((box) => box.checked), [true, true, true]);
  assert.deepEqual(plain(f.restyleCalls.at(-1)), {
    update: {visible: [true, true, true]}, indices: [0, 1, 2]
  });

  f.clickMenuButton('Clear all buses');
  assert.deepEqual(f.checkboxes.map((box) => box.checked), [false, false, false]);
  assert.deepEqual(plain(f.restyleCalls.at(-1)), {
    update: {visible: ['legendonly', 'legendonly', 'legendonly']}, indices: [0, 1, 2]
  });

  // A label the controller does not know is ignored rather than mis-dispatched.
  const before = f.restyleCalls.length;
  f.clickMenuButton('Play steps');
  assert.equal(f.restyleCalls.length, before);
});
