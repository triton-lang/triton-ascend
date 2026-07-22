// Render Mermaid diagrams in the ReadTheDocs (sphinx_rtd_theme) output.
// GitHub renders ```mermaid blocks natively, but the RTD theme needs a
// client-side renderer.  This script loads mermaid from a CDN and converts
// every <pre><code class="language-mermaid"> block into an SVG diagram.
(function () {
  'use strict';

  // Already initialised guard
  if (window.__mermaidRtdLoaded) return;
  window.__mermaidRtdLoaded = true;

  var script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js';
  script.onload = function () {
    mermaid.initialize({ startOnLoad: false, theme: 'default' });

    var blocks = document.querySelectorAll('pre code.language-mermaid');
    if (!blocks.length) return;

    // Replace each <pre><code> block with a <div class="mermaid"> so
    // mermaid.run() can render it in-place.
    for (var i = 0; i < blocks.length; i++) {
      var code = blocks[i];
      var pre = code.parentElement;
      var source = code.textContent;
      var div = document.createElement('div');
      div.className = 'mermaid';
      div.textContent = source;
      pre.parentNode.replaceChild(div, pre);
    }

    mermaid.run();
  };
  document.head.appendChild(script);
})();
