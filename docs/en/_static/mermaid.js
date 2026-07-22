// Render Mermaid diagrams in the ReadTheDocs (sphinx_rtd_theme) output.
// GitHub renders ```mermaid blocks natively, but Sphinx/MyST wraps them in
// Pygments highlighting markup.  This script loads mermaid from a CDN and
// converts every Pygments-highlighted mermaid block into an SVG diagram.
(function () {
  'use strict';

  // Already initialised guard
  if (window.__mermaidRtdLoaded) return;
  window.__mermaidRtdLoaded = true;

  const script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js';
  script.onload = function () {
    mermaid.initialize({ startOnLoad: false, theme: 'default' });

    // Sphinx + Pygments wraps code blocks as:
    //   <div class="highlight-mermaid notranslate">
    //     <div class="highlight"><pre><span></span>...mermaid src...</pre></div>
    //   </div>
    // Note: TextLexer output has no <code> wrapper — the source is directly inside <pre>.
    const blocks = document.querySelectorAll('div.highlight-mermaid');
    if (!blocks.length) return;

    for (let i = 0; i < blocks.length; i++) {
      const wrapper = blocks[i];
      const pre = wrapper.querySelector('pre');
      if (!pre) continue;
      // textContent skips the empty <span></span> that Pygments inserts
      const source = pre.textContent;
      const div = document.createElement('div');
      div.className = 'mermaid';
      div.textContent = source;
      wrapper.parentNode.replaceChild(div, wrapper);
    }

    mermaid.run();
  };
  document.head.appendChild(script);
})();
