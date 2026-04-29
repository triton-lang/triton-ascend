// Sidebar switcher: lets users hop between languages and versions.
//
// URL shapes this script understands:
//   /<slug>/<lang>/<rest>                       (production)
//   /pr-preview/pr-<N>/<slug>/<lang>/<rest>     (PR preview)
//
// versions.json is fetched once relative to the deployment root so we can
// populate the version dropdown. If it's missing we fall back to a single
// version derived from the current URL (and only the language switcher works).
(function () {
  function init() {
    var loc = parseLocation(window.location.pathname);
    if (!loc) return; // unknown layout; bail quietly.

    var versionsUrl = loc.deployRoot + 'versions.json';
    fetchJSON(versionsUrl).then(function (data) {
      var versions = (data && Array.isArray(data.versions))
        ? data.versions.map(function (v) { return v.slug; })
        : [loc.slug];
      mount(loc, versions);
    });
  }

  // Returns: { deployRoot, slug, lang, rest } or null.
  // deployRoot is "/" + optional pr-preview prefix, e.g. "/" or "/pr-preview/pr-42/".
  function parseLocation(path) {
    var m;

    m = path.match(/^(\/pr-preview\/pr-\d+\/)([^/]+)\/(zh|en)\/(.*)$/);
    if (m) {
      return { deployRoot: m[1], slug: m[2], lang: m[3], rest: m[4] };
    }

    m = path.match(/^\/([^/]+)\/(zh|en)\/(.*)$/);
    if (m) {
      // Defensive: skip if first segment is the umbrella.
      if (m[1] === 'pr-preview') return null;
      return { deployRoot: '/', slug: m[1], lang: m[2], rest: m[3] };
    }

    return null;
  }

  function fetchJSON(url) {
    return fetch(url, { cache: 'no-cache' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; });
  }

  // HEAD-probe a candidate URL; if it 404s, fall back to the version's index.
  function resolveTarget(deployRoot, slug, lang, rest) {
    var candidate = deployRoot + slug + '/' + lang + '/' + rest;
    var fallback = deployRoot + slug + '/' + lang + '/index.html';
    return fetch(candidate, { method: 'HEAD' })
      .then(function (r) { return r.ok ? candidate : fallback; })
      .catch(function () { return fallback; });
  }

  function mount(loc, versions) {
    var existing = document.getElementById('lang-version-switcher');
    if (existing) existing.remove();

    var wrap = document.createElement('div');
    wrap.id = 'lang-version-switcher';
    wrap.style.cssText = [
      'padding:.6rem 1rem',
      'border-bottom:1px solid rgba(255,255,255,.2)',
      'font-size:.85rem',
      'background:#2980b9',
      'color:#fff',
      'display:flex',
      'flex-direction:column',
      'gap:.4rem',
    ].join(';');

    // ---- Language row ------------------------------------------------------
    var langRow = document.createElement('div');
    langRow.style.cssText = 'display:flex;justify-content:center;gap:.5rem;';
    var otherLang = loc.lang === 'zh' ? 'en' : 'zh';

    function renderLangRow(otherUrl) {
      langRow.innerHTML = '🌐 ' +
        (loc.lang === 'zh'
          ? '<span style="font-weight:600;">中文</span> | <a style="color:#fff;text-decoration:underline;" href="' + otherUrl + '">English</a>'
          : '<a style="color:#fff;text-decoration:underline;" href="' + otherUrl + '">中文</a> | <span style="font-weight:600;">English</span>');
    }
    renderLangRow('#');
    resolveTarget(loc.deployRoot, loc.slug, otherLang, loc.rest).then(renderLangRow);

    // ---- Version row -------------------------------------------------------
    var verRow = document.createElement('div');
    verRow.style.cssText = 'display:flex;align-items:center;justify-content:center;gap:.4rem;';

    var label = document.createElement('span');
    label.textContent = '📦';
    verRow.appendChild(label);

    var select = document.createElement('select');
    select.style.cssText = 'background:#fff;color:#2980b9;border:none;border-radius:3px;padding:.15rem .35rem;font:inherit;cursor:pointer;';
    versions.forEach(function (slug) {
      var opt = document.createElement('option');
      opt.value = slug;
      opt.textContent = slug;
      if (slug === loc.slug) opt.selected = true;
      select.appendChild(opt);
    });
    select.addEventListener('change', function () {
      var newSlug = select.value;
      if (newSlug === loc.slug) return;
      // Try same page in the new version, fall back to its index.
      resolveTarget(loc.deployRoot, newSlug, loc.lang, loc.rest)
        .then(function (url) { window.location.href = url; });
    });
    verRow.appendChild(select);

    wrap.appendChild(langRow);
    wrap.appendChild(verRow);

    var sidebar = document.querySelector('.wy-side-nav-search');
    if (sidebar && sidebar.parentNode) {
      sidebar.parentNode.insertBefore(wrap, sidebar);
    } else {
      document.body.insertBefore(wrap, document.body.firstChild);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
