// Loads the active theme from /api/theme once and exposes it as
// `window.THEME_READY` (a Promise<theme>) for pages to read defaults from.
// Also rewrites the page title with the theme name.
window.THEME_READY = fetch("/api/theme")
  .then((r) => r.json())
  .then((t) => {
    window.THEME = t;
    document.title = document.title.replace(/jazzbot/i, `${t.name} IR playground`);
    document.querySelectorAll("[data-theme-name]").forEach((el) => {
      el.textContent = t.name;
    });
    return t;
  })
  .catch(() => ({ corpus: "jazz", name: "jazz", default_queries: {} }));
