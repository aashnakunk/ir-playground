// Click-to-explain tooltip system.
// Any element with `data-explain="..."` will show its text on click/hover.
(function () {
  const tip = document.createElement("div");
  tip.id = "tooltip";
  document.body.appendChild(tip);

  function show(e) {
    const text = e.currentTarget.dataset.explain;
    if (!text) return;
    tip.textContent = text;
    tip.style.display = "block";
    const r = e.currentTarget.getBoundingClientRect();
    tip.style.left = Math.min(window.innerWidth - 340, r.left) + "px";
    tip.style.top = (r.bottom + 6) + "px";
  }
  function hide() { tip.style.display = "none"; }

  function bind(root) {
    root.querySelectorAll("[data-explain]").forEach((el) => {
      el.classList.add("explainable");
      el.addEventListener("mouseenter", show);
      el.addEventListener("mouseleave", hide);
      el.addEventListener("click", show);
    });
  }

  window.Explain = { bind };
  document.addEventListener("DOMContentLoaded", () => bind(document));
})();
