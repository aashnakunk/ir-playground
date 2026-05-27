// Chat sidebar widget. Each page provides:
//   window.JazzbotView = {
//     page: "bm25",
//     state: () => ({ ...whatever is currently on screen... })
//   };
// We POST { message, view_state: state(), history } to /api/chat.
(function () {
  const root = document.getElementById("chat");
  if (!root) return;

  root.innerHTML = `
    <header>
      <h3>tutor</h3>
      <button class="ghost" id="chat-clear" type="button">clear</button>
    </header>
    <div class="log" id="chat-log">
      <div class="msg assistant">
        <div class="who">tutor</div>
        Ask me anything about what's on your screen. I can see your current query, results, and parameters.
      </div>
    </div>
    <form class="composer" id="chat-form">
      <textarea id="chat-input" placeholder="why did doc #7 win?" rows="2"></textarea>
      <button type="submit">send</button>
    </form>
  `;

  const log = root.querySelector("#chat-log");
  const form = root.querySelector("#chat-form");
  const input = root.querySelector("#chat-input");
  const clear = root.querySelector("#chat-clear");
  const history = [];

  function append(role, text) {
    const el = document.createElement("div");
    el.className = "msg " + role;
    el.innerHTML = `<div class="who">${role === "user" ? "you" : "tutor"}</div>`;
    const body = document.createElement("div");
    body.textContent = text;
    el.appendChild(body);
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
    return body;
  }

  clear.addEventListener("click", () => {
    history.length = 0;
    log.innerHTML = "";
    append("assistant", "cleared. ask me anything about what's on screen.");
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = input.value.trim();
    if (!message) return;
    input.value = "";
    append("user", message);
    history.push({ role: "user", content: message });
    const placeholder = append("assistant", "thinking...");

    const view_state = (window.JazzbotView && window.JazzbotView.state)
      ? window.JazzbotView.state()
      : { page: "unknown" };

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, view_state, history: history.slice(0, -1) })
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      placeholder.textContent = data.reply;
      history.push({ role: "assistant", content: data.reply });
    } catch (err) {
      placeholder.textContent = "error: " + err.message;
    }
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
})();
