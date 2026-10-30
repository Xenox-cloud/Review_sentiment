const form = document.getElementById("form");
const textBox = document.getElementById("text");
const resultBox = document.getElementById("result");
const errorBox = document.getElementById("error");
const historyList = document.getElementById("history");
const summaryBox = document.getElementById("summary");

function show(el, html) {
  el.innerHTML = html;
  el.classList.remove("hidden");
}

// Escape user text before putting it in the page.
function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

async function loadHistory() {
  const [reviews, summary] = await Promise.all([
    fetch("/api/reviews?limit=10").then((r) => r.json()),
    fetch("/api/summary").then((r) => r.json()),
  ]);
  summaryBox.innerHTML = ["total", "positive", "negative", "neutral"]
    .map((k) => `<div class="stat ${k}"><b>${summary[k]}</b><span>${k}</span></div>`)
    .join("");
  historyList.innerHTML = reviews
    .map((r) => `<li><span class="badge ${r.label}">${r.label}</span> ${escapeHtml(r.text)}</li>`)
    .join("") || "<li class='empty'>Nothing analysed yet.</li>";
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.classList.add("hidden");
  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: textBox.value }),
  });
  const data = await res.json();
  if (!res.ok) {
    // FastAPI validation errors come back as a list in "detail".
    const msg = Array.isArray(data.detail) ? data.detail[0].msg : data.detail;
    show(errorBox, escapeHtml(msg || "Something went wrong"));
    return;
  }
  const pct = Math.round(data.confidence * 100);
  show(resultBox, `<span class="badge ${data.label}">${data.label}</span> ${pct}% confidence
    <div class="bar"><div class="${data.label}" style="width:${pct}%"></div></div>`);
  textBox.value = "";
  loadHistory();
});

loadHistory().catch(() => show(errorBox, "Could not reach the server."));
