const EXAMPLES = [
  "WINNER!! You have been selected to receive a £900 prize reward. To claim call 09061701461. Claim code KL341.",
  "Hey, are we still meeting at 6? I'll grab the tickets on my way back from college.",
  "Free entry? No thanks. I'm just asking if you're free to call me tonight.",
];
let exIdx = 0, last = null;
const msg = $("#msg");
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const word = (t) => (t === "numtoken" ? "‹number›" : t === "urltoken" ? "‹link›" : t);

msg.addEventListener("input", () => { $("#count").textContent = `${msg.value.length} / 1000`; });
$("#example").addEventListener("click", () => { msg.value = EXAMPLES[exIdx++ % EXAMPLES.length]; msg.dispatchEvent(new Event("input")); });
$("#clear").addEventListener("click", () => { msg.value = ""; msg.dispatchEvent(new Event("input")); $("#result").hidden = true; $("#result-empty").hidden = false; msg.focus(); });
msg.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyze(); });
$("#analyze").addEventListener("click", analyze);

const bar = (name, p) => `<div class="bar-row"><div><span>${name}</span><b>${pct(p)} spam</b></div>
  <div class="bar"><i style="width:${(p * 100).toFixed(1)}%;--w:${(100 / Math.max(p, .01)).toFixed(0)}%"></i></div></div>`;

async function analyze() {
  const text = msg.value.trim();
  if (!text) { toast("Enter a message to analyze.", "error"); msg.focus(); return; }
  const btn = $("#analyze");
  btn.disabled = true;
  btn.innerHTML = '<i data-lucide="loader-circle" class="spin"></i><span>Analyzing…</span>';
  icons();
  try {
    const r = await api("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
    last = r;
    const el = $("#label");
    el.textContent = r.label;
    el.className = "label " + (r.label === "SPAM" ? "spam" : "ham");
    $("#by").textContent = `Verdict from ${r.primary === "lstm" ? "the LSTM" : "Naive Bayes"} (the model with the higher test F1).`;
    $("#bars").innerHTML = bar("Naive Bayes", r.naive_bayes.probability) + (r.lstm ? bar("LSTM", r.lstm.probability) : "");
    const nb = r.naive_bayes, seen = new Set();
    $("#drivers").innerHTML = r.tokens.map((t, i) => ({ t, s: nb.scores[i] }))
      .filter((x) => Math.abs(x.s) > 1e-6 && !seen.has(x.t) && seen.add(x.t))
      .sort((a, b) => Math.abs(b.s) - Math.abs(a.s)).slice(0, 8)
      .map((x) => `<span class="chip ${x.s > 0 ? "s" : "h"}">${esc(word(x.t))} ${x.s > 0 ? "+" : ""}${x.s.toFixed(2)}</span>`).join("") || "None";
    $("#result").hidden = false;
    $("#result-empty").hidden = true;
    loadRecent();
  } catch (e) { toast(e.message, "error"); }
  btn.disabled = false;
  btn.innerHTML = '<i data-lucide="scan-search"></i><span>Analyze Message</span>';
  icons();
}

$("#copy").addEventListener("click", async () => {
  if (!last) return;
  const t = `${last.label} (${pct(last.spam_probability)} spam probability): ${msg.value.trim().slice(0, 120)}`;
  try { await navigator.clipboard.writeText(t); toast("Result copied.", "ok"); } catch { toast("Copy is not available in this browser.", "error"); }
});

async function loadRecent() {
  try {
    const rows = await api("/api/history");
    if (rows.length) $("#recent").innerHTML = '<div class="recent">' + rows.map((r) =>
      `<div><span class="badge ${r.label === "SPAM" ? "spam" : "ham"}">${r.label}</span><span title="${esc(r.text)}">${esc(r.text)}</span><small>${new Date(r.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</small></div>`).join("") + "</div>";
  } catch { /* leave empty state */ }
}
loadRecent();
