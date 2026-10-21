let D = null;
const COLORS = { spam: "#f43f5e", ham: "#10b981", nb: "#38bdf8", lstm: "#a78bfa" };
const modelColor = (k) => (k === "lstm" ? COLORS.lstm : COLORS.nb);

function kpis() {
  const d = D.dataset, p = D.models.find((m) => m.key === D.primary), m = p.metrics;
  const items = [
    ["Messages analyzed", d.total.toLocaleString(), "mail", "g1"],
    ["Spam detected", d.spam.toLocaleString(), "shield-alert", "g2"],
    ["Legitimate", d.ham.toLocaleString(), "shield-check", "g3"],
    ["Spam rate", d.spam_pct + "%", "percent", "g4"],
    [`Accuracy (${p.name})`, pct(m.accuracy), "target", "g1"],
    ["Precision", pct(m.precision), "crosshair", "g3"],
    ["Recall", pct(m.recall), "radar", "g5"],
    ["F1-score", m.f1.toFixed(3), "award", "g2"],
  ];
  $("#kpis").innerHTML = items.map(([l, v, i, g]) =>
    `<div class="kpi ${g}"><i data-lucide="${i}"></i><small>${l}</small><b>${v}</b></div>`).join("");
  icons();
}

function render() {
  const d = D.dataset, p = D.models.find((m) => m.key === D.primary), m = p.metrics;
  kpis();
  chart("c-split", "doughnut", { labels: ["Legitimate", "Spam"], datasets: [{ data: [d.ham, d.spam], backgroundColor: [COLORS.ham, COLORS.spam], borderWidth: 0 }] },
    { cutout: "65%", plugins: { legend: { position: "bottom" } } });

  $("#outcome-hint").textContent = `${p.name}, ${d.test.toLocaleString()} test messages`;
  if (m.tp === undefined) emptyChart("c-outcomes", "Retrain the models to record test outcomes.");
  else chart("c-outcomes", "bar", { labels: ["True legit", "False alarms", "Missed spam", "True spam"],
    datasets: [{ data: [m.tn, m.false_alarms, m.missed_spam, m.tp], backgroundColor: [COLORS.ham, "#f59e0b", "#f59e0b", COLORS.spam], borderRadius: 6 }] },
    { plugins: { legend: { display: false } } });

  const h = D.history;
  if (!h.length) emptyChart("c-history", "No predictions yet. Analyze a message on the Detect spam page.");
  else chart("c-history", "line", { labels: h.map((x) => new Date(x.time).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })),
    datasets: [{ label: "Spam probability", data: h.map((x) => x.spam_probability), borderColor: COLORS.lstm, backgroundColor: "rgb(167 139 250 / .2)", fill: true, tension: .3, pointRadius: 4 }] },
    { scales: { y: { min: 0, max: 1 } } });

  $("#cm-hint").textContent = p.name + ", test set";
  confusionMatrix($("#cm"), m);

  const keys = ["accuracy", "precision", "recall", "f1"], names = ["Accuracy", "Precision", "Recall", "F1"];
  chart("c-models", "bar", { labels: names, datasets: D.models.map((x) => ({ label: x.name, data: keys.map((k) => x.metrics[k]), backgroundColor: modelColor(x.key), borderRadius: 6 })) },
    { scales: { y: { min: 0, max: 1 } }, plugins: { legend: { position: "bottom" } } });

  if (!m.prob_hist) emptyChart("c-probs", "Retrain the models to record probability scores.");
  else chart("c-probs", "bar", { labels: m.prob_hist.ham.map((_, i) => `${i * 10}-${(i + 1) * 10}%`),
    datasets: [{ label: "Actually legitimate", data: m.prob_hist.ham, backgroundColor: COLORS.ham }, { label: "Actually spam", data: m.prob_hist.spam, backgroundColor: COLORS.spam }] },
    { scales: { x: { stacked: true, title: { display: true, text: "Predicted spam probability" } }, y: { stacked: true, title: { display: true, text: "Messages" } } }, plugins: { legend: { position: "bottom" } } });

  const w = D.top_words.spam.slice(0, 10);
  chart("c-words", "bar", { labels: w.map((x) => x.word), datasets: [{ data: w.map((x) => x.weight), backgroundColor: COLORS.spam, borderRadius: 4 }] },
    { indexAxis: "y", plugins: { legend: { display: false } } });
}
window.rerender = () => D && render();

api("/api/dashboard").then((d) => { D = d; render(); })
  .catch((e) => { toast(e.message, "error"); $("#kpis").innerHTML = `<p class="empty">${e.message}</p>`; });
