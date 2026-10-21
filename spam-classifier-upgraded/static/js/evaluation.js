let M = null;
const COL = { naive_bayes: "#38bdf8", lstm: "#a78bfa" };

function reportTable(r) {
  if (!r) return '<p class="empty">Retrain the models to record the classification report.</p>';
  const row = (n, x) => `<tr><td>${n}</td><td>${x.precision.toFixed(3)}</td><td>${x.recall.toFixed(3)}</td><td>${x.f1.toFixed(3)}</td><td>${x.support}</td></tr>`;
  return `<table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>${row("Legitimate", r.legitimate)}${row("Spam", r.spam)}</tbody></table>`;
}

function render() {
  const d = M.dataset;
  $("#dataset").textContent = `${d.name}: ${d.total.toLocaleString()} unique messages (${d.spam.toLocaleString()} spam, ${d.ham.toLocaleString()} legitimate, ${d.spam_pct}% spam). ` +
    `Stratified split: ${d.train.toLocaleString()} train, ${d.test.toLocaleString()} test (80/20). All metrics below are measured on the test set.`;

  $("#models").innerHTML = M.models.map((x) => {
    const m = x.metrics;
    return `<article class="card"><div class="row between"><h2>${x.name}</h2><span class="tag" style="--c:${COL[x.key]}">${x.key === M.primary ? "Best F1" : "Comparison"}</span></div>
      <p class="hint">${x.algorithm}</p>
      <table><tbody><tr><th>Accuracy</th><td>${pct(m.accuracy)}</td><th>Precision</th><td>${pct(m.precision)}</td></tr>
      <tr><th>Recall</th><td>${pct(m.recall)}</td><th>F1</th><td>${m.f1.toFixed(3)}</td></tr>
      <tr><th>Training time</th><td>${m.train_seconds} s</td><th>Parameters</th><td>${m.parameters.toLocaleString()}</td></tr></tbody></table>
      <h3>Classification report</h3>${reportTable(x.report)}
      <h3>Confusion matrix</h3><div id="cm-${x.key}"></div></article>`;
  }).join("");
  M.models.forEach((x) => confusionMatrix($("#cm-" + x.key), x.metrics));

  const keys = ["accuracy", "precision", "recall", "f1"];
  chart("c-cmp", "bar", { labels: ["Accuracy", "Precision", "Recall", "F1"], datasets: M.models.map((x) => ({ label: x.name, data: keys.map((k) => x.metrics[k]), backgroundColor: COL[x.key], borderRadius: 6 })) },
    { scales: { y: { min: 0, max: 1 } }, plugins: { legend: { position: "bottom" } } });

  const lc = M.learning_curve;
  if (!lc.length) emptyChart("c-curve", "Run python spam_classifier.py without --no-curve to record this.");
  else chart("c-curve", "line", { labels: lc.map((r) => r.n.toLocaleString()), datasets: [
    { label: "Naive Bayes", data: lc.map((r) => r.nb_f1), borderColor: COL.naive_bayes, tension: .25 },
    ...(lc.every((r) => "lstm_f1" in r) ? [{ label: "LSTM", data: lc.map((r) => r.lstm_f1), borderColor: COL.lstm, tension: .25 }] : [])] },
    { scales: { y: { min: 0, max: 1 }, x: { title: { display: true, text: "Training messages" } } }, plugins: { legend: { position: "bottom" } } });

  const h = (M.models.find((x) => x.key === "lstm") || { metrics: {} }).metrics.history;
  const curves = (id, a, b, la, lb) => {
    if (!h || !h[a]) return emptyChart(id, "No LSTM training history recorded.");
    chart(id, "line", { labels: h[a].map((_, i) => i + 1), datasets: [
      { label: la, data: h[a], borderColor: COL.lstm, tension: .25 }, { label: lb, data: h[b], borderColor: "#f59e0b", tension: .25 }] },
      { scales: { x: { title: { display: true, text: "Epoch" } } }, plugins: { legend: { position: "bottom" } } });
  };
  curves("c-loss", "loss", "val_loss", "Training loss", "Validation loss");
  curves("c-acc", "accuracy", "val_accuracy", "Training accuracy", "Validation accuracy");
}
window.rerender = () => M && render();

api("/api/model-metrics").then((m) => { M = m; render(); icons(); })
  .catch((e) => { toast(e.message, "error"); $("#dataset").textContent = e.message; });
