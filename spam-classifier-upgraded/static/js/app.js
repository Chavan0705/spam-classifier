const $ = (s, el = document) => el.querySelector(s);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const icons = () => window.lucide && lucide.createIcons();
const pct = (v) => (v * 100).toFixed(1) + "%";

async function api(url, options) {
  let res;
  try { res = await fetch(url, options); } catch { throw new Error("Cannot reach the server."); }
  let data = {};
  try { data = await res.json(); } catch { /* non-JSON body */ }
  if (!res.ok) throw new Error(data.error || "Request failed.");
  return data;
}

function toast(msg, type = "info") {
  const t = document.createElement("div");
  t.className = "toast " + type;
  t.textContent = msg;
  $("#toasts").append(t);
  setTimeout(() => t.remove(), 4000);
}

const charts = {};
function chart(id, type, data, options = {}) {
  const canvas = $("#" + id);
  const box = canvas.parentElement;
  box.classList.remove("is-empty");
  $(".empty", box).hidden = true;
  if (!window.Chart) return emptyChart(id, "Chart.js could not load. Check your internet connection.");
  if (charts[id]) charts[id].destroy();
  Chart.defaults.color = css("--muted");
  Chart.defaults.borderColor = css("--line");
  Chart.defaults.font.family = "Inter, system-ui, sans-serif";
  charts[id] = new Chart(canvas, { type, data, options: { responsive: true, maintainAspectRatio: false, ...options } });
}
function emptyChart(id, msg) {
  const box = $("#" + id).parentElement;
  box.classList.add("is-empty");
  const p = $(".empty", box);
  p.textContent = msg;
  p.hidden = false;
}

function confusionMatrix(el, m) {
  if (m.tp === undefined) { el.innerHTML = '<p class="empty">Retrain the models to record the confusion matrix.</p>'; return; }
  el.innerHTML = `<div class="cm"><span></span><b>Predicted legit</b><b>Predicted spam</b>
    <b>Actual legit</b><i class="ok">${m.tn}</i><i class="bad">${m.false_alarms}</i>
    <b>Actual spam</b><i class="bad">${m.missed_spam}</i><i class="ok">${m.tp}</i></div>`;
}

/* theme, mobile menu */
$("#theme").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("theme", next);
  if (window.rerender) rerender();
});
$("#menu").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
document.querySelector("main").addEventListener("click", () => $("#sidebar").classList.remove("open"));
icons();
