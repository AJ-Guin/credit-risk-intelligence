// Same-origin by default (FastAPI serves this page). If hosting the UI elsewhere,
// set this to your API URL, e.g. "https://your-api.onrender.com"
const API_BASE = "";

const $ = (s) => document.querySelector(s);
const form = $("#form"), result = $("#result"), btn = $("#submit"), errBox = $("#error");
const f = (n) => form.elements[n];

const SAMPLES = {
  low: { person_age: 35, person_income: 85000, person_home_ownership: "MORTGAGE", person_emp_length: 10,
         loan_intent: "EDUCATION", loan_grade: "A", loan_amnt: 8000, loan_int_rate: 7.5,
         cb_person_default_on_file: "N", cb_person_cred_hist_length: 12 },
  high: { person_age: 22, person_income: 24000, person_home_ownership: "RENT", person_emp_length: 1,
          loan_intent: "MEDICAL", loan_grade: "F", loan_amnt: 15000, loan_int_rate: 17.8,
          cb_person_default_on_file: "Y", cb_person_cred_hist_length: 2 },
};

function updatePercent() {
  const inc = parseFloat(f("person_income").value), amt = parseFloat(f("loan_amnt").value);
  f("loan_percent_income").value = inc > 0 && amt >= 0 ? (amt / inc).toFixed(2) : "";
}
form.addEventListener("input", (e) => { e.target.classList.remove("bad"); updatePercent(); });

document.querySelectorAll("[data-sample]").forEach((b) =>
  b.addEventListener("click", () => {
    Object.entries(SAMPLES[b.dataset.sample]).forEach(([k, v]) => (f(k).value = v));
    updatePercent();
  })
);

function countUp(el, to, ms = 1200) {
  const t0 = performance.now();
  (function tick(t) {
    const p = Math.min((t - t0) / ms, 1), e = 1 - Math.pow(1 - p, 3);
    el.textContent = (to * e).toFixed(1);
    if (p < 1) requestAnimationFrame(tick);
  })(t0);
}

function showResult(d) {
  const prob = d.default_probability, thr = d.threshold, high = d.default_prediction === 1;
  $("#needle").style.transform = `rotate(${-90 + prob * 180}deg)`;
  $("#thr").setAttribute("transform", `rotate(${-90 + thr * 180} 110 110)`);
  countUp($("#pct"), prob * 100);
  result.dataset.state = high ? "high" : "low";
  $("#verdict").textContent = high ? "High risk" : "Low risk";
  $("#note").textContent = high
    ? "The default probability is above the decision threshold."
    : "The default probability is below the decision threshold.";
  $("#s-prob").textContent = (prob * 100).toFixed(2) + "%";
  $("#s-thr").textContent = (thr * 100).toFixed(1) + "%";
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errBox.hidden = true;
  let valid = true;
  form.querySelectorAll("[required]").forEach((i) => {
    const bad = i.value === "";
    i.classList.toggle("bad", bad);
    if (bad) valid = false;
  });
  if (!valid) { errBox.textContent = "Please fill in the highlighted fields."; errBox.hidden = false; return; }

  const payload = {};
  new FormData(form).forEach((v, k) => (payload[k] = v));
  ["person_age", "cb_person_cred_hist_length"].forEach((k) => (payload[k] = parseInt(payload[k], 10)));
  ["person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income"]
    .forEach((k) => (payload[k] = parseFloat(payload[k])));

  btn.disabled = true; btn.classList.add("loading");
  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`Server returned ${res.status}. Check the values and try again.`);
    showResult(await res.json());
  } catch (err) {
    errBox.textContent = err.message.includes("Failed to fetch")
      ? "Can't reach the server. It may be waking up, so wait a few seconds and try again."
      : err.message;
    errBox.hidden = false;
  } finally {
    btn.disabled = false; btn.classList.remove("loading");
  }
});
