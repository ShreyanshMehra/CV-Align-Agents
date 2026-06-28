"use strict";

const form = document.getElementById("screen-form");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const submitBtn = document.getElementById("submit-btn");

const SECTION_ORDER = ["skills", "experience", "projects", "education"];

/* Small DOM helper that sets text safely (no innerHTML for untrusted data). */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function showStatus(message, isError) {
  statusEl.hidden = false;
  statusEl.textContent = message;
  statusEl.classList.toggle("error", Boolean(isError));
}

function clearStatus() {
  statusEl.hidden = true;
  statusEl.textContent = "";
  statusEl.classList.remove("error");
}

function pct(value) {
  return Math.round((Number(value) || 0) * 100);
}

function renderBreakdown(breakdown) {
  const wrap = el("div", "breakdown");
  SECTION_ORDER.forEach((section) => {
    const value = breakdown && section in breakdown ? breakdown[section] : 0;
    const row = el("div", "bar-row");
    row.appendChild(el("span", "bar-label", section));

    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill");
    fill.style.width = pct(value) + "%";
    track.appendChild(fill);
    row.appendChild(track);

    row.appendChild(el("span", "bar-value", pct(value) + "%"));
    wrap.appendChild(row);
  });
  return wrap;
}

function renderList(title, items, emptyText) {
  const frag = document.createDocumentFragment();
  frag.appendChild(el("p", "section-title", title));
  if (items && items.length) {
    const ul = el("ul");
    items.forEach((item) => ul.appendChild(el("li", null, item)));
    frag.appendChild(ul);
  } else {
    frag.appendChild(el("p", "empty", emptyText));
  }
  return frag;
}

function renderCandidate(candidate, index) {
  const card = el("div", "candidate");

  const head = el("div", "candidate-head");
  const titleWrap = el("div");
  const name = el("h3", "candidate-name", candidate.candidate_name || "Candidate");
  const file = el("span", "candidate-file", "  " + (candidate.filename || ""));
  name.appendChild(file);
  titleWrap.appendChild(name);
  head.appendChild(titleWrap);

  const scoreWrap = el("div");
  scoreWrap.appendChild(el("span", "score", pct(candidate.score) + "%  "));
  const verdict = candidate.verdict || "weak_fit";
  scoreWrap.appendChild(el("span", "badge " + verdict, verdict.replace("_", " ")));
  head.appendChild(scoreWrap);

  card.appendChild(head);
  card.appendChild(renderBreakdown(candidate.breakdown));
  card.appendChild(renderList("Gaps vs the job", candidate.gaps, "No major gaps found."));
  card.appendChild(
    renderList("Suggestions", candidate.suggestions, "No suggestions.")
  );

  if (candidate.hygiene_score !== null && candidate.hygiene_score !== undefined) {
    const hygiene = el(
      "p",
      "hygiene-line",
      "Resume hygiene: " + pct(candidate.hygiene_score) + "%"
    );
    if (candidate.hygiene_issues && candidate.hygiene_issues.length) {
      hygiene.textContent += " — " + candidate.hygiene_issues.join("; ");
    }
    card.appendChild(hygiene);
  }

  return card;
}

function renderResults(data) {
  resultsEl.innerHTML = "";
  const candidates = data.candidates || [];

  const summary = el(
    "p",
    "results-summary",
    `${candidates.length} result(s) · mode: ${data.mode}` +
      (data.job_title ? ` · role: ${data.job_title}` : "")
  );
  resultsEl.appendChild(summary);

  candidates.forEach((c, i) => resultsEl.appendChild(renderCandidate(c, i)));
}

async function handleSubmit(event) {
  event.preventDefault();
  resultsEl.innerHTML = "";

  const files = document.getElementById("resumes").files;
  if (!files || files.length === 0) {
    showStatus("Please choose at least one resume PDF.", true);
    return;
  }

  const formData = new FormData(form);
  submitBtn.disabled = true;
  showStatus("Screening… the first request can take ~30s if the app was asleep.");

  try {
    const resp = await fetch("/screen", { method: "POST", body: formData });
    if (!resp.ok) {
      let detail = "Request failed (" + resp.status + ").";
      try {
        const err = await resp.json();
        if (err.detail) detail = typeof err.detail === "string"
          ? err.detail
          : JSON.stringify(err.detail);
      } catch (_) {
        /* ignore JSON parse errors */
      }
      showStatus(detail, true);
      return;
    }
    const data = await resp.json();
    clearStatus();
    renderResults(data);
  } catch (e) {
    showStatus("Network error: " + e.message, true);
  } finally {
    submitBtn.disabled = false;
  }
}

form.addEventListener("submit", handleSubmit);
