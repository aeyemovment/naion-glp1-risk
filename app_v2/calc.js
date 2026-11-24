// Client-side XGBoost inference for the NAION research calculator. No network calls; nothing is stored.
(function (root) {
  const M = root.NAION_MODEL;
  const S = M.spec;
  const logit = p => Math.log(p / (1 - p));
  const clamp = (v, [lo, hi]) => Math.min(hi, Math.max(lo, v));

  function features(therapy, indication, age, horizon) {
    const x = Object.fromEntries(S.columns.map(c => [c, 0]));
    const set = { therapy, indication, ...S.reference_setting };
    for (const [k, v] of Object.entries(set)) { if (`${k}_${v}` in x) x[`${k}_${v}`] = 1; }
    x.glp1ra_exposed = S.therapies[therapy];
    x.followup_years = horizon; x.mean_age = age; x.pct_female = S.pct_female_default;
    return S.columns.map(c => x[c]);
  }
  function predict(x) {
    let m = logit(M.base_score);
    for (const t of M.trees) {
      let n = 0;
      while (t.l[n] !== -1) {
        const v = x[t.f[n]];
        n = (Number.isNaN(v) ? t.d[n] : v < t.c[n]) ? t.l[n] : t.r[n];
      }
      m += t.c[n];
    }
    return 1 / (1 + Math.exp(-m));
  }
  function predictLR(x) {
    const L = M.lr; let z = L.intercept;
    for (let i = 0; i < x.length; i++) z += L.coef[i] * (x[i] - L.mu[i]) / L.sd[i];
    return 1 / (1 + Math.exp(-z));
  }
  // core = mean of XGBoost and logistic-regression probabilities (the paper's mean ensemble)
  function core(x) { const a = predict(x), b = predictLR(x); return { xgb: a, lr: b, ens: (a + b) / 2 }; }
  // optional literature modifiers: centred log-odds contributions; unknown inputs contribute 0
  function modifiers(values) {
    const out = [];
    for (const m of M.modifiers || []) {
      const v = values[m.id];
      if (v === undefined || v === "" || v === "unknown") continue;
      let c, lab;
      if (m.type === "binary") { const x = v === "yes" ? 1 : 0; c = Math.log(m.or) * (x - m.p_ref); lab = x ? (m.yes || "Yes") : (m.no || "No"); }
      else { const l = m.levels.find(l => l.v === v); c = Math.log(l.or) - m.center; lab = l.label; }
      out.push({ id: m.id, label: m.label, level: lab, logodds: c, mult: Math.exp(c), source: m.source });
    }
    return out;
  }
  const shift = (p, d) => 1 / (1 + Math.exp(-(logit(p) + d)));
  root.NAION = { features, predict, predictLR, core, modifiers, shift, clamp };

  if (typeof document === "undefined") return;
  const $ = id => document.getElementById(id);
  $("ok").addEventListener("change", e => { $("go").disabled = !e.target.checked; });
  const per100k = p => (p * 1e5).toFixed(1);
  $("go").addEventListener("click", () => {
    const warns = [];
    const ageIn = parseFloat($("age").value), fuIn = parseFloat($("fu").value);
    if (!(ageIn > 0) || !(fuIn > 0)) { $("warns").innerHTML = '<div class="warnlist">Enter an age and a time horizon.</div>'; $("out").hidden = false; return; }
    const age = clamp(ageIn, S.training_ranges.mean_age), fu = clamp(fuIn, S.training_ranges.followup_years);
    if (age !== ageIn) warns.push(`Age ${ageIn} is outside the training range; age ${age.toFixed(1)} was used.`);
    if (fu !== fuIn) warns.push(`Horizon ${fuIn} y is outside the training range; ${fu} y was used.`);
    const ind = $("ind").value;
    if (ind === "iih") warns.push("Only one small cohort studied IIH, so this estimate is especially uncertain.");
    const cd = core(features($("drug").value, ind, age, fu)), cc = core(features($("comp").value, ind, age, fu));
    const vals = {}; for (const m of M.modifiers || []) { const el = $("m_" + m.id); if (el) vals[m.id] = el.value; }
    const mods = modifiers(vals), d = mods.reduce((a, m) => a + m.logodds, 0);
    const pd = shift(cd.ens, d), pc = shift(cc.ens, d);
    const hi = cd.ens >= S.ensemble_top10_threshold;
    $("h").textContent = `${fu} year${fu === 1 ? "" : "s"}`;
    $("pd").textContent = per100k(pd); $("pc").textContent = per100k(pc);
    $("rr").textContent = (pd / pc).toFixed(2) + "×";
    $("tier").textContent = hi ? "Higher (top 10%)" : "Not in top 10%";
    $("tierd").textContent = `core model only; threshold ${per100k(S.ensemble_top10_threshold)} per 100,000`;
    const mx = Math.max(pd, pc, S.prevalence) * 1.15;
    [["bd", "nd", pd], ["bc", "nc", pc], ["bp", "np", S.prevalence]].forEach(([b, n, v]) => { $(b).style.width = (100 * v / mx) + "%"; $(n).textContent = per100k(v); });
    $("barsimg").setAttribute("aria-label", `Risk per 100,000: GLP-1RA ${per100k(pd)}, comparator ${per100k(pc)}, cohort average ${per100k(S.prevalence)}`);
    const rows = [`<tr><td>Core: XGBoost</td><td>${per100k(cd.xgb)}</td><td>${per100k(cc.xgb)}</td><td></td></tr>`,
      `<tr><td>Core: logistic regression</td><td>${per100k(cd.lr)}</td><td>${per100k(cc.lr)}</td><td></td></tr>`,
      `<tr><td><strong>Core ensemble (mean)</strong></td><td><strong>${per100k(cd.ens)}</strong></td><td><strong>${per100k(cc.ens)}</strong></td><td>validated</td></tr>`]
      .concat(mods.map(m => `<tr><td>+ ${m.label}: ${m.level}</td><td colspan="2">odds × ${m.mult.toFixed(2)}</td><td>published OR, not validated here</td></tr>`))
      .concat(mods.length ? [`<tr><td><strong>With optional factors</strong></td><td><strong>${per100k(pd)}</strong></td><td><strong>${per100k(pc)}</strong></td><td>exploratory</td></tr>`] : []);
    $("contrib").innerHTML = rows.join("");
    if (mods.length) warns.push(`${mods.length} optional factor${mods.length > 1 ? "s" : ""} applied using published odds ratios. These were not validated in this model and assume the factors act independently, which can overstate combined risk.`);
    if (vals.cdr && vals.cdr !== "unknown") warns.push("Cup-to-disc ratio odds ratios come from fellow eyes measured after NAION; no baseline (pre-treatment) estimate exists.");
    $("warns").innerHTML = warns.length ? `<div class="warnlist">${warns.map(w => `<div>${w}</div>`).join("")}</div>` : "";
    $("out").hidden = false; $("out").scrollIntoView({ behavior: "smooth", block: "start" });
  });
})(typeof window !== "undefined" ? window : globalThis);
