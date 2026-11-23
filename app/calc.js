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
  root.NAION = { features, predict, clamp };

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
    const pd = predict(features($("drug").value, ind, age, fu));
    const pc = predict(features($("comp").value, ind, age, fu));
    const hi = pd >= S.top10_threshold;
    $("h").textContent = `${fu} year${fu === 1 ? "" : "s"}`;
    $("pd").textContent = per100k(pd); $("pc").textContent = per100k(pc);
    $("rr").textContent = (pd / pc).toFixed(2) + "×";
    $("tier").textContent = hi ? "Higher (top 10%)" : "Not in top 10%";
    $("tierd").textContent = `threshold ${per100k(S.top10_threshold)} per 100,000`;
    const mx = Math.max(pd, pc, S.prevalence) * 1.15;
    [["bd", "nd", pd], ["bc", "nc", pc], ["bp", "np", S.prevalence]].forEach(([b, n, v]) => { $(b).style.width = (100 * v / mx) + "%"; $(n).textContent = per100k(v); });
    $("barsimg").setAttribute("aria-label", `Risk per 100,000: GLP-1RA ${per100k(pd)}, comparator ${per100k(pc)}, cohort average ${per100k(S.prevalence)}`);
    $("warns").innerHTML = warns.length ? `<div class="warnlist">${warns.map(w => `<div>${w}</div>`).join("")}</div>` : "";
    $("out").hidden = false; $("out").scrollIntoView({ behavior: "smooth", block: "start" });
  });
})(typeof window !== "undefined" ? window : globalThis);
