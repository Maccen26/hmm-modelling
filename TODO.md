# To Do

Thesis hand-in: 7 December 2026. References are handled separately.

## 1. Answer the research question (content is missing)

- [ ] Write Chapter 6 Simulation (`Chapters/06_Simulation.tex` only contains "A"); use `drivers/simulation/sample_size`
- [ ] Write Chapter 7 Discussion (empty)
- [ ] Write Chapter 8 Conclusion (empty)
- [ ] Write Chapter 9 Future work (empty)
- [ ] Write text for Ch. 5 State Characterization: interpret the states (empty / arriving / occupied / airing), State dynamics, Feedback by state
- [ ] Write the Ch. 5 Summary
- [ ] Introduction: add explicit research questions + written chapter outline, replace the "X%" placeholders
- [ ] Abstract (write last, so it matches what was delivered)
- [ ] If the simulation can't be finished: narrow the title ("Estimating Thermal Comfort Curves") and research questions
- [ ] Fix template leftovers:
  - [ ] `Setup/Statics.tex` says "Bachelor Thesis"
  - [ ] `Frontmatter/Approval.tex` says "six months… Department of Civil Engineering… MSc Eng"
  - [ ] `Frontmatter/Acknowledgements.tex` still has the template credit and `[Name]` placeholders

Suggested order: state interpretation → Simulation → Discussion/Conclusion → Introduction research questions → Abstract.

## 2. Add a methodology / experimental setup section

Ch. 5 says "As described in the experimental setup…" but that section doesn't exist (Ch. 4 only covers libraries + class design). Add 1–2 pages covering:

- [ ] Train / validation / test split (use the unused `Plots/data/room_015_split.tex`)
- [ ] Definition of the covariates `x_t` (off day, working hours) and how they're built
- [ ] Fitting procedure: seeded + random starts, rejecting collapsed states, keeping the best validation RMSE
- [ ] The ridge penalty (only mentioned in the model comparison table caption)
- [ ] h-step-ahead forecasting + definitions of RMSE and MAPE
- [ ] Standard errors: Hessian + delta method → Wald CIs (fills "MISSING REFERENCE TO THEORY" in Ch. 5)

## 3. Fix inconsistencies and wrong statements

### Data chapter
- [ ] `eq:calibration` uses $Q_{0.02}$ but the text says 1.5% / $Q_{0.015}$
- [ ] Descriptive analysis and Summary say "calibrated as in `eq:calibration`", but the models use the causal anchor — commit to the anchor method
- [ ] Contradiction: missing bins "linearly interpolated" vs. "forward-filled instead of interpolated"

### Theory chapter
- [ ] Backward sampling: delete the old (wrong) derivation, keep the "Corrected derivation"; remove the claim that it doesn't depend on $\pi$
- [ ] General HMM section uses a discrete emission matrix $B$, but the observations are continuous densities
- [ ] Notation: $S_t$ vs $X_t$, $u_t$ vs $u_{t|t-1}$, $\beta$ used for both production rate and regression coefficients
- [ ] AR(1)-HMM($\beta$,1): "as in" is missing its reference

### Results chapter
- [ ] Wrong claim: "Working day covariates do not have a significant difference" — e.g. $S_3\to S_3$ is 0.991 [0.977, 1] at base vs 0.506 [0.369, 0.642] in working hours
- [ ] States 2 and 3 have identical $\mu$ = 598.96: the ordering constraint $\mu_2<\mu_3$ is binding (boundary), so the Wald CI is invalid — say so
- [ ] State 4 $\mu$ ≈ 4619 ppm vs observed mean 1822 ppm: interpret $\mu$ as the OU equilibrium $P/\alpha$; report $\alpha = -\ln\phi/\Delta t$ (≈ 0.09–0.12 h⁻¹) and production rate per state
- [ ] Model selection: state the criterion up front and apply it consistently
  - [ ] K=3 beats K=4 on test RMSE at every horizon, but K=4 is chosen on BIC
  - [ ] K=4 is the BIC minimum *within* AR(1)-HMM($\beta$) (32185), but the lowest BIC across all models is AR(1)-HMM K=5 (32106) → make explicit that the model family is chosen on predictive RMSE + LRT, and BIC only picks K within that family
  - [ ] AR(1)-HMM K=6 has a lower LL than K=5 → optimiser didn't find the maximum; rerun or comment
- [ ] Discuss the working-parameter tables or move them to the appendix

### Small fixes
- [ ] "For state 6" → "For K=6" (and similar "states" wording)
- [ ] `Pictures/model_hierarchy.tex` is never used — include it or delete it
