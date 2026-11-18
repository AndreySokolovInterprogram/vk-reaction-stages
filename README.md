# vk-reaction-stages

Code and aggregated data for the paper

> A. Sokolov, A. S. Shutkin, E. Isaeva, A. L. Beklaryan, A. A. Sokolov, S. V. Rusakov.
> **From likes to reposts and new users: identifying the stages at which content features act from aggregated counters.**
> Submitted to *IEEE Access*, 2026.

The paper proposes a multistage model of the reaction to a view (attention → consumption → evaluation → action), in which reposts, likes and comments share the early stages and differ in motive and evaluation. A content feature that acts on the shared stages shifts the shares of all three reactions equally; a feature that acts on the sharing motive shifts only the repost-to-like ratio. The repository contains the analysis of a stratified VK panel (310 communities, 166,523 posts, 10 thematic strata), the multimodal feature pipeline, the synthetic identifiability check, and scripts that redraw every figure from the aggregated data in `data/`.

## Repository structure

| Path | Contents |
|---|---|
| `analysis/` | Main analysis of the VK panel: pools, descriptive statistics, NB2 regressions, robustness checks (`run_all.sh`, scripts `10`–`40`) |
| `media_features/` | Feed features of posts: input table, image/first-line embeddings (Qwen3-VL-Embedding-2B), VLM labels of 16 concepts (Qwen3-VL-4B-Instruct, 4-bit), text rules, mixed models, invariance across strata, leave-one-stratum-out transfer, agreement with blinded expert labels |
| `synthetic/` | Synthetic identifiability check of the multistage model (Fisher information and maximum likelihood on simulated counters) |
| `figdata/` | Builds the aggregated tables for Figs. 5–6 from the panel export |
| `figures/` | One script per figure (`make_sokolN_*.py`), shared style (`common_style.py`), `make_all.sh`; the figures in the paper are the output of `figures/make_all.sh` |
| `collection/` | Measurement schedule of the panel (`schedule.py`, standalone excerpt of the collector) with a test/example (`test_schedule.py`) |
| `data/` | Aggregated, de-identified data needed to redraw the figures (see below) |
| `config.example.env` | Template for local paths to the (non-public) raw data |

## Installation

Python 3.12.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # analysis, synthetic check, figures
# media pipeline only (GPU with CUDA required), in a separate environment:
pip install -r requirements-media.txt
```

Paths to the raw data are not hard-coded. Copy `config.example.env` to `config.env` (ignored by git) and fill it in; the shell runners read it automatically. For Python scripts run directly, export the variables first: `set -a; . ./config.env; set +a`.

## How to run

**Figures from the aggregated data (no raw data needed):**

```bash
./figures/make_all.sh          # → figures/out/sokol1_system.pdf … sokol8_invariance.pdf (+ PNG, 600 dpi)
```

The figures in the paper (Figs. 1–8) are exactly the output of `figures/make_all.sh`.

**Synthetic identifiability check (no input data; ~1 min on a CPU):**

```bash
./synthetic/run_all.sh         # → synthetic/out/run1.txt … run6_real_alpha.txt
```

`run1`–`run5` cover scenarios S1–S6 (Fisher information at icc 0 / 0.002 / 0.01, features allowed in several stages, reposts only, missing negative-comment path, post-level heterogeneity, merged T and O, wrong link function). The numbers reported in Sec. V-E at the real overdispersion level — SEs of the feature effects 0.013 (motive) / 0.043 (attention) / 0.063 (consumption) / 0.094 (evaluation) and SEs of 0.29–0.35 for discriminating attention from consumption — are reproduced by `run6_real_alpha.py` (means over 20 simulation seeds; the script is a reconstruction, see its docstring).

**Full pipeline (requires the raw panel export, see "Data availability"):**

```bash
./analysis/run_all.sh                        # 1. main analysis → $ANALYSIS_OUT (pools.parquet, CSV tables, key_numbers.txt); ~2–30 min
python figdata/make_figdata.py               # 2. aggregated tables for Figs. 5–6 → data/
python media_features/build_media_items.py   # 3. media slice → $MEDIA_ROOT/data/media_items.parquet
./media_features/run_pipeline.sh             # 4. embeddings + VLM labels (GPU)
python media_features/humor_pass2.py         # 5. second pass for the exploratory humor feature (GPU)
python media_features/analysis_media.py      # 6. effects, invariance, transfer → $MEDIA_ROOT/out/analysis/
python media_features/make_validation.py     # 7. samples for blinded expert labelling (40 + 80 posts)
python media_features/make_validation2.py
python media_features/validation_agreement.py   # 8. precision / recall of VLM labels → $MEDIA_ROOT/out/validation_agreement/
./figures/make_all.sh                        # 9. figures
```

Step 6 reads the NB2 dispersion parameters produced by step 1; step 8 needs the expert labels (`VALIDATION_DIR/validation/manual_labels_*.csv`, `VALIDATION_DIR/validation2/manual_labels_*.csv`), which are not public because they are keyed by post.

## Correspondence to the paper

| Paper | Script(s) | Aggregated data |
|---|---|---|
| Fig. 1 — five-module system (module 4.4 highlighted) | `figures/make_sokol1_system.py` | — (schematic) |
| Fig. 2 — multistage model and stage signatures | `figures/make_sokol2_model.py` | — (schematic) |
| Fig. 3 — data and methods pipeline | `figures/make_sokol3_pipeline.py` | — (schematic) |
| Fig. 4 — precision / recall of VLM labels (95% Wilson CI) | `media_features/validation_agreement.py` → `figures/make_sokol4_audit.py` | `data/audit_agreement_random80.csv`, `data/audit_precision_all120.csv` |
| Fig. 5 — within-community scatter of log shares (R–L, C–L) | `analysis/30_pools.py`, `analysis/31_step3_descriptive.py` → `figdata/make_figdata.py` → `figures/make_sokol5_shares.py` | `data/figdata_shares_posts.csv`, `data/check_spearman.csv` |
| Fig. 6 — accumulation of counters over 72 h | `analysis/31_step3_descriptive.py` (step 3.3) → `figdata/make_figdata.py` → `figures/make_sokol6_accumulation.py` | `data/figdata_accumulation.csv` |
| Fig. 7 — feature effects on log ρ_R, log ρ_L, log(ρ_R/ρ_L) | `media_features/analysis_media.py` → `figures/make_sokol7_effects.py` | `data/effects_confirm_main_mixed.csv` |
| Fig. 8 — invariance across thematic strata | `media_features/analysis_media.py` → `figures/make_sokol8_invariance.py` | `data/effects_confirm_by_stratum.csv`, `data/effects_confirm_heterogeneity_tau.csv` |
| Table 2 — effects of confirmatory features (with τ) | `media_features/analysis_media.py` | `data/effects_confirm_main_mixed.csv`, `data/effects_confirm_heterogeneity_tau.csv` |
| Table 3 — leave-one-stratum-out transfer | `media_features/analysis_media.py` | `data/table3_confirm_transfer_summary.csv` |
| Sec. IV — panel description, pools, exclusions | `analysis/10_*`–`35_*`, `analysis/40_key_numbers.py` | — |
| Sec. V-A — NB2 regressions, overdispersion α, log-exposure coefficient | `analysis/32_step3_5_negbin.py` | — |
| Sec. V-E — synthetic identifiability | `synthetic/model.py`, `synthetic/run1.py`–`run6_real_alpha.py` | generated |
| Sec. IV — measurement schedule of the panel | `collection/schedule.py`, `collection/test_schedule.py` | — |

Section numbers follow the submitted manuscript. The code for the YouTube part (Sec. VI-F) will be added to this repository later.

## Data collection protocol (VK panel)

The panel was collected by the Markly data collector (not included: it contains production infrastructure code). Protocol:

- **Sample.** 310 open Russian-language VK communities with ≥ 1,000 members and at least one non-pinned post in the previous 7 days; 10 thematic strata (food & delivery, beauty, auto, finance & business, education & science, humor & entertainment, news & society, sport & fitness, tech & gadgets, viral aggregators) assigned by a regular expression on the community name and checked against the VK community topic; 4 size buckets by current member count (S 1–10k, M 10–100k, L 100k–1M, XL > 1M); 8 communities per stratum × size cell drawn in random order (seed 20261109); three XL cells have 4–5 communities because no more eligible communities exist.
- **Live posts.** Each community was polled hourly with `wall.get` (latest 30 posts, batched via `execute`). Each new post was then measured with `wall.getById` (batches of 100 via `execute`) at scheduled ages of 1, 3, 6, 12, 24, 48 and 72 h after publication; a measurement counts for a scheduled mark if taken within max(20 min, 10% of the mark); the actual age is always stored. Measured counters: views, likes, reposts, comments. Member counts were refreshed daily with `groups.getById`.
- **Retrospective posts.** All posts published in the 90 days before the panel start, plus video and clip posts back to 11 Nov 2025, with a single counter snapshot (final values are taken at age ≥ 7 days).
- **Schedule.** The planning rule (scheduled marks, tolerance, skipped marks) is reproduced in `collection/schedule.py`; `python collection/test_schedule.py` runs the tests and prints the plan of marks 1/3/6/12/24/48/72 h with tolerance max(20 min, 10%).
- **Period.** 8–14 Nov 2026 (final export 14 Nov 2026, 03:00 UTC).
- **Clip exposure.** For clips, exposure is the view count of the main video (`video.get`), collected separately.
- **Analysis exclusions** (`analysis/30_pools.py`): reposts, pinned, advertising and deleted posts; communities flagged by a pre-specified auto-posting rule; communities with missing view counts for ≥ 50% of posts; posts with reactions exceeding exposure; exposure ≥ 100 for share-based analyses.

The input tables expected by `analysis/` are Parquet files `communities`, `posts`, `snapshots`, `community_snapshots`, `media` and `manifest.json` (schema in `analysis/common.py` and the scripts).

## Data availability

Raw data are **not** publicly available: they consist of VK API exports (posts, texts, media, community identifiers) whose redistribution is restricted by the VK API terms, and they may contain personal data of users. The `data/` folder contains only aggregated, de-identified tables sufficient to reproduce all data figures:

| File | Columns | Content |
|---|---|---|
| `figdata_shares_posts.csv` | `modality`, `cm_log_rho_R`, `cm_log_rho_L`, `cm_log_rho_C` | one row per post of the share pool (127,972); log shares centered by the community × block median; no community, stratum or post identifiers; rows shuffled |
| `check_spearman.csv` | `subset`, `pair`, `n`, `spearman_*` | Spearman correlations of the centered log shares |
| `figdata_accumulation.csv` | `modality`, `counter`, `age_h`, `rule`, `n_posts`, `n_posts_72h_pos`, `median_share`, `q25`, `q75`, `mean_share`, `share_of_sums` | accumulation of counters of live posts relative to 72 h |
| `audit_agreement_random80.csv` | per VLM concept: TP, FP, FN, TN, precision, recall (Wilson CI), κ | agreement with blinded expert labels, random posts |
| `audit_precision_all120.csv` | per VLM concept: precision on all VLM-positive posts | including enriched samples |
| `effects_confirm_main_mixed.csv` | feature, outcome, n, coef, se, p, CI, BH q | mixed-model effects of confirmatory features |
| `effects_confirm_by_stratum.csv` | feature, outcome, stratum, coef, se, p, CI | per-stratum effects |
| `effects_confirm_heterogeneity_tau.csv` | feature, outcome, β (fixed / random effects), τ², τ, Q, I², p | between-stratum heterogeneity (DerSimonian–Laird) |
| `table3_confirm_transfer_summary.csv` | feature, outcome, transfer gain, β range across held-out strata | leave-one-stratum-out transfer |

Synthetic data are generated by the scripts in `synthetic/`.

## Notes on reproducibility

- The main analysis is deterministic: on the final export, `analysis/run_all.sh` reproduces `key_numbers.txt` and all CSV tables of the reported run with identical numbers.
- `synthetic/run6_real_alpha.py` is a reconstruction of the calculation at the real overdispersion level (the original script was not preserved).
- VLM labels depend on the GPU, driver and library versions; model revisions are pinned in `media_features/features.py`.

## Citation

See `CITATION.cff`. If you use this code, please cite the paper.

## License

MIT, see `LICENSE`.
