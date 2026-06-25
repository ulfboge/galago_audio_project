# Next steps — 2026-06-24

State assessed after ~3 months of inactivity since the March 2026 session.

---

## 1. Rotate the exposed HF token (do first — security)

A Hugging Face token was exposed in chat during the March 2026 deploy debugging.

- Revoke it at https://huggingface.co/settings/tokens
- Generate a new token with write access to both `ulfboge/galago-demo-vikter` and `ulfboge/galago-call-demo`
- Re-authenticate: `hf auth login --token <new_token> --add-to-git-credential --force`
- Update `.env` with the new token if it's stored there

---

## 2. Verify the HF Space is running

**Update 2026-06-24:** Space startar och Gradio lyssnar (`3ef054e` — Starlette-pin + `species_ranges.json` i git). Container-logg ren efter redeploy.

- ~~Check build / fix Starlette crash~~ — **klart** (se `docs/session_2026-06-24_space_fix.md`)
- **Kvar:** smoke-test i webb-UI — upload `.wav` → prediction → (valfritt) observer-feedback + Slack
- Remember: two remotes — `git push origin main` (GitHub) **and** `git push space main` (HF Space)

---

## 3. Review relabels and retrain

~175 candidate rows in `data/relabels/` have never been ingested. Tanzania is the weak spot (81.8% top-1) driven by `rondoensis`/`zanzibaricus` confusion in the Pugu folder — likely a mislabeled folder, worth prioritising.

```bash
# Review pending candidates (listen + mark ingest=yes/no)
python scripts/review_feedback.py --only-pending

# After marking rows ingest=yes in the CSVs:
python scripts/ingest_relabels.py

# Full retrain
python scripts/train_cnn_all_species_improved.py

# Re-evaluate and compare against March 2026 numbers
python scripts/run_eval_regions.py --tag after_retrain_v3
```

Reference numbers to beat (post-fix, 2026-03-25):

| Region | Top-1 | Top-3 |
|--------|-------|-------|
| Malawi (24) | 100.0% | 100.0% |
| Kenya (34) | 97.1% | 97.1% |
| Tanzania (11) | 81.8% | 100.0% |

See `docs/session_eval_2026-03_retrain_v2.md` for full details.

---

## 4. Check for real user feedback

The demo log currently has only 1 entry (a March test). If the Space has been live, there may be accumulated feedback.

```bash
python scripts/review_feedback.py --only-pending
python scripts/export_feedback_to_relabels.py --dry-run
```

If there's usable feedback, ingest it alongside the relabel CSVs in step 3.

---

## 5. Upload Malawi ft4 weights to the Hub (if not already there)

The deploy status doc flagged this as "can be added later." Check whether `galago_cnn_malawi_spnov_ft4_best.keras` and `class_names_19_malawi_spnov_ft4.json` are in `ulfboge/galago-demo-vikter`. If not, upload them so the Space can use the best Malawi model.

```bash
huggingface-cli upload ulfboge/galago-demo-vikter \
  models/all_species/galago_cnn_malawi_spnov_ft4_best.keras \
  models/all_species/galago_cnn_malawi_spnov_ft4_best.keras
```

---

## Background context

- **Inference bug (fixed 2026-03-25):** mel images were not flipped vertically and were scaled `[0,255]` instead of `[0,1]` at inference time. Fix is in `predict_3stage_with_context.py::wav_window_to_rgb_fixed`. Do not revert.
- **Active model:** `models/all_species/galago_cnn_all_19classes_improved_best.keras` (16/19-class variants in same folder — ignore legacy top6/top7).
- **Mel parameters must stay consistent:** `SR=22050, N_MELS=128, N_FFT=2048, HOP_LENGTH=512, FMIN=200, FMAX=10000, TARGET_FRAMES=128`
- **Deployment profiles:** `configs/deployment_profiles.json` — use `--profile kenya_balanced`, `tanzania_balanced`, or `malawi_balanced` as appropriate.
