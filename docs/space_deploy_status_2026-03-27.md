# Statuslogg: Hugging Face Space & demo-vikter (2026-03-27)

Sammanfattning av var projektet ligger efter arbete med Model-repo, GitHub-sync och Space `galago-call-demo`.

**Uppdatering 2026-06-25:** Space **live** — inferens, karta (`/galago-map`, commit `524d854`), Starlette-pin (`3ef054e`). Hub-vikter **ej** uppdaterade med juni-omträning. Se `docs/session_2026-06-24_space_fix.md` och `docs/next_steps_2026-06-24.md`.

---

## Klart

### Model-repo (vikter på Hub)

- **Repo:** `ulfboge/galago-demo-vikter`
- **Licens på Hub:** MIT
- **Filer** (relativa sökvägar som demot förväntar sig):
  - `models/detector/galago_detector_best.keras`
  - `models/all_species/galago_cnn_all_19classes_improved_best.keras`
  - `models/all_species/class_names_19.json`
  - (Valfritt) Malawi + `species_ranges.json` kan läggas till senare
- Uppladdning skedde via `hf upload` med `path_in-repo` / korrekt mappstruktur (inte bara filer i roten).

### GitHub (`galago_audio_project`)

- **Branch:** `main`
- **Viktiga commits:**
  - `1102eb6` — Gradio-demo, `demo/`, rot-`requirements.txt`, `packages.txt`, HF-frontmatter i `README.md`, `configs/deployment_profiles.json`, uppdaterad `predict_3stage_with_context.py`
  - `229a7df` — `tensorflow-cpu>=2.20,<2.22` för **Python 3.13** på HF Spaces (äldre TF-pin hade inga cp313-hjul → build-fel)

### Space-repo (kod direkt på Hub)

- **Space:** `ulfboge/galago-call-demo`
- **Skäl till separat push:** Skapa-Space-formuläret hade **ingen** “koppla GitHub”-väg; lösningen blev **andra git-remote** `space` → `git push space main` från samma lokala repo som GitHub.
- **Auth:** Fine-grained token måste ha **write** till **Space-repot** (inte bara Model-repot). `hf auth login --token … --add-to-git-credential` synkar Git Credential Manager.
- **Vanliga fel vi såg:** `hf_hf_…` (dubbel prefix) = ogiltig token; tom `Bearer` = tom token / tom `HF_TOKEN` i miljö; `not authorized` = saknad write-rätt till just Space-repot.

### Kodstöd i repot

- `demo/ensure_hf_hub_models.py` + `GALAGO_HF_MODEL_REPO` laddar vikter vid start om variabeln är satt.
- `demo/README_spaces.md` — utökad guide (secrets, filstruktur på Hub, m.m.).

---

## Pågående / att verifiera

### Space-inställningar

- Under **Variables and secrets** ska finnas minst:
  - **`GALAGO_HF_MODEL_REPO`** = `ulfboge/galago-demo-vikter`
- Valfritt: **`DEMO_FEEDBACK_WEBHOOK_URL`** (secret) för Slack/Discord.

### Build & runtime (juni 2026)

- **Starlette:** pin `starlette>=0.37.2,<1.0.0` i `requirements.txt` / `requirements-demo.txt` (Gradio 4.44 + Starlette 1.0 → Jinja2-krasch).
- **Karta:** `/galago-map` via patch av `App.create_app` i `upload_predict_gradio.py` — inte `mount_gradio_app`+uvicorn på Space.
- **Modell på Hub:** fortfarande pre-juni `improved_best` tills ny uppladdning efter snippet-relabel + acceptans-test.

### Build & loggar

- Första bygget efter TF-fix kan ta **lång tid** (TensorFlow + dependencies).
- Om **Build**-loggen är tom: använd **Logs** bredvid status **Building**, vänta, F5, ev. **Factory reboot**; tillägg som blockerar WebSocket kan ge tom vy.

### Säkerhet

- En **access token** har exponerats i chat under felsökning — **den ska antas komprometterad:** revoke på [HF token settings](https://huggingface.co/settings/tokens) och skapa ny, sedan `hf auth login --token … --add-to-git-credential --force`.

---

## Rekommenderade nästa steg

1. Bekräfta att **Build** går igenom utan pip-fel (särskilt efter `229a7df`).
2. Öppna **App** och testa uppladdning av en kort `.wav`.
3. Vid runtime-fel: läs **Container**-loggar (efter lyckad build).
4. Långsiktigt: `git push origin main` + `git push space main` när demot ändras (två remotes).

---

## Snabbreferens

| Del | Plats |
|-----|--------|
| Kod (GitHub) | `github.com/ulfboge/galago_audio_project` |
| Kod (Space git) | `huggingface.co/spaces/ulfboge/galago-call-demo` |
| Vikter | `huggingface.co/ulfboge/galago-demo-vikter` |
| Demofil | `demo/upload_predict_gradio.py` (styrs av `README.md` YAML) |
