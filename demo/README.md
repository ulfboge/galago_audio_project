# Web demo (upload → species guess)

## Setup

```powershell
cd c:\Users\galag\GitHub\galago_audio_project
pip install -r requirements-demo.txt
```

If you skip this, `ModuleNotFoundError: No module named 'gradio'` will appear when you start the app.

## Run

```powershell
python demo/upload_predict_gradio.py
```

Open **http://127.0.0.1:7860** in your browser.

## Plats (lat / long)

- **Format:** *decimalgrader* (WGS84), t.ex. lat `-13.9626`, lon `33.7741`. **Inte** grader/minuter/sekunder (DMS). Om du bara har DMS, konvertera först (t.ex. i GIS eller sökmotor).
- **Förvald plats:** rullistan fyller lat/long (redigerbara efteråt).
- **Ingen plats:** bocka *Ingen plats — skicka inte lat/long…* så används ingen koordinat för kontext (samma som att lämna fält tomma och inte välja en punkt med riktiga värden).
- **Karta:** öppna **`demo/pick_location_map.html`** i webbläsaren (dubbelklick i Utforskaren), klicka på kartan, kopiera siffrorna till fälten i Gradio.

## Testlogg (filnamn + förslag)

Om **Spara till demo/logs/predictions.jsonl** är ikryssad (standard) läggs en rad per körning i **`demo/logs/predictions.jsonl`** (JSON Lines): filnamn, profil, `species_result`, top-3, detektor, plats-flaggor, tidsstämpel UTC. Bra för att jämföra manuella tester. Filen är **gitignorad** som standard; ta bort raden `demo/logs/predictions.jsonl` i `.gitignore` om du vill versionera loggen.

## Var ska referens-WAV ligga?

- I repot under **`data/raw_audio/<artsmapp>/`** (samma struktur som träningspipelinen förväntar sig). Många mappar är **tomma** i en minimal checkout — det betyder bara att inga filer är kopierade dit än; de är platshållare per art/slag.
- Du kan kopiera eller länka in mer material från t.ex. **`E:\Galagidae`** (en fil per art enligt undermapp och/eller filnamn). Därefter: fönster/mels via `scripts/ingest_raw_audio_to_training_mels.py` / befintliga ingest-flöden, och ev. uppdatera `data/recording_locations.json` för filspecifik plats.

## Notes

- **`.wav` input** — same format as training inference (`scripts/predict_3stage_with_context.py`).
- **`--wav`** scores a single file from the CLI without a filelist.
- **Gradio** uses **`demo/cached_predictor.py`**: detector + classifier load **once**; later clicks only run `run_single_wav` (much faster than spawning a subprocess each time). When you change **profile**, the cached predictor reloads the classifier if the profile sets `classifier_model` (e.g. Malawi **ft4**); the first prediction after switching may take a few seconds.
- **Not** intended as a public-facing deployment without HTTPS, rate limits, and disclaimer copy.

## Quick verification checklist

1. From repo root: `pip install -r requirements-demo.txt` (or `pip install gradio`) if needed.
2. Run `python demo/upload_predict_gradio.py` and open **http://127.0.0.1:7860**.
3. Upload a short **.wav**; with **balanced**, confirm you get a species line and top-3 block without errors.
4. Switch to **`malawi_balanced_auto`**, upload again (or same file): confirm output still runs (classifier reloads to ft4 on first use after the switch).
5. Optionally **`kenya_balanced_auto`** vs **`tanzania_balanced_auto`**: for Tanzania, try the checkbox *disable sp.nov→rondoensis remap* and confirm `postprocess_mode` in the log reflects your choice.
6. Optional: enter **real lat/lon** for context priors. Leave empty; if the UI shows **0 / 0**, the demo now treats that as *no coordinates* (Gradio otherwise sent null-island and skewed priors).

## Troubleshooting

| Symptom | Cause | What to do |
|--------|--------|------------|
| **This site can’t be reached** / `ERR_CONNECTION_REFUSED` on `127.0.0.1:7860` | No server is listening (most often: the app was never started, or it crashed on import). | In a terminal at repo root run `python demo/upload_predict_gradio.py` and leave that window open. Install Gradio first (see Setup). |
| Same error after starting the script | Port **7860** already used by another process (old Gradio, another tool). | Close the other app, or set `set GRADIO_SERVER_PORT=7861` then start again. The script also tries **7861–7869** automatically if 7860 is busy. |
| `ModuleNotFoundError: gradio` | Gradio not installed in the active Python. | `pip install -r requirements-demo.txt` |
