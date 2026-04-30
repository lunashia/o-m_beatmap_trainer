# osu-mania-beatmap-trainer

`osu-mania-beatmap-trainer` is a work-in-progress project for processing 7-key osu!mania charts into model-friendly representations.

## Recommended environment

- Python: `3.11`

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Project structure

```text
o-m_beatmap_trainer/
|
|- src/
|  |- parser/          # .osu parsing
|  |- quantize/        # tick conversion
|  |- events/          # chord/event building
|  |- grid/            # grid representation
|  |- audio/           # mel extraction
|  `- utils/
|
|- tests/              # unit tests
|
|- requirements.txt
|- README.md
`- .gitignore
```

## Current modules

- `src/parser/osu_parser.py`: parse `.osu` mania maps into structured data.
- `src/quantize/quantizer.py`: convert note timestamps to ticks.
- `src/events/events_builder.py`: group quantized notes into chord events.
- `src/grid/grid_builder.py`: convert notes into binary `[tick][lane]` grids.
- `src/audio/mel_extractor.py`: extract log-mel spectrogram features.

## Run tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Status

⚠️ Work in progress

Implemented:
- [x] .osu parser
- [x] quantization (time → tick)
- [x] event encoding (chords)
- [x] grid representation
- [x] mel spectrogram extraction

Not implemented:
- [ ] training samples builder
- [ ] model training
- [ ] event generation
