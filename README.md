# Instagram carousel agent

This program publishes one Instagram carousel on a schedule. Three agents run in order, and the third one can stop a post from going out.

1. **Analyst** reads saved post performance (likes, comments, and insights such as reach, saves, and shares) and chooses the line each slide must deliver.
2. **Creator** writes the caption, the on-slide headline and body, and a photograph prompt per slide. OpenRouter generates the photographs. Headlines are drawn onto the photos in code, because image models still render words unreliably.
3. **Visual editor** looks at the finished slides. Publishing runs only after this agent approves the set. A rejected carousel is revised and checked again, up to `MAX_QA_RETRIES`.

Each run is saved under `output/<timestamp>/` with the JPEGs, `caption.txt`, the brief, and `manifest.json`.

## Setup

Use Python 3.11 or newer. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`. These values start as placeholders:

- `OPENROUTER_API_KEY` — analyst, creator, visual editor, and photograph generation
- `INSTAGRAM_ACCESS_TOKEN` and `INSTAGRAM_ACCOUNT_ID` — the professional Instagram account you own

Also replace `ACCOUNT_NICHE` with what the account actually posts about. `POST_DRY_RUN` defaults to `true`, so a run will not publish until you set it to `false` or pass `--publish`.

Check configuration without calling any API:

```powershell
python -m insta_agent check
```

## Commands

```powershell
python -m insta_agent refresh
python -m insta_agent run --dry-run
python -m insta_agent run --publish
python -m insta_agent schedule
python -m insta_agent publish output\20261109T120000Z
```

`refresh` stores recent posts in `data/performance.json`. The analyst uses that file on the next run. `schedule` posts every `POST_INTERVAL_HOURS` (minimum 1, default 24) and starts the first run immediately. Stop it with Ctrl+C.

`run --dry-run` still calls OpenRouter. It saves an approved carousel and does not upload it. `publish` sends an already approved folder. It refuses a folder the visual editor rejected, and it refuses files that were swapped after the check.

## Instagram

The token has to belong to an Instagram professional account linked the way your app logs in:

- Facebook Login: leave `GRAPH_API_BASE` at `https://graph.facebook.com/v27.0`
- Instagram Login: set `GRAPH_API_BASE` to `https://graph.instagram.com/v27.0`

The token needs `instagram_basic`, `instagram_content_publish`, `pages_read_engagement`, and `instagram_manage_insights`. Insights can be missing on some media; the analyst still sees likes and comments. If Instagram or your region asks you to label AI-generated images, leave that label on when you post.

Instagram fetches carousel images by URL. The default host is catbox.moe, which needs no extra key. If that host is blocked, or Instagram cannot fetch the file, set:

```
IMAGE_HOST=public_dir
IMAGE_PUBLIC_DIR=C:\path\to\a\folder\your\site\already\serves
IMAGE_PUBLIC_BASE_URL=https://cdn.example.com/instagram
```

The base URL must be `https://`. Carousels are 2 to 10 slides, 1080 by 1350 pixels.

## Output

| Status in manifest.json | Meaning |
| --- | --- |
| `ready` | Visual editor approved it. Safe to publish. |
| `rejected` | The editor still disagreed after the retries. Nothing was posted. |
| `published` | Instagram accepted the carousel. The media id and permalink are in the manifest, and the post is in `data/performance.json` for the next analyst run. |
