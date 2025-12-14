# morning-paper-bot

Daily news digest bot. It fetches RSS feeds from tech, Thai, finance, and crypto sources,
selects the ten most important stories of the last 24 hours with a Large Language Model (LLM),
writes a short summary of each (with a link to the source), and posts the digest to Discord
and/or Telegram every morning at 09:00 Asia/Bangkok via GitHub Actions.

## How it works

1. `load_sources` reads the feed list from `sources.toml`.
2. `fetch_all` downloads each feed; a feed that fails is logged and skipped, never fatal.
3. `filter_today` keeps entries published in the last 24 hours (capped at 15 per feed).
4. `llm_select` sends all candidates to the LLM in one call and receives the top ten, each with
   a summary (Thai sources summarized in Thai, others in English).
5. `format_discord` / `format_telegram` render the digest; delivery posts to whichever channels
   have credentials configured.

## Prerequisites

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)
- One LLM credential: an Anthropic API key, or any OpenAI-compatible API key
- At least one delivery target: a Discord webhook, or a Telegram bot token and chat ID

## Setup

```sh
uv sync
cp .env.example .env   # fill in your keys
```

## Run locally

```sh
uv run python digest.py --dry-run   # fetch and summarize, print instead of posting
uv run python digest.py             # fetch, summarize, and post
```

The script auto-loads `.env` from the project root. Real environment variables take precedence,
so continuous integration (CI) secrets override any `.env` value. Sourcing `.env` manually is
not required.

Set `LOG_LEVEL=DEBUG` for verbose logs, including full tracebacks on feed failures.

## Edit news sources

Edit `sources.toml`. Each feed is one `[[source]]` block:

```toml
[[source]]
name = "Hacker News"                  # shown in the digest footer
url = "https://hnrss.org/frontpage"   # RSS or Atom feed URL
topic = "tech"                        # tech, thai, finance, crypto, or your own label
lang = "en"                           # "th" produces a Thai summary, "en" produces English
```

Add, remove, or reorder blocks freely. All four fields are required; a missing field stops the
run with the offending source named.

## Edit the LLM prompt

The instruction sent to the model lives in `prompt.txt`, editable without touching code. Three
tokens are filled in at runtime:

| Token | Replaced with |
|---|---|
| `{{DATE}}` | The current date in Asia/Bangkok |
| `{{COUNT}}` | The number of candidate stories |
| `{{CANDIDATES}}` | The rendered candidate list, one per line |

Edit the wording, language rules, or story count freely. The digest size is the number written in
`prompt.txt` (default 10); the code applies only a safety cap of 50. Keep the JSON contract intact:
each pick must return `id`, `title`, and `summary`. The `title` is what aligns each summary to the
correct source, so the model must copy the candidate headline verbatim.

Summaries longer than 600 characters and headlines longer than 150 are truncated with an ellipsis
for readability and to keep Telegram messages within its 4096-character limit.

## Configuration

Configuration is entirely through environment variables (see `.env.example`). An unset or blank
variable falls back to its default.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `LLM_PROVIDER` | no | `anthropic` | `anthropic` or `openai` |
| `LLM_MODEL` | openai path only | `claude-opus-4-8` (anthropic path) | Model identifier |
| `ANTHROPIC_API_KEY` | anthropic path | — | Anthropic API key |
| `OPENAI_BASE_URL` | no | `https://api.openai.com/v1` | OpenAI-compatible endpoint (for example `https://openrouter.ai/api/v1`) |
| `OPENAI_API_KEY` | openai path | — | OpenAI-compatible API key |
| `DISCORD_WEBHOOK_URL` | no | — | Set it to enable Discord delivery |
| `TELEGRAM_BOT_TOKEN` | no | — | Set with `TELEGRAM_CHAT_ID` to enable Telegram delivery |
| `TELEGRAM_CHAT_ID` | no | — | Target chat identifier |
| `LOG_LEVEL` | no | `INFO` | `INFO` or `DEBUG` |

## Deploy (GitHub Actions)

1. Push this repository to GitHub.
2. Open Settings, then Secrets and variables, then Actions:
   - **Secrets**: `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`), `DISCORD_WEBHOOK_URL`,
     `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` — only the ones in use.
   - **Variables** (optional): `LLM_PROVIDER`, `LLM_MODEL`, `OPENAI_BASE_URL`.
3. Open the Actions tab, select `daily-digest`, and choose *Run workflow* to test. The cron then
   runs daily at 02:00 UTC (09:00 Asia/Bangkok).

A failed run appears as a red mark in the Actions tab. That is the complete alerting mechanism.

## Getting the credentials

- **Discord webhook**: channel, then Edit Channel, then Integrations, then Webhooks, then New
  Webhook, then Copy Webhook URL.
- **Telegram**: create a bot with [@BotFather](https://t.me/BotFather) (`/newbot`) to get the
  token. Message the bot once, then open
  `https://api.telegram.org/bot<TOKEN>/getUpdates` and read `chat.id`.

## Limitations

This is a version 1 with deliberate scope boundaries:

- **Selection window is the last 24 hours only.** There is no cross-day deduplication state, so a
  story whose feed timestamp is later revised can appear on two days.
- **No retry on rate limits.** An HTTP 429 (or any API error) fails the run with the reason in the
  log; rerun the workflow.
- **Delivery is not atomic.** If Discord succeeds and Telegram then fails, the run exits nonzero
  and a rerun re-posts to Discord.
- **Undated feed entries are skipped**, because without a timestamp they cannot be placed in the
  24-hour window.
- **An empty news day sends nothing** and exits successfully.

## Development

```sh
uv run pytest       # tests
uv run ruff check . # lint
uv run ty check     # types
```

CI (`.github/workflows/ci.yml`) runs all three on every push.

## License

Released under the MIT License. See [LICENSE](LICENSE).
