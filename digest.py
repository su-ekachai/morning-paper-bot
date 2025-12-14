"""Morning Paper Bot — fetch RSS, pick top 10 with an LLM, post to Discord/Telegram."""

import argparse
import calendar
import difflib
import html
import json
import logging
import os
import re
import time
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

log = logging.getLogger("digest")

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
BANGKOK = timezone(timedelta(hours=7))  # Fixed offset; Thailand observes no daylight saving time
TOPIC_COLORS = {"tech": 3447003, "thai": 15158332, "finance": 3066993, "crypto": 15105570}

SUMMARY_MAX = 600  # Characters per summary; keeps items scannable and Telegram under its 4096 limit
TITLE_MAX = 150    # Characters for the source reference link
MAX_STORIES = 50   # Safety cap only; the real digest size is the number written in prompt.txt

SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "stories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["id", "title", "summary"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["stories"],
    "additionalProperties": False,
}


# --- config ---

def env(key, default=None):
    """Return the environment value for key, treating blank/whitespace as unset.

    GitHub Actions passes an unconfigured ${{ vars.X }} as an empty string rather than
    omitting it, so os.environ.get(key, default) would return "" instead of the default.
    """
    return os.environ.get(key, "").strip() or default


def load_env(path=None):
    """Load KEY=VALUE lines from .env into os.environ (real env wins). No-op if absent."""
    path = Path(path) if path else Path(__file__).parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def load_sources(path=None):
    """Read sources.toml into a list of (name, url, topic, lang) tuples."""
    path = Path(path) if path else Path(__file__).parent / "sources.toml"
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    sources = []
    for i, s in enumerate(data.get("source", []), 1):
        missing = [k for k in ("name", "url", "topic", "lang") if not s.get(k)]
        if missing:
            label = s.get("name") or f"entry #{i}"
            raise ValueError(f"{path.name}: source '{label}' is missing {', '.join(missing)}")
        sources.append((s["name"], s["url"], s["topic"], s["lang"]))
    if not sources:
        raise ValueError(f"{path.name}: no [[source]] entries found")
    return sources


# --- fetch ---

def fetch_feed(name, url, topic, lang):
    import feedparser

    resp = requests.get(url, timeout=10, headers=UA)
    resp.raise_for_status()
    feed = feedparser.parse(resp.content)
    entries = []
    skipped = 0
    for e in feed.entries:
        link, title = e.get("link"), e.get("title")
        dated = e.get("published_parsed") or e.get("updated_parsed")
        if not (link and title and dated):
            skipped += 1
            continue
        snippet = re.sub(r"<[^>]+>", "", e.get("summary", "")).strip()[:300]
        entries.append({
            "title": title.strip(),
            "url": link,
            "snippet": snippet,
            "source": name,
            "topic": topic,
            "lang": lang,
            "published_ts": calendar.timegm(dated),
        })
    if skipped:
        log.debug("%s: skipped %d entries without link/title/date", name, skipped)
    return entries


def fetch_all(sources):
    entries = []
    for name, url, topic, lang in sources:
        try:
            got = fetch_feed(name, url, topic, lang)
            log.info("%s: %d entries", name, len(got))
            entries.extend(got)
        except Exception as e:
            # one dead feed must not kill the digest; full traceback under LOG_LEVEL=DEBUG
            log.warning("%s: fetch failed: %s", name, e, exc_info=log.isEnabledFor(logging.DEBUG))
    return entries


def filter_today(entries, hours=24, per_feed_cap=15):
    cutoff = time.time() - hours * 3600
    fresh = [e for e in entries if e["published_ts"] >= cutoff]
    fresh.sort(key=lambda e: e["published_ts"], reverse=True)
    capped, counts = [], {}
    for e in fresh:
        n = counts.get(e["source"], 0)
        if n < per_feed_cap:
            counts[e["source"]] = n + 1
            capped.append(e)
    for i, e in enumerate(capped, 1):
        e["id"] = i
    return capped


# --- LLM ---

def build_prompt(candidates):
    """Render the editable prompt.txt template, filling the DATE/COUNT/CANDIDATES tokens.

    Uses str.replace (not str.format) because the template contains literal JSON braces.
    """
    template = (Path(__file__).parent / "prompt.txt").read_text(encoding="utf-8")
    date_str = datetime.now(BANGKOK).strftime("%A %d %B %Y")
    candidate_lines = "\n".join(
        f"[{c['id']}] ({c['topic']}, lang={c['lang']}, {c['source']}) {c['title']} — {c['snippet']}"
        for c in candidates
    )
    return (
        template.replace("{{DATE}}", date_str)
        .replace("{{COUNT}}", str(len(candidates)))
        .replace("{{CANDIDATES}}", candidate_lines)
    )


def check_response(resp, what):
    """Raise with the response body included — the body is where the real reason lives."""
    if not resp.ok:
        raise RuntimeError(f"{what} returned HTTP {resp.status_code}: {resp.text[:500]}")


def call_anthropic(prompt):
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("anthropic path requires ANTHROPIC_API_KEY")
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=env("LLM_MODEL", "claude-opus-4-8"),
        max_tokens=8000,
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    texts = [b.text for b in resp.content if b.type == "text"]
    if not texts:
        raise RuntimeError(f"Anthropic returned no text block (stop_reason={resp.stop_reason})")
    return parse_json_object("".join(texts))


def call_openai(prompt):
    key, model = env("OPENAI_API_KEY"), env("LLM_MODEL")
    if not key or not model:
        raise RuntimeError("openai path requires OPENAI_API_KEY and LLM_MODEL")
    base = env("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    resp = requests.post(
        f"{base}/chat/completions",
        timeout=300,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        },
    )
    check_response(resp, f"LLM ({model})")
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"unexpected LLM response: {str(data)[:500]}") from None
    if not content:
        raise RuntimeError(f"LLM returned empty content: {str(data)[:500]}")
    return parse_json_object(content)


def parse_json_object(text):
    """Parse a JSON object from an LLM reply, tolerating code fences and surrounding prose."""
    text = text.strip()
    try:
        return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text))
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in LLM reply: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def _similarity(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def resolve_candidate(candidates, by_id, cid, title):
    """Map an LLM pick back to its candidate, correcting a wrong id via the returned title.

    Weak models sometimes attach a coherent title+summary to the wrong id. The id is a
    fast path; when its candidate's headline does not match the returned title, re-route to
    the best title match so source/url/topic follow the story the summary is actually about.
    """
    c = by_id.get(cid)
    if not title:
        return c  # model omitted the title — trust the id (legacy behaviour)
    if c and _similarity(c["title"], title) >= 0.6:
        return c
    best = max(candidates, key=lambda x: _similarity(x["title"], title))
    if _similarity(best["title"], title) >= 0.6:
        return best
    return c  # no confident title match; fall back to the id, may be None


def llm_select(candidates):
    prompt = build_prompt(candidates)
    provider = env("LLM_PROVIDER", "anthropic")
    if provider not in ("anthropic", "openai"):
        raise RuntimeError(f"LLM_PROVIDER must be 'anthropic' or 'openai', got {provider!r}")
    log.info("asking %s to pick top 10 from %d candidates", provider, len(candidates))
    data = call_openai(prompt) if provider == "openai" else call_anthropic(prompt)
    by_id = {c["id"]: c for c in candidates}
    stories, seen = [], set()
    for s in data.get("stories", []):
        summary = (s.get("summary") or "").strip()
        if not summary:
            continue
        c = resolve_candidate(candidates, by_id, s.get("id"), (s.get("title") or "").strip())
        if c and c["id"] not in seen:
            seen.add(c["id"])
            stories.append({**c, "summary": summary})
    if not stories:
        raise RuntimeError(f"LLM returned no valid stories: {data}")
    return stories[:MAX_STORIES]


# --- format ---

def clip(text, limit):
    """Trim text to limit characters, appending an ellipsis when cut, for readable truncation."""
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def format_discord(stories, date_str):
    # Summary is the body; the headline is a small reference link at the bottom. No embed
    # title (it would render large and dominate). 4 embeds/message stays under Discord's
    # 6000-char total-per-message embed budget.
    embeds = [
        {
            "description": f"{clip(s['summary'], SUMMARY_MAX)}\n\n[{clip(s['title'], TITLE_MAX)}]({s['url']})",
            "footer": {"text": f"{s['source']} · {s['topic']}"},
            "color": TOPIC_COLORS.get(s["topic"], 0),
        }
        for s in stories
    ]
    header = f"**Morning Paper — {date_str}**"
    return [
        {"content": header, "embeds": embeds[:4]} if i == 0 else {"embeds": embeds[i : i + 4]}
        for i in range(0, len(embeds), 4)
    ]


def format_telegram(stories, date_str):
    blocks = []
    for i, s in enumerate(stories, 1):
        # Summary first (the focus), then a small reference line linking the source headline.
        blocks.append(
            f"<b>{i}.</b> {html.escape(clip(s['summary'], SUMMARY_MAX))}\n"
            f'<a href="{html.escape(s["url"], quote=True)}">↗ {html.escape(clip(s["title"], TITLE_MAX))}</a>'
            f' · {s["topic"]}'
        )
    messages, current = [], f"<b>Morning Paper — {date_str}</b>"
    for block in blocks:
        if len(current) + len(block) + 2 > 3900:
            messages.append(current)
            current = block
        else:
            current += "\n\n" + block
    messages.append(current)
    return messages


# --- deliver ---

def post_discord(payloads):
    url = os.environ["DISCORD_WEBHOOK_URL"]
    for p in payloads:
        check_response(requests.post(url, json=p, timeout=30), "Discord")


def post_telegram(texts):
    token, chat_id = os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
    for t in texts:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            timeout=30,
            json={"chat_id": chat_id, "text": t, "parse_mode": "HTML", "disable_web_page_preview": True},
        )
        check_response(resp, "Telegram")  # token stays out of the error (not in `what`)


def main():
    load_env()
    logging.basicConfig(
        level=env("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Daily news digest bot")
    parser.add_argument("--dry-run", action="store_true", help="print payloads instead of posting")
    args = parser.parse_args()

    candidates = filter_today(fetch_all(load_sources()))
    log.info("%d candidates in the last 24h", len(candidates))
    if not candidates:
        log.info("no candidates today, nothing to send")
        return

    stories = llm_select(candidates)
    date_str = datetime.now(BANGKOK).strftime("%a %d %b %Y")
    discord_payloads = format_discord(stories, date_str)
    telegram_messages = format_telegram(stories, date_str)

    if args.dry_run:
        print("=== DISCORD PAYLOADS ===")
        print(json.dumps(discord_payloads, ensure_ascii=False, indent=2))
        print("=== TELEGRAM MESSAGES ===")
        for m in telegram_messages:
            print(f"--- ({len(m)} chars) ---\n{m}")
        return

    sent = []
    if os.environ.get("DISCORD_WEBHOOK_URL"):
        post_discord(discord_payloads)
        sent.append("discord")
    if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        post_telegram(telegram_messages)
        sent.append("telegram")
    if not sent:
        raise RuntimeError(
            "no delivery channel configured (set DISCORD_WEBHOOK_URL and/or TELEGRAM_BOT_TOKEN+TELEGRAM_CHAT_ID)"
        )
    log.info("sent %d stories to: %s", len(stories), ", ".join(sent))


if __name__ == "__main__":
    main()
