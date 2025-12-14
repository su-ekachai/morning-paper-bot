import os
import time

import pytest

import digest


class FakeResp:
    def __init__(self, status=200, text="", payload=None):
        self.status_code = status
        self.ok = status < 400
        self.text = text
        self._payload = payload

    def json(self):
        return self._payload


def make_candidate(i=1, topic="tech", lang="en", **over):
    c = {
        "id": i,
        "title": f"Title {i}",
        "url": f"https://example.com/{i}",
        "snippet": "snippet",
        "source": "Src",
        "topic": topic,
        "lang": lang,
        "published_ts": time.time(),
    }
    c.update(over)
    return c


# --- load_sources (sources.toml) ---

def test_load_sources_valid(tmp_path):
    p = tmp_path / "sources.toml"
    p.write_text(
        '[[source]]\nname = "HN"\nurl = "https://hnrss.org/frontpage"\ntopic = "tech"\nlang = "en"\n'
        '\n[[source]]\nname = "Thairath"\nurl = "https://thairath.co.th/rss"\ntopic = "thai"\nlang = "th"\n',
        encoding="utf-8",
    )
    assert digest.load_sources(p) == [
        ("HN", "https://hnrss.org/frontpage", "tech", "en"),
        ("Thairath", "https://thairath.co.th/rss", "thai", "th"),
    ]


def test_load_sources_missing_key_raises(tmp_path):
    p = tmp_path / "sources.toml"
    p.write_text('[[source]]\nname = "HN"\nurl = "https://x.com"\ntopic = "tech"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="HN"):
        digest.load_sources(p)


def test_load_sources_default_file():
    # sources.toml is user-editable; assert it parses to well-formed tuples, not a fixed count.
    sources = digest.load_sources()
    assert len(sources) >= 1
    assert all(len(s) == 4 for s in sources)


# --- env (blank-tolerant getter) ---

def test_env_returns_value_when_set(monkeypatch):
    monkeypatch.setenv("X_TEST", "openai")
    assert digest.env("X_TEST", "anthropic") == "openai"


def test_env_returns_default_when_missing(monkeypatch):
    monkeypatch.delenv("X_TEST", raising=False)
    assert digest.env("X_TEST", "anthropic") == "anthropic"


def test_env_returns_default_when_blank(monkeypatch):
    monkeypatch.setenv("X_TEST", "   ")  # GH Actions passes unset vars as empty strings
    assert digest.env("X_TEST", "anthropic") == "anthropic"


def test_llm_select_blank_provider_uses_anthropic_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "")  # the P1: unset GH variable -> ""
    monkeypatch.setattr(digest, "call_anthropic", lambda prompt: {"stories": [{"id": 1, "summary": "S."}]})
    out = digest.llm_select([make_candidate(1)])
    assert out[0]["summary"] == "S."


# --- load_env (.env auto-loading) ---

def test_load_env_parses_and_strips_quotes(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    p = tmp_path / ".env"
    p.write_text('# comment\nLLM_PROVIDER=openai\n\nLLM_MODEL="openrouter/free"\n', encoding="utf-8")
    digest.load_env(p)
    assert os.environ["LLM_PROVIDER"] == "openai"
    assert os.environ["LLM_MODEL"] == "openrouter/free"


def test_load_env_does_not_override_real_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    p = tmp_path / ".env"
    p.write_text("LLM_PROVIDER=openai\n", encoding="utf-8")
    digest.load_env(p)
    assert os.environ["LLM_PROVIDER"] == "anthropic"  # CI secrets win over committed .env


def test_load_env_missing_file_is_noop(tmp_path):
    digest.load_env(tmp_path / "nope.env")  # must not raise


# --- filter_today ---

def test_filter_today_drops_old_and_assigns_ids():
    fresh = make_candidate(published_ts=time.time() - 3600)
    old = make_candidate(published_ts=time.time() - 48 * 3600)
    out = digest.filter_today([old, fresh])
    assert [e["title"] for e in out] == [fresh["title"]]
    assert out[0]["id"] == 1


def test_filter_today_caps_per_feed():
    entries = [make_candidate(i, source="Same") for i in range(30)]
    out = digest.filter_today(entries, per_feed_cap=15)
    assert len(out) == 15
    assert [e["id"] for e in out] == list(range(1, 16))


# --- llm_select ---

def test_llm_select_joins_dedupes_drops_bad_ids(monkeypatch):
    cands = [make_candidate(1, title="Alpha story"), make_candidate(2, title="Beta story")]
    fake = {"stories": [
        {"id": 1, "title": "Alpha story", "summary": "One."},
        {"id": 1, "title": "Alpha story", "summary": "dup"},
        {"id": 999, "title": "Unrelated ghost headline zzz", "summary": "bad"},
        {"id": 2, "title": "Beta story", "summary": "Two."},
    ]}
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(digest, "call_anthropic", lambda prompt: fake)
    out = digest.llm_select(cands)
    assert [(s["id"], s["summary"]) for s in out] == [(1, "One."), (2, "Two.")]
    assert out[0]["url"] == cands[0]["url"]  # canonical url from candidate, not LLM


def test_llm_select_realigns_summary_when_id_is_wrong(monkeypatch):
    # The observed bug: model writes a coherent title+summary but tags it with the wrong id.
    cands = [make_candidate(1, title="Oil tankers face worst case in Hormuz"),
             make_candidate(2, title="Apple Music raises prices worldwide")]
    fake = {"stories": [{"id": 1, "title": "Apple Music raises prices worldwide", "summary": "Prices up."}]}
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(digest, "call_anthropic", lambda prompt: fake)
    out = digest.llm_select(cands)
    assert out[0]["url"] == cands[1]["url"]  # realigned to the Apple Music candidate by title
    assert out[0]["title"] == "Apple Music raises prices worldwide"
    assert out[0]["summary"] == "Prices up."


def test_llm_select_falls_back_to_id_when_title_absent(monkeypatch):
    cands = [make_candidate(1, title="Alpha story")]
    fake = {"stories": [{"id": 1, "summary": "One."}]}  # no title -> old id-join
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(digest, "call_anthropic", lambda prompt: fake)
    out = digest.llm_select(cands)
    assert out[0]["title"] == "Alpha story" and out[0]["summary"] == "One."


def test_llm_select_returns_all_up_to_safety_cap(monkeypatch):
    # prompt.txt may ask for more than 10; picks are not silently clamped to 10.
    cands = [make_candidate(i, title=f"Story {i}") for i in range(1, 16)]
    fake = {"stories": [{"id": i, "title": f"Story {i}", "summary": f"S{i}."} for i in range(1, 16)]}
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(digest, "call_anthropic", lambda prompt: fake)
    assert len(digest.llm_select(cands)) == 15


def test_llm_select_caps_at_safety_max(monkeypatch):
    n = digest.MAX_STORIES + 10
    cands = [make_candidate(i, title=f"Story {i}") for i in range(1, n + 1)]
    fake = {"stories": [{"id": i, "title": f"Story {i}", "summary": f"S{i}."} for i in range(1, n + 1)]}
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(digest, "call_anthropic", lambda prompt: fake)
    assert len(digest.llm_select(cands)) == digest.MAX_STORIES


def test_llm_select_raises_on_no_valid_stories(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(digest, "call_anthropic", lambda prompt: {"stories": []})
    with pytest.raises(RuntimeError):
        digest.llm_select([make_candidate(1)])


# --- parse_json_object (robust LLM output parsing) ---

def test_parse_json_object_plain():
    assert digest.parse_json_object('{"stories": []}') == {"stories": []}


def test_parse_json_object_fenced():
    assert digest.parse_json_object('```json\n{"stories": [1]}\n```') == {"stories": [1]}


def test_parse_json_object_prose_wrapped():
    text = 'Sure! Here you go:\n{"stories": [{"id": 1, "summary": "x"}]}\nHope that helps.'
    assert digest.parse_json_object(text) == {"stories": [{"id": 1, "summary": "x"}]}


def test_parse_json_object_no_json_raises():
    with pytest.raises(ValueError):
        digest.parse_json_object("I cannot help with that.")


# --- HTTP error surfacing ---

def test_check_response_ok_passes():
    digest.check_response(FakeResp(200), "X")  # no raise


def test_check_response_error_includes_body():
    with pytest.raises(RuntimeError, match="model not found"):
        digest.check_response(FakeResp(404, text='{"error":{"message":"model not found"}}'), "LLM")


def test_call_openai_surfaces_http_error_body(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "bad/model")
    monkeypatch.setattr(
        digest.requests, "post",
        lambda *a, **k: FakeResp(404, text='{"error":{"message":"no such model bad/model"}}'),
    )
    with pytest.raises(RuntimeError, match="no such model"):
        digest.call_openai("p")


def test_call_openai_handles_error_shaped_200(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setattr(digest.requests, "post", lambda *a, **k: FakeResp(200, payload={"error": {"message": "x"}}))
    with pytest.raises(RuntimeError):
        digest.call_openai("p")


def test_call_openai_requires_config(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        digest.call_openai("p")


def test_llm_select_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    with pytest.raises(RuntimeError, match="LLM_PROVIDER"):
        digest.llm_select([make_candidate(1)])


# --- prompt (loaded from prompt.txt) ---

def test_build_prompt_lists_candidates_and_fills_tokens():
    p = digest.build_prompt([make_candidate(1, lang="th", source="Thairath")])
    assert "[1] (tech, lang=th, Thairath)" in p  # candidate line rendered by code
    assert "{{CANDIDATES}}" not in p and "{{DATE}}" not in p and "{{COUNT}}" not in p


# --- clip (readable truncation) ---

def test_clip_short_passes_through():
    assert digest.clip("hello", 10) == "hello"


def test_clip_truncates_with_ellipsis():
    out = digest.clip("a" * 100, 20)
    assert len(out) <= 20 and out.endswith("…")


def test_format_telegram_caps_long_summary():
    stories = [make_candidate(1, title="T", summary="ก" * 5000)]  # pathologically long Thai summary
    messages = digest.format_telegram(stories, "Sat 18 Jul 2026")
    assert all(len(m) <= 4096 for m in messages)
    assert "…" in "".join(messages)


# --- formatters ---

def ten_stories():
    return [
        make_candidate(i, title="ทดสอบ & <หัวข้อ> " + "x" * 300, summary="สรุป & ข่าว " + "y" * 1200)
        for i in range(1, 11)
    ]


def test_format_discord_summary_focused_and_under_limits():
    payloads = digest.format_discord(ten_stories(), "Sat 18 Jul 2026")
    assert sum(len(p["embeds"]) for p in payloads) == 10
    assert "content" in payloads[0] and all("content" not in p for p in payloads[1:])
    for p in payloads:
        assert len(p["embeds"]) <= 4  # stay under Discord's 6000-char/message embed budget
        assert sum(len(e["description"]) + len(e["footer"]["text"]) for e in p["embeds"]) <= 6000
        for e in p["embeds"]:
            assert "title" not in e          # summary is the body, not a big title
            assert "http" in e["description"]  # source link lives inside the description


def test_format_telegram_summary_first_escapes_and_packs():
    messages = digest.format_telegram(ten_stories(), "Sat 18 Jul 2026")
    assert all(len(m) <= 3900 for m in messages)
    joined = "\n".join(messages)
    assert "&amp;" in joined and "&lt;" in joined
    assert "<a href=" in joined


# --- fetch_feed date handling ---

SAMPLE_RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>T</title>
<item><title>Dated</title><link>https://x.com/a</link>
<pubDate>Fri, 17 Jul 2026 09:00:00 GMT</pubDate><description>d</description></item>
<item><title>Undated</title><link>https://x.com/b</link><description>d</description></item>
</channel></rss>"""


def test_fetch_feed_skips_undated_entries(monkeypatch):
    class FakeResp:
        content = SAMPLE_RSS

        def raise_for_status(self):
            pass

    monkeypatch.setattr(digest.requests, "get", lambda *a, **k: FakeResp())
    entries = digest.fetch_feed("T", "https://x.com/rss", "tech", "en")
    assert [e["title"] for e in entries] == ["Dated"]
    assert entries[0]["url"] == "https://x.com/a"
