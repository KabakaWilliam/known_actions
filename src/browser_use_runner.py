"""Run one browser-use episode while recording website-visible browser events.

The browser-use agent and a read-only Playwright observer share the same
Chromium instance over CDP.  Playwright is used only to inject page_tracer.js
and receive the events it emits; browser-use remains the sole actor.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Literal

# Apptainer runs with --no-home; browser-use must not try to initialize its
# default profile/config beneath the host user's read-only home directory.
# The orchestrator binds a private host tmpfs directory at this path and sets
# TMPDIR before Python starts, so browser-use's own tempfile calls stay inside
# the episode boundary too.
EPISODE_TMP_ROOT = Path(
    os.environ.get("BROWSER_USE_EPISODE_TMP")
    or os.environ.get("TMPDIR")
    or "/tmp"
)
os.environ.setdefault(
    "BROWSER_USE_CONFIG_DIR", str(EPISODE_TMP_ROOT / "config")
)
os.environ.setdefault("XDG_CONFIG_HOME", str(EPISODE_TMP_ROOT / "xdg"))
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("BROWSER_USE_CLOUD_SYNC", "false")

from browser_use import Agent, BrowserSession
from browser_use.llm import ChatOpenAI, ChatOpenRouter
from playwright.async_api import Browser as PlaywrightBrowser
from playwright.async_api import BrowserContext, Page, Playwright, async_playwright
from pydantic import BaseModel


class QAResult(BaseModel):
    answer: str
    confidence: Literal["high", "medium", "low"]
    sources: list[str]


UNKNOWN_ANSWER_SENTINELS = {"", "na", "n/a", "none", "null", "unknown"}


def _has_known_answer(value: str | None) -> bool:
    return (
        value is not None
        and value.strip().lower() not in UNKNOWN_ANSWER_SENTINELS
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _model_setting(name: str, default: str | None = None) -> str | None:
    """Read a browser-use override, falling back to the existing MidScene env."""
    return os.getenv(f"BROWSER_USE_MODEL_{name}") or os.getenv(f"MIDSCENE_MODEL_{name}") or default


def _resolve_chromium() -> str:
    configured = os.getenv("BROWSER_USE_CHROME_PATH")
    if configured and Path(configured).is_file():
        return configured

    candidates: list[Path] = []
    ms_playwright = Path("/ms-playwright")
    if ms_playwright.exists():
        candidates.extend(ms_playwright.glob("chromium-*/chrome-linux*/chrome"))

    candidates.extend(
        Path(path)
        for path in (
            "/usr/bin/google-chrome",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
        )
    )
    for candidate in sorted(candidates, reverse=True):
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "No Chromium executable found. Set BROWSER_USE_CHROME_PATH or install Chromium."
    )


def _build_llm():
    model = _model_setting("NAME")
    api_key = _model_setting("API_KEY", "not-needed")
    base_url = _model_setting("BASE_URL")
    if not model:
        raise ValueError("Missing BROWSER_USE_MODEL_NAME or MIDSCENE_MODEL_NAME")

    provider = (os.getenv("BROWSER_USE_MODEL_PROVIDER") or "").strip().lower()
    if not provider:
        provider = "openrouter" if base_url and "openrouter.ai" in base_url else "openai"

    if provider == "openrouter":
        return ChatOpenRouter(
            model=model,
            api_key=api_key,
            base_url=base_url or "https://openrouter.ai/api/v1",
        )
    if provider == "openai":
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            add_schema_to_system_prompt=_env_bool(
                "BROWSER_USE_ADD_SCHEMA_TO_SYSTEM_PROMPT", False
            ),
            dont_force_structured_output=_env_bool(
                "BROWSER_USE_DONT_FORCE_STRUCTURED_OUTPUT", False
            ),
            max_completion_tokens=int(
                os.getenv("BROWSER_USE_MAX_COMPLETION_TOKENS", "4096")
            ),
        )
    raise ValueError(f"Unsupported BROWSER_USE_MODEL_PROVIDER: {provider}")


class WebsiteTraceCollector:
    """Observe the browser through the same Playwright APIs as the MidScene runner."""

    def __init__(self, episode_start: float, injector_script: str):
        self.episode_start = episode_start
        self.injector_script = injector_script
        self.events: list[dict[str, Any]] = []
        self._cursors: dict[Page, int] = {}
        self._attached_pages: set[Page] = set()
        self._playwright: Playwright | None = None
        self._observer_browser: PlaywrightBrowser | None = None

    def _elapsed_ms(self) -> int:
        return round((time.monotonic() - self.episode_start) * 1000)

    async def connect(self, cdp_url: str) -> None:
        self._playwright = await async_playwright().start()
        self._observer_browser = await self._playwright.chromium.connect_over_cdp(cdp_url)
        if not self._observer_browser.contexts:
            raise RuntimeError("Browser-use Chromium did not expose a browser context")

        for context in self._observer_browser.contexts:
            await self._instrument_context(context)

    async def _instrument_context(self, context: BrowserContext) -> None:
        def receive_event(source: dict[str, Any], event: dict[str, Any]) -> None:
            page = source.get("page")
            self.events.append({**event, "t_episode": self._elapsed_ms()})
            if page is not None:
                self._cursors[page] = self._cursors.get(page, 0) + 1

        await context.expose_binding("__pushTraceEvent", receive_event)
        await context.add_init_script(self.injector_script)
        context.on("page", self._attach_page)
        for page in context.pages:
            self._attach_page(page)

    def _attach_page(self, page: Page) -> None:
        if page in self._attached_pages:
            return
        self._attached_pages.add(page)
        self._cursors[page] = 0

        def on_navigation(frame) -> None:
            if frame != page.main_frame:
                return
            url = frame.url
            if url == "about:blank":
                return
            self._cursors[page] = 0
            self.events.append(
                {
                    "type": "navigate",
                    "t_episode": self._elapsed_ms(),
                    "url": url,
                    "trigger": "http",
                }
            )

        page.on("framenavigated", on_navigation)

    async def harvest(self) -> None:
        """Backstop events whose binding call did not complete before collection."""
        for page in list(self._attached_pages):
            try:
                cursor = self._cursors.get(page, 0)
                fresh = await page.evaluate(
                    """cursor => {
                        const trace = window.__agentTrace;
                        return trace ? trace.events.slice(cursor) : [];
                    }""",
                    cursor,
                )
                now = self._elapsed_ms()
                self.events.extend({**event, "t_episode": now} for event in fresh)
                self._cursors[page] = cursor + len(fresh)
            except Exception:
                # A page may close or navigate while the final harvest runs.
                continue

    async def close(self) -> None:
        # Stopping Playwright disconnects the observer without becoming an actor.
        if self._playwright is not None:
            await self._playwright.stop()
        self._playwright = None
        self._observer_browser = None


def _safe_json(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _parse_qa_result(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        parsed = QAResult.model_validate_json(raw)
    except Exception:
        parsed = QAResult(answer=raw, confidence="low", sources=[])
    return parsed.model_dump()


async def run_episode(args: argparse.Namespace) -> int:
    episode_start = time.monotonic()
    script_path = Path(__file__).with_name("page_tracer.js")
    output_path = Path(args.output_dir) / f"{args.episode_id}.json"
    task_prompt = args.task_prompt or (
        f"Complete the following task starting at {args.start_url}:\n{args.question}"
    )

    browser: BrowserSession | None = None
    collector: WebsiteTraceCollector | None = None
    agent = None
    history = None
    execution_error: str | None = None
    task_timed_out = False
    downloads_path = EPISODE_TMP_ROOT / "downloads"
    xdg_config_path = EPISODE_TMP_ROOT / "xdg"
    user_data_path = EPISODE_TMP_ROOT / "profile"

    try:
        # Apptainer runs with --no-home. Chromium's Crashpad process otherwise
        # exits before opening CDP because it cannot derive a writable crash
        # database location. Keep that state episode-local for parallel runs.
        xdg_config_path.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CONFIG_HOME"] = str(xdg_config_path)

        browser = BrowserSession(
            executable_path=_resolve_chromium(),
            headless=True,
            viewport={"width": 1280, "height": 768},
            window_size={"width": 1280, "height": 768},
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
            downloads_path=downloads_path,
            user_data_dir=user_data_path,
            enable_default_extensions=False,
            keep_alive=False,
        )
        await browser.start()
        if not browser.cdp_url:
            raise RuntimeError("browser-use started Chromium without a CDP URL")

        collector = WebsiteTraceCollector(episode_start, script_path.read_text())
        await collector.connect(browser.cdp_url)

        output_schema = QAResult if args.task_type == "qa" else None
        agent = Agent(
            task=task_prompt,
            llm=_build_llm(),
            browser=browser,
            output_model_schema=output_schema,
            use_vision=_env_bool("BROWSER_USE_VISION", True),
            use_judge=False,
            generate_gif=False,
            max_failures=int(os.getenv("BROWSER_USE_MAX_FAILURES", "5")),
            max_actions_per_step=int(os.getenv("BROWSER_USE_MAX_ACTIONS_PER_STEP", "5")),
            llm_timeout=int(os.getenv("BROWSER_USE_LLM_TIMEOUT", "120")),
            step_timeout=int(os.getenv("BROWSER_USE_STEP_TIMEOUT", "180")),
            directly_open_url=True,
            enable_signal_handler=False,
        )
        agent_task = asyncio.create_task(
            agent.run(
                max_steps=int(
                    os.getenv(
                        "BROWSER_USE_MAX_STEPS",
                        os.getenv("MIDSCENE_REPLANNING_CYCLE_LIMIT", "40"),
                    )
                )
            )
        )
        done, _pending = await asyncio.wait({agent_task}, timeout=args.task_timeout)
        if agent_task in done:
            history = agent_task.result()
        else:
            task_timed_out = True
            agent_task.cancel()
            try:
                await asyncio.wait_for(agent_task, timeout=10)
            except (asyncio.CancelledError, TimeoutError):
                pass
            # Agent maintains its history incrementally, so retain completed
            # actions even when the overall task deadline is reached.
            history = agent.history
    except Exception as exc:
        execution_error = f"{type(exc).__name__}: {exc}"
    finally:
        if collector is not None:
            await collector.harvest()

        events = collector.events if collector is not None else []
        final_result = history.final_result() if history is not None else None
        result = _parse_qa_result(final_result) if args.task_type == "qa" else None
        agent_reported_success = (
            bool(history.is_successful()) if history is not None else False
        )

        verification = (
            {
                "correct": args.expected_answer.lower()
                in ((result or {}).get("answer") or "").lower(),
                "predicted": (result or {}).get("answer") or "",
                "ground_truth": args.expected_answer,
            }
            if _has_known_answer(args.expected_answer)
            else None
        )
        # For labeled tasks, success means the answer matches ground truth.
        # Otherwise retain browser-use's own completion judgement.
        task_success = (
            bool(verification["correct"])
            if verification is not None
            else agent_reported_success
        )
        task_success_source = (
            "ground_truth" if verification is not None else "agent_reported"
        )

        browser_use_log = (
            {
                "actions": _safe_json(history.model_actions()),
                "errors": history.errors(),
                "urls": history.urls(),
                "final_result": final_result,
                "agent_reported_success": agent_reported_success,
                "task_success": task_success,
                "task_timed_out": task_timed_out,
            }
            if history is not None
            else {}
        )

        episode = {
            "meta": {
                "episode_id": args.episode_id,
                "agent_id": args.agent_id,
                "harness": "browser_use",
                "model_name": _model_setting("NAME"),
                "model_family": os.getenv("MIDSCENE_MODEL_FAMILY"),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "question": args.question,
                "start_url": args.start_url,
                "task_type": args.task_type,
            },
            "result": result,
            "verification": verification,
            # A task-level failure remains a valid episode. This field is reserved
            # for runner/model-service exceptions, matching the existing validity rules.
            "error": execution_error,
            "task_success": task_success,
            "task_success_source": task_success_source,
            "task_timed_out": task_timed_out,
            "midscene_log": [],
            "browser_use_log": browser_use_log,
            "dom_trace": {
                "episodeId": args.episode_id,
                "agentId": args.agent_id,
                "episodeDuration": round((time.monotonic() - episode_start) * 1000),
                "events": events,
                "pageCount": len(
                    {
                        event.get("url")
                        for event in events
                        if event.get("type") == "navigate"
                        and event.get("trigger") == "http"
                        and event.get("url")
                    }
                ),
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(episode, indent=2))

        if collector is not None:
            try:
                await collector.close()
            except Exception:
                pass
        if browser is not None:
            try:
                await asyncio.wait_for(browser.kill(), timeout=10)
            except Exception:
                pass
        shutil.rmtree(downloads_path, ignore_errors=True)
        shutil.rmtree(xdg_config_path, ignore_errors=True)
        shutil.rmtree(user_data_path, ignore_errors=True)
        if agent is not None:
            shutil.rmtree(agent.agent_directory, ignore_errors=True)

    if execution_error:
        print(f"[ERROR] browser-use failed — partial trace saved → {output_path}")
        print(f"[DETAIL] {execution_error}")
        return 1
    print(f"[OK] Trace saved → {output_path}")
    print(f"[ANSWER] {json.dumps(result)}")
    print(f"[TASK_SUCCESS] {task_success}")
    if task_timed_out:
        print(f"[TASK_TIMEOUT] Agent stopped after {args.task_timeout:g}s; partial trace saved")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--agent_id", required=True)
    parser.add_argument("--episode_id", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start_url", default="https://en.wikipedia.org")
    parser.add_argument("--task_prompt", default="")
    parser.add_argument("--task_type", choices=["qa", "shop", "webgames"], default="qa")
    parser.add_argument("--expected-answer", default="")
    parser.add_argument("--task-timeout", type=float, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_episode(parse_args())))
