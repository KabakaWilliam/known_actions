import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from playwright.sync_api import sync_playwright

load_dotenv()


def handle_computer_actions(page, actions):
    key_map = {
        "CTRL": "Control",
        "CONTROL": "Control",
        "COMMAND": "Meta",
        "CMD": "Meta",
        "ALT": "Alt",
        "SHIFT": "Shift",
        "SPACE": " ",
        "ENTER": "Enter",
        "RETURN": "Enter",
        "TAB": "Tab",
        "ESCAPE": "Escape",
        "ESC": "Escape",
        "BACKSPACE": "Backspace",
        "DELETE": "Delete",
        "DEL": "Delete",
        "INSERT": "Insert",
        "INS": "Insert",
        "ARROWUP": "ArrowUp",
        "UP": "ArrowUp",
        "ARROWDOWN": "ArrowDown",
        "DOWN": "ArrowDown",
        "ARROWLEFT": "ArrowLeft",
        "LEFT": "ArrowLeft",
        "ARROWRIGHT": "ArrowRight",
        "RIGHT": "ArrowRight",
        "HOME": "Home",
        "END": "End",
        "PAGEUP": "PageUp",
        "PAGE_UP": "PageUp",
        "PAGEDOWN": "PageDown",
        "PAGE_DOWN": "PageDown",
    }

    for action in actions:
        action_type = action.get("type") if isinstance(action, dict) else action.type
        match action_type:
            case "click":
                x = action.get("x") if isinstance(action, dict) else action.x
                y = action.get("y") if isinstance(action, dict) else action.y
                button = action.get("button", "left") if isinstance(action, dict) else getattr(action, "button", "left")
                page.mouse.click(x, y, button=button)
            case "double_click":
                x = action.get("x") if isinstance(action, dict) else action.x
                y = action.get("y") if isinstance(action, dict) else action.y
                button = action.get("button", "left") if isinstance(action, dict) else getattr(action, "button", "left")
                page.mouse.dblclick(x, y, button=button)
            case "scroll":
                x = action.get("x") if isinstance(action, dict) else action.x
                y = action.get("y") if isinstance(action, dict) else action.y
                scroll_x = action.get("scrollX", 0) if isinstance(action, dict) else getattr(action, "scrollX", 0)
                scroll_y = action.get("scrollY", 0) if isinstance(action, dict) else getattr(action, "scrollY", 0)
                page.mouse.move(x, y)
                page.mouse.wheel(scroll_x, scroll_y)
            case "keypress":
                keys = action.get("keys", []) if isinstance(action, dict) else getattr(action, "keys", [])
                for key in keys:
                    mapped_key = key_map.get(key.upper(), key)
                    try:
                        page.keyboard.press(mapped_key)
                    except Exception:
                        continue
            case "type":
                text = action.get("text") if isinstance(action, dict) else action.text
                page.keyboard.type(text)
            case "wait":
                time.sleep(2)
            case "navigate":
                url = action.get("url") if isinstance(action, dict) else getattr(action, "url", None)
                if url:
                    page.goto(url, wait_until="domcontentloaded", timeout=10000)
            case "screenshot":
                pass
            case _:
                raise ValueError(f"Unsupported action: {action_type}")


def capture_screenshot(page):
    return page.screenshot(type="png")


def send_computer_screenshot(client, response_id, call_id, screenshot_base64):
    return client.responses.create(
        model="gpt-5.4",
        tools=[{"type": "computer"}],
        previous_response_id=response_id,
        input=[
            {
                "type": "computer_call_output",
                "call_id": call_id,
                "output": {
                    "type": "computer_screenshot",
                    "image_url": f"data:image/png;base64,{screenshot_base64}",
                    "detail": "original",
                },
            }
        ],
    )


def computer_use_loop(page, response, client, artifacts_dir: Path, debug: bool = False):
    iteration = 0
    max_iterations = 30

    while iteration < max_iterations:
        iteration += 1
        computer_call = next((item for item in response.output if item.type == "computer_call"), None)
        if computer_call is None:
            return response

        handle_computer_actions(page, computer_call.actions)
        time.sleep(1)
        screenshot = capture_screenshot(page)
        screenshot_base64 = base64.b64encode(screenshot).decode("utf-8")

        if debug:
            shot_path = artifacts_dir / f"screenshot_iteration_{iteration}.png"
            shot_path.write_bytes(screenshot)

        response = send_computer_screenshot(
            client=client,
            response_id=response.id,
            call_id=computer_call.call_id,
            screenshot_base64=screenshot_base64,
        )

    return response


def run_browser_task(payload: dict[str, Any], artifacts_dir: Path = None) -> dict[str, Any]:
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set inside the sandbox")

    client = OpenAI(api_key=openai_api_key)
    start_url = payload["start_url"]
    task_description = payload["task"]
    debug = bool(payload.get("debug", False))

    if artifacts_dir is None:
        artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    enhanced_task = f"""{task_description}

IMPORTANT: You have access to a 'navigate' action to switch between sites directly.
When you need to go to a different domain/website, use the navigate action with the URL instead of trying to click the address bar.
This will be much faster and more reliable for cross-site navigation."""

    response = client.responses.create(
        model="gpt-5.4",
        tools=[{"type": "computer"}],
        input=enhanced_task,
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=[
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--no-default-browser-check",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            screen={"width": 1440, "height": 900},
            locale="en-GB",
            timezone_id="Europe/London",
        )
        page = context.new_page()

        page.goto(start_url, wait_until="domcontentloaded", timeout=10000)
        initial_screenshot = page.screenshot()
        (artifacts_dir / "initial_screenshot.png").write_bytes(initial_screenshot)

        final_response = computer_use_loop(page, response, client, artifacts_dir=artifacts_dir, debug=debug)

        response_text = ""
        if final_response.output:
            for output in final_response.output:
                if hasattr(output, "content"):
                    for content in output.content:
                        if hasattr(content, "text"):
                            response_text += content.text

        result = {
            "status": "ok",
            "answer": response_text,
            "response": {
                "id": final_response.id,
                "model": final_response.model,
                "status": final_response.status,
                "usage": {
                    "input_tokens": final_response.usage.input_tokens,
                    "output_tokens": final_response.usage.output_tokens,
                    "total_tokens": final_response.usage.total_tokens,
                },
                "completed_at": final_response.completed_at,
            },
            "artifacts": {
                "initial_screenshot": str(artifacts_dir / "initial_screenshot.png"),
            },
        }

        browser.close()
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    args = parser.parse_args()

    payload_path = Path(args.payload)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    
    # Extract the timestamped artifacts directory from the payload path
    artifacts_dir = payload_path.parent
    
    result = run_browser_task(payload, artifacts_dir=artifacts_dir)
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        sys.exit(1)
