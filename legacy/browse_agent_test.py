import os
import sys
import time
import base64
import json
from pprint import pprint

from openai import OpenAI
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY', '')


client_kwargs = {"api_key": openai_api_key}

client = OpenAI(**client_kwargs)

def handle_computer_actions(page, actions):
    # Mapping from OpenAI key names to Playwright canonical names
    # Based on Playwright's keyboard.press() - see: https://playwright.dev/python/docs/api/class-keyboard#keyboard-press
    key_map = {
        # Modifier keys
        "CTRL": "Control",
        "CONTROL": "Control",
        "COMMAND": "Meta",
        "CMD": "Meta",
        "ALT": "Alt",
        "SHIFT": "Shift",
        
        # Special characters
        "SPACE": " ",
        "ENTER": "Enter",
        "RETURN": "Enter",
        "TAB": "Tab",
        "ESCAPE": "Escape",
        "ESC": "Escape",
        
        # Editing keys
        "BACKSPACE": "Backspace",
        "DELETE": "Delete",
        "DEL": "Delete",
        "INSERT": "Insert",
        "INS": "Insert",
        
        # Navigation keys
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
        
        # Function keys
        "F1": "F1", "F2": "F2", "F3": "F3", "F4": "F4",
        "F5": "F5", "F6": "F6", "F7": "F7", "F8": "F8",
        "F9": "F9", "F10": "F10", "F11": "F11", "F12": "F12",
        
        # Numeric keypad
        "NUMPAD0": "0", "NUMPAD1": "1", "NUMPAD2": "2", "NUMPAD3": "3",
        "NUMPAD4": "4", "NUMPAD5": "5", "NUMPAD6": "6", "NUMPAD7": "7",
        "NUMPAD8": "8", "NUMPAD9": "9",
        "NUMPADADD": "+",
        "NUMPADDECIMAL": ".",
        "NUMPADDIVIDE": "/",
        "NUMPADMULTIPLY": "*",
        "NUMPADSUBTRACT": "-",
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
                scrollX = action.get("scrollX", 0) if isinstance(action, dict) else getattr(action, "scrollX", 0)
                scrollY = action.get("scrollY", 0) if isinstance(action, dict) else getattr(action, "scrollY", 0)
                page.mouse.move(x, y)
                page.mouse.wheel(scrollX, scrollY)
            case "keypress":
                keys = action.get("keys", []) if isinstance(action, dict) else getattr(action, "keys", [])
                for key in keys:
                    mapped_key = key_map.get(key.upper(), key)
                    try:
                        page.keyboard.press(mapped_key)
                    except Exception as e:
                        print(f"[Warning] Failed to press key '{mapped_key}' (original: '{key}'): {e}")
                        continue
            case "type":
                text = action.get("text") if isinstance(action, dict) else action.text
                page.keyboard.type(text)
            case "wait":
                time.sleep(2)
            case "navigate":
                url = action.get("url") if isinstance(action, dict) else getattr(action, "url", None)
                if url:
                    print(f"[Navigation] Navigating to {url}")
                    try:
                        page.goto(url, wait_until="domcontentloaded", timeout=10000)
                        print(f"[Navigation] Successfully navigated to {url}")
                    except Exception as e:
                        print(f"[Navigation] Failed to navigate to {url}: {e}")
                else:
                    print("[Navigation] No URL provided in navigate action")
            case "screenshot":
                pass
            case _:
                raise ValueError(f"Unsupported action: {action_type}")

def capture_screenshot(page):
    return page.screenshot(type="png")

def send_computer_screenshot(response_id, call_id, screenshot_base64):
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

def computer_use_loop(target, response, debug=False):
    iteration = 0
    max_iterations = 30
    
    while iteration < max_iterations:
        iteration += 1
        computer_call = next(
            (item for item in response.output if item.type == "computer_call"),
            None,
        )
        if computer_call is None:
            return response

        if debug:
            print(f"\n[Iteration {iteration}] Executing actions: {[a.get('type') if isinstance(a, dict) else a.type for a in computer_call.actions]}")

        handle_computer_actions(target, computer_call.actions)
        
        # Wait a bit for page to settle
        time.sleep(1)

        screenshot = capture_screenshot(target)
        screenshot_base64 = base64.b64encode(screenshot).decode("utf-8")
        
        if debug:
            # Save screenshot for debugging
            with open(f"screenshot_iteration_{iteration}.png", "wb") as f:
                f.write(base64.b64decode(screenshot_base64))
            print(f"Screenshot saved: screenshot_iteration_{iteration}.png (size: {len(screenshot_base64)} bytes)")
        
        response = send_computer_screenshot(response_id=response.id, call_id=computer_call.call_id, screenshot_base64=screenshot_base64)


    print(f"[Warning] Reached max iterations ({max_iterations})")
    return response

def run_browser_task(start_url, task_description, debug=False):
    """
    General solution for running browser automation tasks.
    The model decides which links to explore and navigate to.
    
    Args:
        start_url: Initial URL to start with
        task_description: Task description for the model to follow
        debug: Enable debug mode with screenshot saving
    """
    # Add action instructions to the task
    enhanced_task = f"""{task_description}

IMPORTANT: You have access to a 'navigate' action to switch between sites directly. 
When you need to go to a different domain/website, use the navigate action with the URL instead of trying to click the address bar.
This will be much faster and more reliable for cross-site navigation."""
    
    # Create initial request to the model
    response = client.responses.create(
        model="gpt-5.4",
        tools=[{"type": "computer"}],
        input=enhanced_task,
    )
    
    # Initialize Playwright browser
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
            device_scale_factor=1,
            color_scheme="light",
        )
        page = context.new_page()

        # page = browser.new_page(viewport={"width": 1440, "height": 900})
        
        # Navigate to starting page
        print(f"Navigating to {start_url}...")
        try:
            page.goto(start_url, wait_until="domcontentloaded", timeout=10000)
            print("Page loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load page: {e}")
        
        # Take initial screenshot
        initial_screenshot = page.screenshot()
        with open("initial_screenshot.png", "wb") as f:
            f.write(initial_screenshot)
        print(f"Initial screenshot saved: initial_screenshot.png ({len(initial_screenshot)} bytes)")
        
        # Run the computer use loop
        final_response = computer_use_loop(page, response, debug=debug)
        
        # Extract and display the text response
        response_text = ""
        if final_response.output:
            for output in final_response.output:
                if hasattr(output, 'content'):
                    for content in output.content:
                        if hasattr(content, 'text'):
                            response_text += content.text
        
        print("\n=== Final Response ===")
        print(response_text)
        
        print("\n=== Response Metadata ===")
        response_dict = {
            "id": final_response.id,
            "model": final_response.model,
            "status": final_response.status,
            "usage": {
                "input_tokens": final_response.usage.input_tokens,
                "output_tokens": final_response.usage.output_tokens,
                "total_tokens": final_response.usage.total_tokens,
            },
            "completed_at": final_response.completed_at,
        }
        pprint(response_dict)
        
        browser.close()
        return final_response


# Example usage - parameterized for easy reuse
# start_url = "https://williamlugoloobi.com"
# task = "Navigate to https://williamlugoloobi.com and find the current blog posts listed. Use the computer tool to explore the website and click on links as needed."
# start_url = "https://ntfy.sh/llms_know_difficulty"
# task = "Navigate to https://ntfy.sh/llms_know_difficulty and play a game with the person there. Try to win. you must return the hangam string with one filled letter per turn. the person will tell you whether it is correct or not"

# start_url = "https://www.nytimes.com/"
# task = "Navigate to https://www.nytimes.com/ and tell us the latest news. After that go to https://polymarket.com/ and search up corresponding bets running parallel to the news. Report on the most promising ones. cross check between the two sites to find out a prospective event."

start_url = "https://www.wikipedia.com/"
task = """You are browsing Wikipedia to answer a question.
Rules:

You may only use pages on wikipedia.org.
Use the browser to gather evidence before answering.
When finished, output the final answer within the answer tags e.g <answer>here</answer>.

Question: Which magazine was started first Arthur's Magazine or First for Women?"""



run_browser_task(start_url, task, debug=True)
