from sandbox_browser import run_browser_worker

payload = {
    "start_url": "https://www.wikipedia.org/",
    "task": (
        "You are browsing Wikipedia to answer a question. "
        "You may only use pages on wikipedia.org. "
        "Use the browser to gather evidence before answering."
        "When finished, output the final answer within <answer>ANSWER HERE</answer> tags. "
        "Question: When was Jack Livesey's father born?"
        # "Question: When did the director of film Morchha die?"
        # "Question: Which country the director of film One Law For The Woman is from? "
        # "Question: Where did the performer of song White Noise (Linkin Park Song) die?"
    ),
    "debug": True,
}

result = run_browser_worker(payload=payload, worker_script="browser_worker.py")
print(result)
