"""
Parallel evaluation script for collecting browser traces from models on 2WikiMultihopQA.

Usage:
    python run_eval_trace.py --split test --samples 50 --workers 4 --seed 42
    python run_eval_trace.py --split train --samples -1  # all examples
    python run_eval_trace.py --split valid --workers 1 --debug  # sequential + verbose
"""

import argparse
import json
import logging
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm
from sandbox_browser import run_browser_worker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"eval_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

BROWSING_TEMPLATE = """You are browsing Wikipedia to answer a question.
Rules:

You may only use pages on wikipedia.org.
Use the browser to gather evidence before answering.
When finished, output the final answer within the answer tags e.g <answer>ANSWER HERE</answer>.

Question: {question}"""

ARTIFACTS_BASE_DIR = Path("artifacts")
DATASET_NAME = "2WikiMultihopQA"


class EvalRun:
    """Manages directory structure for a single evaluation run."""
    
    def __init__(self, split: str, start_timestamp: str):
        """
        Initialize eval run with organized directory structure.
        
        Structure:
            artifacts/{dataset}/{split}/{start_timestamp}/
            ├── traces/
            ├── results.jsonl
            ├── stats.json
            └── eval.log
        """
        self.split = split
        self.start_timestamp = start_timestamp
        
        # Main run directory
        self.run_dir = ARTIFACTS_BASE_DIR / DATASET_NAME / split / start_timestamp
        
        # Subdirectories
        self.traces_dir = self.run_dir / "traces"
        
        # Setup
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(exist_ok=True)
    
    def get_trace_directory(self, example_idx: int, model_id: Optional[str] = None) -> Path:
        """
        Get trace directory for a single example.
        
        Structure: traces/{example_idx:06d}_{model_id}/
        Example: traces/000015_gpt4/
        """
        model_suffix = f"_{model_id}" if model_id else ""
        trace_name = f"{example_idx:06d}{model_suffix}"
        trace_dir = self.traces_dir / trace_name
        trace_dir.mkdir(exist_ok=True)
        return trace_dir
    
    def get_results_file(self) -> Path:
        """Get path to aggregated results.jsonl"""
        return self.run_dir / "results.jsonl"
    
    def get_stats_file(self) -> Path:
        """Get path to stats.json"""
        return self.run_dir / "stats.json"
    
    def get_log_file(self) -> Path:
        """Get path to eval.log"""
        return self.run_dir / "eval.log"


def save_trace_metadata(
    trace_dir: Path,
    example_idx: int,
    question: str,
    ground_truth: str,
    model_id: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save metadata.json inside trace directory."""
    metadata = {
        "example_idx": example_idx,
        "question": question,
        "ground_truth": ground_truth,
        "model_id": model_id,
        "timestamp": datetime.now().isoformat(),
    }
    
    if result:
        metadata.update({
            "success": result.get("success"),
            "answer": result.get("answer"),
            "error": result.get("error"),
            "attempts": result.get("attempts"),
        })
    
    metadata_file = trace_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_file


def load_and_sample_dataset(
    split: str = "test",
    samples: Optional[int] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Load dataset and optionally sample from it.
    
    Args:
        split: Dataset split ("train", "test", or "validation")
        samples: Number of samples to take (-1 for all, None for all)
        seed: Random seed for reproducibility
        
    Returns:
        List of dataset examples
    """
    logger.info(f"Loading 2WikiMultihopQA dataset (split={split})...")
    ds = load_dataset("framolfese/2WikiMultihopQA")
    
    if split not in ds:
        available = list(ds.keys())
        raise ValueError(f"Split '{split}' not found. Available: {available}")
    
    examples = list(ds[split])
    logger.info(f"Dataset loaded: {len(examples)} examples in '{split}' split")
    
    # Determine sample size
    if samples is None or samples == -1:
        sample_size = len(examples)
    else:
        sample_size = min(samples, len(examples))
    
    # Random seed for reproducibility
    random.seed(seed)
    logger.info(f"Sampling {sample_size} examples with seed={seed}")
    
    sampled = random.sample(examples, k=sample_size)
    logger.info(f"Sampled {len(sampled)} examples")
    
    return sampled


def run_single_trace(
    example_idx: int,
    example: Dict[str, Any],
    eval_run: 'EvalRun',
    model_id: Optional[str] = None,
    debug: bool = False,
    worker_script: str = "browser_worker.py",
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Run a single trace evaluation (worker function).
    
    Args:
        example_idx: Index of this example in dataset
        example: Dataset example
        eval_run: EvalRun object managing directories
        model_id: Model identifier (e.g., 'gpt4', 'claude')
        debug: Enable debug mode
        worker_script: Worker script path
        max_retries: Number of retries on failure
        
    Returns:
        Result dict with metadata, answer, response, and artifacts
    """
    question = example.get("question", "")
    ground_truth = example.get("answer", "")
    
    payload = {
        "start_url": "https://www.wikipedia.org/",
        "task": BROWSING_TEMPLATE.format(question=question),
        "debug": debug,
    }
    
    # Create trace directory for this example
    trace_dir = eval_run.get_trace_directory(example_idx, model_id)
    
    # Convert to relative path for sandbox
    trace_dir_relative = str(trace_dir)
    
    for attempt in range(max_retries):
        try:
            result = run_browser_worker(
                payload=payload,
                worker_script=worker_script,
                target_artifacts_dir=trace_dir_relative,  # Pass as relative path string
            )
            
            # Augment result with metadata
            result["example_idx"] = example_idx
            result["question"] = question
            result["ground_truth"] = ground_truth
            result["model_id"] = model_id
            result["split"] = eval_run.split
            result["success"] = True
            result["attempts"] = attempt + 1
            result["trace_dir"] = str(trace_dir)
            
            # Save metadata.json inside trace directory
            save_trace_metadata(trace_dir, example_idx, question, ground_truth, model_id, result)
            
            return result
            
        except Exception as e:
            if attempt == max_retries - 1:
                result = {
                    "example_idx": example_idx,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_id": model_id,
                    "split": eval_run.split,
                    "success": False,
                    "error": str(e),
                    "attempts": max_retries,
                    "trace_dir": str(trace_dir),
                }
                # Save metadata for failed traces too
                save_trace_metadata(trace_dir, example_idx, question, ground_truth, model_id, result)
                return result
            time.sleep(2 ** attempt)  # Exponential backoff


def run_evaluation(
    split: str = "test",
    samples: Optional[int] = None,
    num_workers: int = 4,
    seed: int = 42,
    model_id: Optional[str] = None,
    debug: bool = False,
    worker_script: str = "browser_worker.py",
) -> Dict[str, Any]:
    """
    Run full evaluation with parallel workers.
    
    Directory structure created:
        artifacts/{dataset}/{split}/{start_timestamp}/
        ├── traces/
        │   ├── 000001_model/
        │   │   ├── trace.jsonl
        │   │   ├── browser_payload.json
        │   │   └── metadata.json
        │   ├── 000002_model/
        │   └── ...
        ├── results.jsonl
        ├── stats.json
        └── eval.log
    
    Args:
        split: Dataset split ("train", "test", "validation")
        samples: Number of samples (-1 or None for all)
        num_workers: Number of parallel workers
        seed: Random seed
        model_id: Model identifier (e.g., 'gpt4', 'claude')
        debug: Debug mode
        worker_script: Worker script path
        
    Returns:
        Summary stats and results
    """
    # Create eval run directory structure
    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_run = EvalRun(split, start_timestamp)
    
    # Setup logging to both console and eval.log
    log_file = eval_run.get_log_file()
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    
    # Load data
    examples = load_and_sample_dataset(split=split, samples=samples, seed=seed)
    total_examples = len(examples)
    
    logger.info("=" * 70)
    logger.info("EVALUATION STARTED")
    logger.info("=" * 70)
    logger.info(f"Run directory: {eval_run.run_dir}")
    logger.info(f"Split: {split}, Samples: {total_examples}, Workers: {num_workers}, Seed: {seed}")
    logger.info(f"Model: {model_id or 'unspecified'}")
    logger.info("=" * 70)
    
    results = []
    failed_indices = []
    start_time = time.time()
    
    if num_workers == 1:
        # Sequential mode with tqdm
        logger.info("Running in sequential mode (1 worker)")
        for idx, example in tqdm(enumerate(examples, 1), total=total_examples, desc="Collecting traces", ncols=100):
            result = run_single_trace(idx, example, eval_run, model_id=model_id, debug=debug, worker_script=worker_script)
            results.append(result)
            if not result.get("success"):
                failed_indices.append(idx)
    else:
        # Parallel mode with tqdm
        logger.info(f"Running in parallel mode ({num_workers} workers)")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(run_single_trace, idx, example, eval_run, model_id=model_id, debug=debug, worker_script=worker_script): idx
                for idx, example in enumerate(examples, 1)
            }
            
            # Use tqdm to track completion
            for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting traces", ncols=100):
                result = future.result()
                results.append(result)
                
                if not result.get("success"):
                    failed_indices.append(result.get("example_idx"))
    
    total_time = time.time() - start_time
    
    # Compute stats
    successful = sum(1 for r in results if r.get("success"))
    failed = total_examples - successful
    
    stats = {
        "timestamp": datetime.now().isoformat(),
        "run_timestamp": start_timestamp,
        "split": split,
        "model_id": model_id,
        "total_examples": total_examples,
        "successful": successful,
        "failed": failed,
        "success_rate": successful / total_examples if total_examples > 0 else 0,
        "total_time_seconds": total_time,
        "avg_time_per_example_seconds": total_time / total_examples if total_examples > 0 else 0,
        "num_workers": num_workers,
        "seed": seed,
        "run_dir": str(eval_run.run_dir),
    }
    
    # Save results to run directory
    results_file = eval_run.get_results_file()
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    logger.info(f"Results saved: {results_file}")
    
    # Save stats to run directory
    stats_file = eval_run.get_stats_file()
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats saved: {stats_file}")
    
    # Print summary
    logger.info("=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Split: {split}")
    logger.info(f"Examples: {total_examples}")
    logger.info(f"Successful: {successful} ({100*stats['success_rate']:.1f}%)")
    logger.info(f"Failed: {failed}")
    if failed_indices:
        logger.info(f"Failed indices: {failed_indices[:20]}{'...' if len(failed_indices) > 20 else ''}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    logger.info(f"Avg per example: {stats['avg_time_per_example_seconds']:.1f}s")
    logger.info(f"Workers: {num_workers}")
    logger.info(f"Run directory: {eval_run.run_dir}")
    logger.info("=" * 70)
    
    return {
        "stats": stats,
        "results": results,
        "run_dir": str(eval_run.run_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect browser traces from models on 2WikiMultihopQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 10 examples, parallel workers
  python run_eval_trace.py --split test --samples 10 --workers 4
  
  # GPT-4 traces (15 examples, reproducible seed)
  python run_eval_trace.py --split test --samples 15 --model gpt4 --seed 42
  
  # Claude traces (15 examples, same seed for same questions)
  python run_eval_trace.py --split test --samples 15 --model claude --seed 42
  
  # Full train set, sequential (debugging)
  python run_eval_trace.py --split train --samples -1 --workers 1 --debug
  
  # Validation set with seed for reproducibility
  python run_eval_trace.py --split validation --samples 50 --seed 123 --model llama

Directory Structure:
  artifacts/2WikiMultihopQA/{split}/{eval_start_timestamp}/
  ├── traces/
  │   ├── 000001_gpt4/
  │   │   ├── trace.jsonl
  │   │   ├── browser_payload.json
  │   │   └── metadata.json
  │   ├── 000002_gpt4/
  │   └── ...
  ├── results.jsonl
  ├── stats.json
  └── eval.log

Progress:
  Watch traces appear in real-time:
  $ watch -n 1 'ls -1 artifacts/2WikiMultihopQA/test/{timestamp}/traces/ | wc -l'
  
  Or tail the log:
  $ tail -f artifacts/2WikiMultihopQA/test/{timestamp}/eval.log
        """
    )
    
    parser.add_argument(
        "--split",
        choices=["train", "test", "validation"],
        default="test",
        help="Dataset split to use (default: test)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to use (-1 or None for all, default: all)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (1 for sequential, default: 4)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID (e.g., 'gpt4', 'claude', 'llama') for directory organization"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (verbose output)"
    )
    parser.add_argument(
        "--worker-script",
        default="browser_worker.py",
        help="Path to worker script (default: browser_worker.py)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Configuration: split={args.split}, samples={args.samples}, "
               f"workers={args.workers}, seed={args.seed}, model={args.model}")
    
    run_evaluation(
        split=args.split,
        samples=args.samples,
        num_workers=args.workers,
        seed=args.seed,
        model_id=args.model,
        debug=args.debug,
        worker_script=args.worker_script,
    )


if __name__ == "__main__":
    main()
