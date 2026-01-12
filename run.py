#!/usr/bin/env python3
# filename: run_experiments.py

import os
import subprocess
import argparse
import time
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Global reproducibility helper
try:
    from seed_utils import set_global_seed
except Exception:
    set_global_seed = None  # Fallback if module unavailable

def run_experiment(experiment_name, strategy, output_dir, num_points=None, debug_expanded_masks=False, **kwargs):
    """
    Run a single experiment with auto_labeler.py
    
    Args:
        experiment_name: Name of the experiment
        strategy: Point selection strategy to use
        output_dir: Base output directory
        num_points: Number of points to process (optional for 'list' strategy)
        kwargs: Additional arguments to pass to auto_labeler.py
    
    Returns:
        Success status and experiment path
    """
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(output_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a copy of parameters for processing
    params = kwargs.copy()
    params.update({
        "strategy": strategy,
        "output_dir": str(exp_dir)
    })
    # We'll append the debug flag explicitly (hyphenated) when building the cmd
    
    # Only add num_points if provided and not using list strategy
    if num_points is not None and strategy != "list":
        params["num_points"] = num_points
    
    # Remove keys that are not CLI arguments for auto_labeler.py
    experiment_name = params.pop("name", experiment_name)
    images = params.pop("images")
    ground_truth = params.pop("ground_truth", None)  # Optional
    
    # Handle points_file for list strategy
    points_file = params.pop("points_file", None)
    
    # Build the command
    cmd = ["python", "auto_labeler.py", "--images", images]
    
    # Add ground_truth if present
    if ground_truth:
        cmd.extend(["--ground-truth", ground_truth])
    
    # Add points_file if using list strategy
    if points_file:
        cmd.extend(["--points-file", points_file])
    
    # Add color_dict if present
    color_dict = params.pop("color_dict", None)
    if color_dict:
        cmd.extend(["--color-dict", color_dict])
    
    # Build/merge strategy kwargs: allow passing lambda_balance (and other strategy args) directly
    # so run.py users don't need to hand-write JSON.
    strat_kwargs_raw = params.pop("strategy_kwargs", None)
    # Start from provided strategy_kwargs (dict or JSON string), else empty dict
    if isinstance(strat_kwargs_raw, str):
        try:
            strategy_kwargs = json.loads(strat_kwargs_raw) if strat_kwargs_raw.strip() else {}
        except Exception:
            strategy_kwargs = {}
    elif isinstance(strat_kwargs_raw, dict):
        strategy_kwargs = dict(strat_kwargs_raw)
    else:
        strategy_kwargs = {}

    # Pull known strategy-specific fields from params into strategy_kwargs
    # Currently requested: lambda_balance (used by DynamicPoints strategies)
    lb = params.pop("lambda_balance", None)
    if lb is not None:
        strategy_kwargs["lambda_balance"] = lb

    hf = params.pop("heatmap_fraction", None)
    if hf is not None:
        strategy_kwargs["heatmap_fraction"] = hf

    if strategy_kwargs:
        params["strategy_kwargs"] = json.dumps(strategy_kwargs)

    # Add remaining parameters
    for key, value in params.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key.replace('_','-')}")
            else:
                if key == "output_dir":
                    cmd.extend(["--output", str(value)])
                elif key == "color_dict":
                    cmd.extend(["--color-dict", str(value)])
                else:
                    arg_name = key.replace("_", "-")
                    cmd.extend([f"--{arg_name}", str(value)])

    # Explicitly append debug flag if requested
    if debug_expanded_masks:
        cmd.append("--debug-expanded-masks")
    
    # Save experiment configuration
    config = {
        "experiment_name": experiment_name,
        "strategy": strategy,
        "timestamp": timestamp,
        **kwargs
    }
    if debug_expanded_masks:
        config["debug_expanded_masks"] = True
    # Only include num_points in config if it was provided
    if num_points is not None:
        config["num_points"] = num_points
        
    with open(exp_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Run the experiment
    print(f"\n{'='*50}")
    print(f"Starting experiment: {experiment_name}")
    print(f"Strategy: {strategy}")
    if num_points is not None:
        print(f"Points: {num_points}")
    else:
        print(f"Points: determined by annotations file")
    print(f"Output directory: {exp_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    try:
        # Run with proper terminal emulation for tqdm
        log_path = exp_dir / "experiment.log"
        
        # Method 1: Try using pty-based approach
        import pty
        import select
        import fcntl
        import termios
        import struct
        
        def run_with_pty():
            # Get terminal size
            rows, cols = struct.unpack('hh', fcntl.ioctl(0, termios.TIOCGWINSZ, '1234'))
            
            # Create a pseudo-terminal
            master, slave = pty.openpty()
            
            # Set the window size of the slave to match our terminal
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('hh', rows, cols))
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                close_fds=True
            )
            
            # Close the slave end (child will have it)
            os.close(slave)
            
            # Read from master and write to both stdout and log file
            with open(log_path, 'w') as log_file:
                while True:
                    try:
                        ready, _, _ = select.select([master], [], [], 0.1)
                        if ready:
                            data = os.read(master, 1024).decode('utf-8', errors='ignore')
                            if not data:
                                break
                            print(data, end='', flush=True)
                            log_file.write(data)
                            log_file.flush()
                        
                        # Check if process has finished
                        if process.poll() is not None:
                            # Read any remaining data
                            try:
                                while True:
                                    data = os.read(master, 1024).decode('utf-8', errors='ignore')
                                    if not data:
                                        break
                                    print(data, end='', flush=True)
                                    log_file.write(data)
                                    log_file.flush()
                            except OSError:
                                pass
                            break
                            
                    except OSError:
                        break
            
            # Clean up
            os.close(master)
            return process.wait()
        
        return_code = run_with_pty()
        
        duration = time.time() - start_time
        print(f"\nExperiment completed in {duration:.1f} seconds")
        
        # Save completion status
        with open(exp_dir / "experiment_status.json", "w") as f:
            json.dump({
                "success": return_code == 0,
                "return_code": return_code,
                "duration": duration
            }, f, indent=2)
        
        return return_code == 0, exp_dir
    
    except Exception as e:
        print(f"Error running experiment: {e}")
        with open(exp_dir / "experiment_status.json", "w") as f:
            json.dump({
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time
            }, f, indent=2)
        return False, exp_dir

def collect_results(experiment_dirs):
    """
    Collect and compare results from multiple experiments
    
    Args:
        experiment_dirs: List of experiment directories to analyze
    """
    results = []
    
    for exp_dir in experiment_dirs:
        try:
            # Load experiment config
            with open(exp_dir / "experiment_config.json", "r") as f:
                config = json.load(f)
            
            # Load metrics
            metrics_path = exp_dir / "stats" / "segmentation_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)

                # Extract key metrics
                results.append({
                    "experiment_name": config.get("experiment_name", "Unknown"),
                    "strategy": config.get("strategy", "Unknown"),
                    "num_points": config.get("num_points", 0),
                    # Replace global_miou column with macro pixel accuracy (global_mpa)
                    "mPA": metrics.get("global_mpa", 0),
                    "mIoU": metrics.get("global_miou", 0),
                    "num_classes": metrics.get("num_classes", 0),
                    "num_images": metrics.get("num_images_evaluated", 0)
                })
        except Exception as e:
            print(f"Error processing experiment {exp_dir}: {e}")
    
    # Create summary dataframe
    if results:
        df = pd.DataFrame(results)
        print("\n===== EXPERIMENT RESULTS =====")
        print(df.to_string(index=False))
        
    # Plot generation removed per user request (global_miou not needed, mpa shown instead)

def main():
    parser = argparse.ArgumentParser(description='Run experiments with auto_labeler.py')
    parser.add_argument('--output-dir', default='experiments', 
                        help='Base directory for experiment outputs')
    parser.add_argument('--seed', type=int, default=None, help='Global random seed for reproducibility (default: None, non-deterministic)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation (cpu or cuda, default: cpu)')
    args = parser.parse_args()
    
    # Only set global seed if user provides --seed
    if set_global_seed is not None and args.seed is not None:
        try:
            set_global_seed(args.seed, deterministic=True, verbose=True)
        except Exception as e:
            print(f"[run.py] Warning: could not set global seed ({e})")

    # Create experiments directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Define experiments to run
    experiments = [

        {
            "name": "demo",
            "strategy": "dynamicPoints",
            "num_points": 25,
            "images": "demo/images/",
            "ground_truth": "demo/labels",
            "default-background-class-id": 34,
            "color_dict": "demo/color_dict.json" # For evaluation
        },

    ]
    
    # Common parameters for all experiments
    common_params = {
        "device": args.device,
        "visualization": False,  # Set to True if you want sparse point visualizations
        "maskSLIC": False,   # true is false and false is true
    }
    
    # Run experiments
    completed_experiments = []
    for exp in experiments:
        # Merge common params with experiment-specific params
        params = {**common_params, **exp}
        
        # Remove keys that are not CLI arguments for auto_labeler.py
        experiment_name = params.pop("name")
        images = params.pop("images")
        ground_truth = params.pop("ground_truth", None)  # Optional
        
        # Handle points_file for list strategy
        points_file = params.pop("points_file", None)
        
        # Get strategy and num_points, making num_points optional for list strategy
        strategy = params.pop("strategy")
        num_points = params.pop("num_points", None)
        
        # For non-list strategies, num_points is required
        if strategy != "list" and num_points is None:
            raise ValueError(f"num_points is required for strategy '{strategy}'")
        
        success, exp_dir = run_experiment(
            experiment_name=experiment_name,
            strategy=strategy,
            output_dir=output_dir,
            num_points=num_points,
            debug_expanded_masks=False,  # Enable debug per user request
            images=images,
            ground_truth=ground_truth,
            points_file=points_file,
            **params
        )
        
        if success:
            completed_experiments.append(exp_dir)
    
    # Collect and compare results
    if completed_experiments:
        collect_results(completed_experiments)
        print(f"\nAll experiments completed! Results in: {output_dir}")
    else:
        print("\nNo experiments were successfully completed.")

if __name__ == "__main__":
    main()
