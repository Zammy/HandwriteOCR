import json
import time
from pathlib import Path
from typing import List, Callable
from datetime import datetime

from .training_pipeline import TrainingPipeline
from .experiment_config import ExperimentConfig


class ExperimentQueue:
    """Manages a queue of experiments to run sequentially.
    
    Features:
    - Add experiments programmatically
    - Save/load queue state
    - Track completed/failed experiments
    - Resume interrupted queues
    - Progress reporting
    """
    
    def __init__(self, queue_name: str = "experiments", base_dir: str = "_experiments"):
        self.queue_name = queue_name
        self.base_dir = Path(base_dir)
        self.queue_dir = self.base_dir / queue_name
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        
        self.queue_file = self.queue_dir / "queue.json"
        self.state_file = self.queue_dir / "state.json"
        
        self.experiments: List[ExperimentConfig] = []
        self.state = {
            "completed": [],
            "failed": [],
            "current_index": 0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Load existing state if available
        if self.state_file.exists():
            self._load_state()
    
    def add_experiment(self, config: ExperimentConfig) -> None:
        """Add a single experiment to the queue."""
        self.experiments.append(config)
    
    def add_experiments(self, configs: List[ExperimentConfig]) -> None:
        """Add multiple experiments to the queue."""
        self.experiments.extend(configs)
        self.save_queue()
    
    def add_from_file(self, filepath: str) -> None:
        """Add experiment from JSON config file."""
        config = ExperimentConfig.load(filepath)
        self.add_experiment(config)
    
    def get_pending_experiments(self) -> List[ExperimentConfig]:
        """Get list of experiments that haven't been run yet."""
        return self.experiments[self.state["current_index"]:]
    
    def get_progress(self) -> dict:
        """Get current progress statistics."""
        total = len(self.experiments)
        completed = len(self.state["completed"])
        failed = len(self.state["failed"])
        pending = total - self.state["current_index"]
        
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "current_index": self.state["current_index"],
            "completion_rate": f"{(completed/total*100):.1f}%" if total > 0 else "0%"
        }
    
    def print_status(self) -> None:
        """Print current queue status."""
        progress = self.get_progress()
        print("\n" + "="*60)
        print(f"Experiment Queue: {self.queue_name}")
        print("="*60)
        print(f"Total experiments: {progress['total']}")
        print(f"Completed: {progress['completed']}")
        print(f"Failed: {progress['failed']}")
        print(f"Pending: {progress['pending']}")
        print(f"Progress: {progress['completion_rate']}")
        print("="*60 + "\n")
    
    def run_all(self, train_function: Callable[[ExperimentConfig, str], dict]) -> None:
        """Run all pending experiments in the queue.
        
        Args:
            train_function: Function that takes ExperimentConfig and tracker_base_dir and returns results dict
        """
        pending = self.get_pending_experiments()
        
        if not pending:
            print("No pending experiments in queue.")
            return
        
        self.print_status()
        
        for i, config in enumerate(pending, start=1):
            current_global_idx = self.state["current_index"]
            
            print(f"\n{'='*60}")
            print(f"Running experiment {i}/{len(pending)}")
            print(f"Global index: {current_global_idx + 1}/{len(self.experiments)}")
            print(f"Name: {config.experiment_name or 'unnamed'}")
            print(f"{'='*60}\n")
            
            try:
                start_time = time.time()
                results = train_function(config, str(self.queue_dir))
                elapsed = time.time() - start_time
                
                # Record success
                self.state["completed"].append({
                    "index": current_global_idx,
                    "name": config.experiment_name,
                    "elapsed_seconds": elapsed,
                    "completed_at": datetime.now().isoformat(),
                    "results": results
                })
                
                print(f"\n✓ Experiment completed in {elapsed/60:.2f} minutes")
                
            except Exception as e:
                # Record failure
                self.state["failed"].append({
                    "index": current_global_idx,
                    "name": config.experiment_name,
                    "error": str(e),
                    "failed_at": datetime.now().isoformat()
                })
                
                print(f"\n✗ Experiment failed: {str(e)}")
                print("Continuing with next experiment...")
            
            # Update state
            self.state["current_index"] += 1
            self.state["updated_at"] = datetime.now().isoformat()
            self._save_state()
        
        # Final summary
        print("\n" + "="*60)
        print("QUEUE COMPLETED")
        print("="*60)
        self.print_status()
        self._print_summary()
    
    def _print_summary(self) -> None:
        """Print summary of completed and failed experiments."""
        if self.state["completed"]:
            print("\nCompleted experiments:")
            for exp in self.state["completed"]:
                print(f"  ✓ {exp['name']} ({exp['elapsed_seconds']/60:.2f} min)")
        
        if self.state["failed"]:
            print("\nFailed experiments:")
            for exp in self.state["failed"]:
                print(f"  ✗ {exp['name']} - Error: {exp['error']}")
    
    def save_queue(self) -> None:
        """Save experiment queue to disk."""
        queue_data = {
            "queue_name": self.queue_name,
            "experiments": [exp.to_dict() for exp in self.experiments]
        }
        with open(self.queue_file, 'w') as f:
            json.dump(queue_data, f, indent=2)
    
    def _save_state(self) -> None:
        """Save queue state to disk."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def _load_state(self) -> None:
        """Load queue state from disk."""
        with open(self.state_file, 'r') as f:
            self.state = json.load(f)
        
        # Load experiments
        if self.queue_file.exists():
            with open(self.queue_file, 'r') as f:
                queue_data = json.load(f)
                self.experiments = [
                    ExperimentConfig.from_dict(exp_dict) 
                    for exp_dict in queue_data["experiments"]
                ]
    
    def export_configs(self, output_dir: str) -> None:
        """Export all experiment configs as individual JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, config in enumerate(self.experiments):
            name = config.experiment_name or f"experiment_{i:03d}"
            filepath = output_path / f"{name}.json"
            config.save(str(filepath))
        
        print(f"Exported {len(self.experiments)} configs to {output_dir}")


def create_example_queue(name) -> ExperimentQueue:
    """Create an example queue with various experiments.
    
    This demonstrates how to programmatically create experiment queues.
    """
    from .experiment_config import (
        get_baseline_config,
        get_heavy_augmentation_config,
        get_large_model_config,
        get_learning_rate_search_configs,
        get_batch_size_search_configs
    )
    
    queue = ExperimentQueue(queue_name=name)
    
    # Add baseline
    queue.add_experiment(get_baseline_config())
    
    # # Add augmentation experiments
    # queue.add_experiment(get_heavy_augmentation_config())
    
    # # # Add model size experiment
    # queue.add_experiment(get_large_model_config())
    
    # # # Add learning rate search
    # queue.add_experiments(get_learning_rate_search_configs())

    # queue.add_experiments(get_batch_size_search_configs())
    
    return queue


if __name__ == "__main__":
    # Example usage
    queue = create_example_queue("test3")
    queue.print_status()
    queue.run_all(train_function=TrainingPipeline.train_experiment)