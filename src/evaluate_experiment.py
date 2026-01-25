import torch
import os
import glob
import re
from torch.utils.data import DataLoader
from data import DETRData
from model import DETR
from loss import DETRLoss, HungarianMatcher
from utils.boxes import stacker
from utils.setup import get_classes
from utils.logger import get_logger
from rich.console import Console
from rich.table import Table
from rich.progress import track

def get_epoch_from_path(path):
    """Extract epoch number from filename like 'scratch_10_model.pt' or '10_model.pt'"""
    match = re.search(r'(\d+)_model\.pt', path)
    if match:
        return int(match.group(1))
    return -1

def evaluate_checkpoints():
    logger = get_logger("evaluation")
    console = Console()
    
    # 1. Setup Data
    test_dataset = DETRData('data/test', train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=stacker, drop_last=False, num_workers=4)
    
    # 2. Setup Config
    classes = get_classes()
    num_classes = len(classes)
    
    weights = {'class_weighting': 1, 'bbox_weighting': 5, 'giou_weighting': 2}
    matcher = HungarianMatcher(weights)
    criterion = DETRLoss(num_classes=num_classes, matcher=matcher, weight_dict=weights, eos_coef=0.1)
    
    # 3. Find Checkpoints
    checkpoint_dir = 'experiment/checkpoints'
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
    
    # Sort checkponits: try to sort by epoch
    checkpoints.sort(key=get_epoch_from_path)
    
    results = []
    
    console.print(f"[bold blue]Found {len(checkpoints)} checkpoints. Starting evaluation...[/bold blue]")
    
    for checkpoint_path in track(checkpoints, description="Evaluating models..."):
        try:
            # Load Model
            model = DETR(num_classes=num_classes)
            # We use our strict/safe loading logic here but since these are from this experiment they should match perfectly
            state_dict = torch.load(checkpoint_path)
            model.load_state_dict(state_dict)
            model.eval()
            
            total_loss = 0.0
            batches = 0
            
            with torch.no_grad():
                for batch in test_dataloader:
                    X, y = batch
                    yhat = model(X)
                    loss_dict = criterion(yhat, y)
                    weight_dict = criterion.weight_dict
                    
                    losses = loss_dict['labels']['loss_ce']*weight_dict['class_weighting'] + \
                             loss_dict['boxes']['loss_bbox']*weight_dict['bbox_weighting'] + \
                             loss_dict['boxes']['loss_giou']*weight_dict['giou_weighting']
                    
                    total_loss += losses.item()
                    batches += 1
            
            avg_loss = total_loss / batches if batches > 0 else float('inf')
            
            results.append({
                "Checkpoint": os.path.basename(checkpoint_path),
                "Loss": avg_loss,
                "Epoch": get_epoch_from_path(checkpoint_path)
            })
            
        except Exception as e:
            logger.error(f"Failed to evaluate {checkpoint_path}: {e}")
            results.append({
                "Checkpoint": os.path.basename(checkpoint_path),
                "Loss": "ERROR",
                "Epoch": get_epoch_from_path(checkpoint_path)
            })

    # 4. Display Results
    table = Table(title="🧪 Experiment Results")
    table.add_column("Epoch", justify="right", style="cyan", no_wrap=True)
    table.add_column("Checkpoint File", style="magenta")
    table.add_column("Test Loss", justify="right", style="green")

    # Sort results by Loss (best first) or Epoch? Let's do Epoch, but highlight best.
    # Actually user wants to see "test all", usually to find the best.
    
    # Let's find best loss
    best_loss = float('inf')
    for r in results:
        if isinstance(r["Loss"], float) and r["Loss"] < best_loss:
            best_loss = r["Loss"]

    for r in results:
        loss_display = f"{r['Loss']:.5f}" if isinstance(r['Loss'], float) else str(r['Loss'])
        if isinstance(r['Loss'], float) and r['Loss'] == best_loss:
             loss_display = f"[bold green]{loss_display} (BEST)[/bold green]"
        
        table.add_row(str(r['Epoch']), r['Checkpoint'], loss_display)

    console.print(table)

if __name__ == '__main__':
    evaluate_checkpoints()
