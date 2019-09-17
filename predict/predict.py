import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

NEG_INF = -1e9


class Predictor(object):
    """Predictor class for a single model."""
    def __init__(self, model, device):

        self.model = model
        self.device = device

    def predict(self, loader):

        self.model.eval()

        probs = []
        gt = []
        with tqdm(total=len(loader.dataset)) as progress_bar:
            for data in loader:
                with torch.no_grad():
                    inputs, targets = data
                    batch_logits = self.model(inputs.to(self.device))

                    batch_probs = torch.sigmoid(batch_logits)

                probs.append(batch_probs)
                gt.append(targets)
                progress_bar.update(targets.size(0))
        probs = [p.cpu().numpy() for p in probs]
        probs_concat = np.concatenate(probs)
        gt_concat = np.concatenate(gt)
        
        probs_df = pd.DataFrame({"Pred": probs_concat.flatten()})
        gt_df = pd.DataFrame({"Label": gt_concat.flatten()})

        self.model.train()

        return probs_df, gt_df
