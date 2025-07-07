import torch
from torch.utils.data import Dataset, DataLoader

class TextPairDataset(Dataset):
    """Dataset for text paiesr input and target pairs."""

    def __init__(self,
                 pairs: list[tuple[str, str]],
                 tokenizer: torch.nn.Module,) -> None:
        """Initialization routine.
        
        Args:
            pairs: List of tuples containing input and target text pairs.
            tokenizer: Tokenizer to convert text to token ids.
        """
        super().__init__()
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Returns the number of pairs in the dataset."""

        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the input and target tensors for a given index.
        
        Args:
            idx: Index of the pair to retrieve.
        
        Returns:
            A tuple containing input and target tensors.
        """
        
        input_text, target_text = self.pairs[idx]
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').squeeze(0)
        target_ids = self.tokenizer.encode(target_text, return_tensors='pt').squeeze(0)
        return input_ids, target_ids