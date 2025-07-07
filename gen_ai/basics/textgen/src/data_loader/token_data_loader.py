"""Data loader for token-level tasks."""

from conf.training_config import TokenDataLoaderConfig

class TockenDataLoader:
    """Data loader for token-level tasks."""

    def __init__(self, config: TokenDataLoaderConfig) -> None:
        """Initialization routine.

        Args:
            config: Configuration for the token data loader.
        """

        self.config = config

    def load_data(self) -> list[str]:
        """Load data based on the configuration."""

        texts: list[str] = []

        with open(self.config.text_path, "r") as file:
            for line in file:
                line = line.strip()
                if line:
                    if self.config.pre_process_fn:
                        line = self.config.pre_process_fn(line)
                    texts.append(line)
        return texts
        
    def window_texts(self, texts: list[str]) -> list[list[str]]:
        """Create context windows from the loaded texts.

        Args:
            texts: List of text strings.

        Returns:
            List of lists containing context windows.
        """

        if not self.config.context_window:
            return texts
        context_windows = []
        for text in texts:
            # Split the text into tokens
            tokens = text.split()

            for i in range(len(tokens) - self.config.context_window + 1):
                if len(tokens[i:i + self.config.context_window]) <= self.config.context_window:
                    # Create a context window of the specified size
                    context_windows.append(tokens[i])
                else:
                    context_windows.append(tokens[i:i + self.config.context_window])

        return context_windows
    
    def create_input_target_pairs(
        self, context_windows: list[list[str]]
    ) -> list[tuple[list[str], str]]:
        """Create input-target pairs from context windows.

        Args:
            context_windows: List of context windows.

        Returns:
            List of tuples containing input and target pairs.
        """

        input_target_pairs = []
        for window in context_windows:
            if len(window) > 1:
                input_tokens = window[:-1]
                target_token = window[-1]
                input_target_pairs.append((input_tokens, target_token))
        return input_target_pairs
    
    def __call__(self,) -> list[tuple[list[str], str]]:
        """Load data, create context windows, and generate input-target pairs.
        
        Returns:
            List of tuples containing input and target pairs.
        """

        texts = self.load_data()
        context_windows = self.window_texts(texts)
        return self.create_input_target_pairs(context_windows)