"""The evaluation script for char generation. """

import argparse
import torch

from conf.training_config import TrainConfig
from models.char_model import CharacterLevelModel
from utilities.char_encoder import CharTockenizer


def load_character_model(
    tockenizer: CharTockenizer,
    embeding_size: int,
    device: torch.device
) -> CharacterLevelModel:
    """
    Loads a CharacterLevelModel using the provided tokenizer, embedding size,
    and device.

    Args:
        tockenizer: The tokenizer used to determine vocabulary size.
        embeding_size: The size of the character embeddings.
        device: The device to load the model onto (CPU or CUDA).

    Returns:
        CharacterLevelModel: An instance of the CharacterLevelModel initialized
            with the given parameters.
    """

    model = CharacterLevelModel(
        vocabulary_size=tockenizer.vocab_size,
        embedding_size=embeding_size
    ).to(device=device)

    return model


def generate_text(
    model: torch.nn.Module,
    tokenizer: CharTockenizer,
    prompt: str,
    context_window: int, 
    max_new_tokens: int = 100,
    device: torch.device = torch.device("cpu")
) -> str:
    """
    Generates text using a character-level language model.

    Args:
        model: The trained CharacterLevelModel for text generation.
        tokenizer: The CharTockenizer used for encoding and decoding text.
        prompt: The initial string to prime the model.
        context_window: the block size 
        max_new_tokens: The maximum number of new characters to generate.
        device: The device to run the model on.

    Returns:
        The generated text as a string.
    """

    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = (
        torch.tensor(input_ids, dtype=torch.long)
        .unsqueeze(0)
        .to(device)
    )

    for _ in range(max_new_tokens):
        input_chunk = input_tensor[:, -context_window:]
        logits = model(input_chunk)
        probs = torch.softmax(logits[:, -1, :], dim=1)
        next_token_id = torch.argmax(probs, dim=1).unsqueeze(0)
        input_tensor = torch.cat([input_tensor, next_token_id], dim=1)

    generated_ids = input_tensor.squeeze().tolist()
    generated_text = tokenizer.decode(generated_ids)
    return generated_text


def parse_evaluation_arguments() -> argparse.Namespace:
    """Parses command-line arguments for evaluating a character-level model.

    Returns:
        argparse.Namespace: Parsed arguments containing evaluation parameters.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a character-level language model."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint file."
    )
    parser.add_argument(
        "--text_path",
        type=str,
        required=True,
        help="Path to the raw text file for tokenizer initialization."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Initial prompt string to prime the model for generation."
    )
    parser.add_argument(
        "--context_window",
        type=int,
        default=16,
        help="Number of previous characters to use as context for generation."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of new characters to generate."
    )
    return parser.parse_args()


def main() -> None:
    """Main function to evaluate a character-level text generation model."""

    # Parse command-line arguments
    args = parse_evaluation_arguments()
    # Select device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load raw text for tokenizer initialization
    with open(args.text_path, 'r', encoding='utf-8') as text_file:
        raw_text = text_file.read()

    # Initialize the character tokenizer
    tockenizer = CharTockenizer(text=raw_text)
    # Load the character-level model with the tokenizer and config
    # embedding size
    model = load_character_model(
        embeding_size=TrainConfig().embedding_size,
        device=device,
        tockenizer=tockenizer
    )
    # Load model weights from checkpoint
    model.load_state_dict(
        torch.load(
            f=args.checkpoint_path,
            map_location=device
        )
    )

    # Generate text using the model and tokenizer
    generated_text = generate_text(
        model=model,
        tokenizer=tockenizer, 
        prompt=args.prompt,
        context_window=args.context_window,
    )
    # Print the generated text
    print("\nGenerated Text\n")
    print(generated_text)


if __name__ == "__main_":
    main()
