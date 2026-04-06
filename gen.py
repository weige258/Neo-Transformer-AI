import re
import torch
import random
import logging
from typing import List, Tuple
from main import train, generation, model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_training_data(file_path: str) -> Tuple[List[str], List[str]]:
    """Load and parse training data from file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Remove extra newlines
        text = re.sub(pattern=r"\n+", repl="\n", string=text)

        # Extract human and assistant interactions
        human_pattern = r'<s>Human:(.*?)</s>'
        assistant_pattern = r'<s>Assistant:(.*?)</s>'

        human_queries = re.findall(human_pattern, text, re.DOTALL)
        assistant_responses = re.findall(assistant_pattern, text, re.DOTALL)

        # Clean up the data
        human_queries = [q.strip() for q in human_queries]
        assistant_responses = [r.strip() for r in assistant_responses]

        # Ensure we have matching pairs
        min_pairs = min(len(human_queries), len(assistant_responses))
        if min_pairs < len(human_queries) or min_pairs < len(assistant_responses):
            logging.warning(f"Data mismatch, using {min_pairs} pairs.")
            human_queries = human_queries[:min_pairs]
            assistant_responses = assistant_responses[:min_pairs]

        return human_queries, assistant_responses

    except Exception as e:
        logging.error(f"Failed to load training data: {e}")
        return [], []


def main() -> None:
    """Main training loop"""
    # Load training data
    human_queries, assistant_responses = load_training_data("train_sft.csv")

    if not human_queries or not assistant_responses:
        logging.error("No training data found, please check train_sft.csv")
        return

    data_count = len(human_queries)
    logging.info(f"Loaded {data_count} training pairs.")

   

    try:
        while True:
            # Select a random training pair
            i = random.randint(0, data_count - 1)
            ask = human_queries[i]
            answer = assistant_responses[i]

            try:
                print("\nask-------------\n")
                print(ask,flush=True)
                generation(ask)
            except Exception as e:
                logging.error(f"Training error: {e}")
                continue
    except :
        pass

if __name__ == "__main__":
    main()