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

    training_rounds = 0
    save_interval = 300

    try:
        while True:
            # Select a random training pair
            i = random.randint(0, data_count - 1)
            ask = human_queries[i]
            answer = assistant_responses[i]

            # Skip empty queries or responses
            if not ask or not answer:
                continue

            # Train on this pair
            try:
                train(ask, answer)

                # Generate a response for evaluation
                generation(ask)

                training_rounds += 1
                print("*" * 100)

                # Save model periodically
                if training_rounds % save_interval == 0:
                    torch.save(obj=model.state_dict(), f="model.pth")
                    logging.info(f"Model saved, training rounds: {training_rounds}")

            except Exception as e:
                logging.error(f"Training error: {e}")
                continue

    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
        # Save final model
        torch.save(obj=model.state_dict(), f="model.pth")
        logging.info(f"Final model saved, training rounds: {training_rounds}")
    except Exception as e:
        logging.error(f"Training loop error: {e}")
        # Save model before exiting
        torch.save(obj=model.state_dict(), f="model.pth")


if __name__ == "__main__":
    main()
