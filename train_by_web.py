import torch
import random
import logging
from main import train, generation, model
from crawler import WebCrawler

crawler = WebCrawler()


training_rounds = 0
save_interval = 300

while True:
    try:
        text = crawler.get()
        if text != None:
            train(text)

            generation(text[0:random.randint(1, len(text))])
        
        training_rounds += 1
        print("*" * 100)

        # Save model periodically
        if training_rounds % save_interval == 0:
            torch.save(obj=model.state_dict(), f="model.pth")
            logging.info(f"Model saved, training rounds: {training_rounds}")
    except Exception as e:
        logging.error(f"Web crawling error: {e}")
        continue