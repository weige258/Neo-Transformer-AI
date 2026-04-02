from main import train, generation


while True:
    train("What is the capital of France?", "The capital of France is Paris.")
    generation("What is the capital of Fr?")
    train("你是什么ai","我是 deepseek")
    generation("你是什么ai")
