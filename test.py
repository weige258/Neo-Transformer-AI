from main import generation, train


for i in range(20):
    train(None,"你是什么ai 我是 deepseek")
    train("你是什么样子的ai", "我是一个大模型 我叫deepseek")
    generation("你是什么ai吗",max_generate_tokens=30)
