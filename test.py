from main import generation, train


for i in range(50):
    train(None,None,"你是什么ai 我是deepseek ai",None)
    train("你是什么样子的ai","我要思考我是什么 我是一个ai" ,"我是一个大模型 我叫deepseek")
    generation("你是什么样子ai",None,max_generate_tokens=60,thinking_available=True)