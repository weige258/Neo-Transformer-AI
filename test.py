from main import generation, train


for i in range(50):
    train(None,None,"你是什么ai 我是deepseek ai",None)
    train("你是什么样子的ai","我要思考我是什么 我是一个ai" ,"我是一个大模型 我叫deepseek")
    generation("你是什么样子的ai",None,max_generate_tokens=64,thinking_available=True)

    train("历史上下文有很多无关内容 忽略,回答我python是什么语言","python是一种编程语言","python是一种高级编程语言，广泛用于数据分析、人工智能、web开发等领域。","上下文 你好")
    generation("回答我python是什么语言",max_generate_tokens=64)


    