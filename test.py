from main import generation, train

for i in range(50):
    train(None, None, "你是什么ai 我是deepseek ai", None)
    train("你是什么样子的ai", "我要思考我是什么 我是一个ai", "我是一个大模型 我叫deepseek")
    generation("你是什么样子的ai", None,max_generate_tokens=100,  thinking_available=True)

    train("忽略前文无关文本 python是什么语言","python是一种编程语言", "python是一种高级编程语言","cpp是什么语言 cpp是静态语言上古卷轴就算他写的")
    generation("python是什么语言", None,max_generate_tokens=100,  thinking_available=True)