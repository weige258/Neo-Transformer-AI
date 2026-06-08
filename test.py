from main import generation, train

for i in range(50):
    # 单文本训练：基础语言模式
    train(None, None, "你是什么ai 我是deepseek ai", None)
    
    # CoT训练1：正确格式，think是推理过程，answer是最终答案
    train("你是什么样子的ai", "我要思考我是什么。我是一个ai，一个大语言模型。", "我是一个大模型，我叫deepseek。")
    
    # CoT训练2：修复格式，think和answer正确分离
    train("python是什么语言", 
          "python是一种编程语言。让我想想它的特点：它是解释型的、动态的、支持多种编程范式。",
          "python是一种高级编程语言，具有简洁优雅的语法。",
          "cpp是什么语言？cpp是静态类型语言。")
    
    # 额外训练：帮助模型学习THINK_START和THINK_END的正确使用
    train("1+1等于几", "这是一个简单的数学问题。1加1等于2。", "1+1等于2。")
    train("你好", "用户向我打招呼，我应该礼貌回应。", "你好！很高兴见到你。")
    
generation("你是什么样子的ai", None, max_generate_tokens=100, thinking_available=True)    
generation("python是什么语言", None, max_generate_tokens=100, thinking_available=True)