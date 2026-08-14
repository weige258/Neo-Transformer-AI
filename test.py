from main import generation, train

for i in range(50):
    # 纯文本训练示例：垂直领域对话提示模板
    train(None, None, "这是一个简单测试", None)

    # CoT示例1：清晰过程思路，think是过程，answer是最终回答
    train("这是一个AI助手提示", "请给出一个简洁的回答。", "这是最终回答。")

    # CoT示例2：概念结构过程，think和answer分别归纳
    train(
        "python是什么？",
        "Python是一种高级编程语言，适合快速开发。",
        "Python适用于数据科学、Web开发和自动化。"
    )

    # 反馈示例：用于换行思路和复核操作
    train("1+1等于多少？", "这是一个数学问题。", "1+1等于2。")
    train("你好", "用户开始对话。", "你好，欢迎使用测试脚本。")


generation("你好", None, max_generate_tokens=256, thinking_available=True)
generation("1+1等于多少？", None, max_generate_tokens=256, thinking_available=True)
generation("python是什么？", None, max_generate_tokens=256, thinking_available=True)