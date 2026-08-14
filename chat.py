from main import generation
import sys


def main() -> None:
    while True:
        try:
            user_input = input("\n请输入: ")
            if not user_input or not user_input.strip():
                continue
            generation(user_input)  # None表示无限制，由模型通过END_TOKEN决定何时停止
        except KeyboardInterrupt:
            # 允许KeyboardInterrupt传播，让用户可以Ctrl+C退出
            print("\n\n程序已退出。", flush=True)
            sys.exit(0)
        except EOFError:
            # 处理Ctrl+D或输入流结束
            print("\n\n输入结束，程序退出。", flush=True)
            sys.exit(0)
        except Exception as e:
            # 只捕获非退出相关的异常
            print(f"\n发生错误: {e}", flush=True)
            print("请输入新内容继续对话，或按Ctrl+C退出。", flush=True)


if __name__ == '__main__':
    main()