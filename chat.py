from main import generation


while True:
    try:
        generation(input("\n请输入: "))
    except Exception as e:
        print(e, flush=True)
