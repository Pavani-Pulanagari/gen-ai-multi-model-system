from pipeline import run_pipeline

if __name__ == "__main__":
    
    print("Select Task: generate / translate / qa")
    task = input("Enter task: ")

    user_input = input("Enter your input: ")

    result = run_pipeline(task, user_input)

    print("\nOutput:")
    print(result)
