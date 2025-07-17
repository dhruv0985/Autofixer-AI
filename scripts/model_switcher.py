import json

AVAILABLE_MODELS = [
    "codellama:7b-python",
    "deepseek-coder:6.7b",
    
]

def main():
    print("\n Available Models:")
    for idx, model in enumerate(AVAILABLE_MODELS, start=1):
        print(f"  {idx}. {model}")

    choice = input("\n Pick a model by number: ")

    try:
        choice = int(choice)
        if 1 <= choice <= len(AVAILABLE_MODELS):
            selected = AVAILABLE_MODELS[choice - 1]
            with open("config.json", "w") as f:
                json.dump({"OLLAMA_MODEL": selected}, f, indent=2)
            print(f"\n  Model switched to: {selected}\n")
        else:
            print("\n  Invalid choice.")
    except ValueError:
        print("\n Please enter a valid number.")

if __name__ == "__main__":
    main()
