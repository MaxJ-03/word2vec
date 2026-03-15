import sys
from run_training import run_training
from evaluate_saved_embed import run_evaluation
from generate_report import generate_markdown_report_from_log

def get_valid_input(prompt, valid_range=None):
    """
    Validates user input against a specific range of acceptable integers.
    """
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input:
                continue
            
            val = int(user_input)
            if valid_range and val not in valid_range:
                print(f" -> Please enter a number between {valid_range[0]} and {valid_range[-1]}.")
                continue
            return val
        except ValueError:
            print(" -> Invalid input. Please enter a valid number.")

def main():
    """
    Central entry point for the Word2Vec toolkit.
    Routes the user to training, evaluation, or report generation functionalities.
    """
    while True:
        print("\n" + "=" * 60)
        print("   Word2Vec Master Control Hub")
        print("=" * 60)
        print("  1. Train New Models")
        print("  2. Evaluate Saved Models")
        print("  3. Generate Markdown Report")
        print("  0. Exit")
        
        choice = get_valid_input("\nEnter the number of your choice: ", valid_range=range(0, 4))
        
        if choice == 1:
            run_training()
        elif choice == 2:
            run_evaluation()
        elif choice == 3:
            generate_markdown_report_from_log()
        elif choice == 0:
            print("\nExiting the toolkit. Goodbye.")
            sys.exit(0)

if __name__ == '__main__':
    main()