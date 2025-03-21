import os
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Test script options")

    # Define the command-line arguments
    parser.add_argument("-d", "--data_set", required=True, help="Pass data_set to be used (obrigatório)")
    parser.add_argument("-h", "--hyperparameters", action="store_true", help="Recalculate hyperparameters (optional)")

    # Parse the arguments
    args = parser.parse_args()

    # Accessing the values from the arguments
    data_set = args.data_set
    recalculate_hyperparameters = args.hyperparameters

    # Print or use the parsed arguments as needed
    print(f"Data set: {data_set}")
    if recalculate_hyperparameters:
        print("Recalculating hyperparameters...")

if __name__ == "__main__":
    main()
