#!/usr/bin/env python3
import pandas as pd

def main():
    # File paths
    aaron_file = "./data/aaron_code_data.csv"
    yulei_file = "./data/yulei_code_data.pkl"

    # Read only the header of the CSV
    df_aaron = pd.read_csv(aaron_file, nrows=0)
    print(f"Aaron CSV columns ({aaron_file}):")
    for col in df_aaron.columns:
        print(f"  {col}")

    # Read the PKL (must load to inspect)
    df_yulei = pd.read_pickle(yulei_file)
    print(f"\nYulei PKL columns ({yulei_file}):")
    for col in df_yulei.columns:
        print(f"  {col}")

if __name__ == "__main__":
    main()
