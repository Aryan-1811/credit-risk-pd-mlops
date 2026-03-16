"""
Interactive Credit Risk Demo — Score a New Customer.

Loads the trained LightGBM model directly from MLflow local storage
and scores a borrower interactively in the terminal.

Run with:
    python scripts/demo.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import pickle
import pandas as pd
import numpy as np

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
CYAN   = "\033[96m"


def load_model():
    """Load the most recently trained model from mlruns folder."""
    pkls = glob.glob("mlruns/**/model.pkl", recursive=True)
    if not pkls:
        print(f"{RED}No trained model found. Run the training pipeline first.{RESET}")
        sys.exit(1)
    latest = max(pkls, key=os.path.getmtime)
    with open(latest, "rb") as f:
        model = pickle.load(f)
    print(f"{GREEN}Model loaded from: {latest}{RESET}")
    return model


def get_feature_names(model):
    """Extract feature names from model or its base estimator."""
    for obj in [model, getattr(model, "estimator", None)]:
        if obj is None:
            continue
        if hasattr(obj, "feature_names_in_"):
            return list(obj.feature_names_in_)
        if hasattr(obj, "feature_name_"):
            return list(obj.feature_name_())
    return None


def get_risk_band(pd_score):
    if pd_score < 0.05:
        return "LOW", GREEN
    elif pd_score < 0.15:
        return "MEDIUM", YELLOW
    elif pd_score < 0.30:
        return "HIGH", YELLOW
    else:
        return "VERY HIGH", RED


def get_input(prompt, min_val, max_val, default):
    while True:
        try:
            raw = input(f"  {prompt} [{default}]: ").strip()
            val = float(raw) if raw else float(default)
            if min_val <= val <= max_val:
                return val
            else:
                print(f"    {YELLOW}Enter a value between {min_val} and {max_val}{RESET}")
        except ValueError:
            print(f"    {YELLOW}Please enter a number{RESET}")


def collect_borrower_details():
    print(f"\n{BOLD}{CYAN}Enter borrower details (press Enter to use default):{RESET}\n")
    return {
        "RevolvingUtilizationOfUnsecuredLines": get_input("Revolving credit utilisation (0.0–1.0)", 0.0, 1.0, 0.3),
        "age": int(get_input("Age", 18, 100, 40)),
        "NumberOfTime30-59DaysPastDueNotWorse": int(get_input("Times 30-59 days late (last 2 years)", 0, 20, 0)),
        "DebtRatio": get_input("Debt ratio (e.g. 0.35 = 35%)", 0.0, 50.0, 0.35),
        "MonthlyIncome": get_input("Monthly income (£)", 0, 500000, 5000),
        "NumberOfOpenCreditLinesAndLoans": int(get_input("Number of open credit lines", 0, 50, 4)),
        "NumberOfTimes90DaysLate": int(get_input("Times 90+ days late", 0, 20, 0)),
        "NumberRealEstateLoansOrLines": int(get_input("Number of real estate loans", 0, 20, 1)),
        "NumberOfTime60-89DaysPastDueNotWorse": int(get_input("Times 60-89 days late", 0, 20, 0)),
        "NumberOfDependents": int(get_input("Number of dependents", 0, 20, 0)),
    }


def print_result(pd_score):
    risk_band, colour = get_risk_band(pd_score)
    lgd = 0.45
    el = pd_score * lgd
    print(f"\n{'='*55}")
    print(f"{BOLD}  CREDIT RISK ASSESSMENT RESULT{RESET}")
    print(f"{'='*55}")
    print(f"  Probability of Default : {colour}{BOLD}{pd_score:.4f} ({pd_score*100:.2f}%){RESET}")
    print(f"  Risk Band              : {colour}{BOLD}{risk_band}{RESET}")
    print(f"  Expected Loss (per £1) : {colour}{BOLD}£{el:.4f}{RESET}")
    print(f"  (EL = PD x LGD, LGD = {lgd} — Basel II floor)")
    print(f"{'='*55}")
    print(f"\n{BOLD}  Risk Band Guide:{RESET}")
    print(f"  {GREEN}LOW       {RESET} PD < 5%   — Strong borrower")
    print(f"  {YELLOW}MEDIUM    {RESET} PD 5-15%  — Acceptable risk")
    print(f"  {YELLOW}HIGH      {RESET} PD 15-30% — Elevated risk")
    print(f"  {RED}VERY HIGH {RESET} PD > 30%  — Decline or premium rate\n")


def main():
    print(f"\n{BOLD}{'='*55}")
    print(f"  CREDIT RISK PD SCORING DEMO")
    print(f"  Basel II-aligned Probability of Default Model")
    print(f"{'='*55}{RESET}")

    print(f"\n{CYAN}Loading model...{RESET}")
    model = load_model()
    print(f"{GREEN}Model loaded successfully.{RESET}")

    # Get the exact feature names the model expects
    expected_features = get_feature_names(model)
    if expected_features:
        print(f"{CYAN}Model expects {len(expected_features)} features.{RESET}")

    while True:
        raw_data = collect_borrower_details()
        borrower_df = pd.DataFrame([raw_data])

        # Align to model's expected features
        if expected_features:
            for col in expected_features:
                if col not in borrower_df.columns:
                    borrower_df[col] = 0
            borrower_df = borrower_df[expected_features]

        try:
            pd_score = float(model.predict_proba(borrower_df)[0, 1])
            print_result(pd_score)
        except Exception as e:
            print(f"{RED}Scoring error: {e}{RESET}")

        again = input("Score another borrower? (y/n): ").strip().lower()
        if again != "y":
            print(f"\n{CYAN}Exiting demo. Goodbye!{RESET}\n")
            break


if __name__ == "__main__":
    main()
