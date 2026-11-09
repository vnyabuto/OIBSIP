
from pathlib import Path

import joblib
import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_car_price_model.joblib"

REFERENCE_YEAR = pd.Timestamp.now().year

CURRENCIES = {
    "1": ("INR", "Indian Rupees", "₹"),
    "2": ("KES", "Kenyan Shillings", "KSh "),
    "3": ("USD", "US Dollars", "$")
}


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run car_price_prediction.py first."
    )

model = joblib.load(MODEL_PATH)


def get_numeric_input(prompt, minimum=None, maximum=None, integer=False):
    while True:
        try:
            value = float(input(prompt))

            if not pd.notna(value):
                raise ValueError

            if minimum is not None and value < minimum:
                print(f"Value must be at least {minimum}.")
                continue

            if maximum is not None and value > maximum:
                print(f"Value must not exceed {maximum}.")
                continue

            if integer:
                if not value.is_integer():
                    print("Please enter a whole number.")
                    continue

                return int(value)

            return value

        except ValueError:
            print("Invalid input. Please enter a valid number.")


def get_categorical_input(prompt, options):
    while True:
        value = input(prompt).strip().title()

        if value in options:
            return value

        print(f"Choose one of: {', '.join(options)}")


def get_currency():
    print("\nSELECT OUTPUT CURRENCY")
    print("=" * 45)

    for key, (code, name, symbol) in CURRENCIES.items():
        print(f"{key}. {name} ({code})")

    while True:
        choice = input("Choose currency (1-3): ").strip()

        if choice in CURRENCIES:
            return CURRENCIES[choice]

        print("Invalid selection. Choose 1, 2 or 3.")


def convert_currency(amount, target_currency):
    url = (
        f"https://api.frankfurter.dev/v2/rate/INR/{target_currency}"
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    rate_data = response.json()

    converted_amount = amount * rate_data["rate"]

    return converted_amount, rate_data["rate"], rate_data["date"]


def predict_car_price():

    print("\nCAR PRICE PREDICTION SYSTEM")
    print("=" * 55)

    year = get_numeric_input(
        "Manufacturing year: ",
        minimum=1900,
        maximum=REFERENCE_YEAR,
        integer=True
    )

    present_price = get_numeric_input(
        "Present showroom price (lakh INR): ",
        minimum=0.01
    )

    kms_driven = get_numeric_input(
        "Kilometres driven: ",
        minimum=0
    )

    owner = get_numeric_input(
        "Previous owners (0, 1, 2, etc.): ",
        minimum=0,
        integer=True
    )

    fuel_type = get_categorical_input(
        "Fuel type (Petrol/Diesel/CNG): ",
        ["Petrol", "Diesel", "Cng"]
    )

    seller_type = get_categorical_input(
        "Seller type (Dealer/Individual): ",
        ["Dealer", "Individual"]
    )

    transmission = get_categorical_input(
        "Transmission (Manual/Automatic): ",
        ["Manual", "Automatic"]
    )

    currency_code, currency_name, currency_symbol = get_currency()

    car_age = REFERENCE_YEAR - year

    input_data = pd.DataFrame([{
        "Car_Age": car_age,
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Owner": owner,
        "Fuel_Type": fuel_type,
        "Seller_Type": seller_type,
        "Transmission": transmission
    }])

    prediction = float(model.predict(input_data)[0])

    print("\nPREDICTION RESULT")
    print("=" * 55)
    print(f"Vehicle age: {car_age} years")
    print(f"Original estimate: ₹{prediction:.2f} lakh INR")
    print(f"Original estimate: ₹{prediction * 100000:,.2f} INR")

    if prediction < 0:
        print("\nWarning: The model produced a negative estimate.")
        print("This prediction should not be treated as reliable.")
        return

    if currency_code == "INR":
        print(f"\nEstimated selling price: ₹{prediction * 100000:,.2f} INR")
    else:
        try:
            converted_amount, exchange_rate, rate_date = convert_currency(
                prediction * 100000,
                currency_code
            )

            print(f"\nEstimated selling price: {currency_symbol}{converted_amount:,.2f}")
            print(f"Currency: {currency_name}")
            print(f"Exchange rate: 1 INR = {exchange_rate:.6f} {currency_code}")
            print(f"Exchange rate date: {rate_date}")

        except requests.RequestException:
            print("\nCurrency conversion service unavailable.")
            print("Showing the original INR prediction instead.")

    print("\nNote: Converted amounts are currency equivalents,")
    print("not local-market vehicle valuations.")
    print("\nPrediction completed.")


if __name__ == "__main__":
    predict_car_price()
