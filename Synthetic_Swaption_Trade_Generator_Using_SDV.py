import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

# ----------------------
# SYNTHETIC SWAPTION TRADE GENERATOR 
# This script generates synthetic swaption trade dataset with IFRS 13 levels.
# Team FAIR & SQUARE
# ----------------------

# ----------------------
# CONFIGURATION
# ----------------------
np.random.seed(42)
NUM_TRADES = 1000            # Total number of trades to generate
USD_BIAS_RATIO = 0.4         # Fraction to force as Level 2 USD
BIAS_COUNT = int(NUM_TRADES * USD_BIAS_RATIO)
GEN_COUNT = NUM_TRADES - BIAS_COUNT

# ----------------------
# HELPERS
# ----------------------
def random_past_date(start_days_ago=730, end_days_ago=30):
    days_ago = np.random.randint(end_days_ago, start_days_ago)
    return (datetime.today() - timedelta(days=days_ago)).date()

def determine_ifrs13_level(currency, expiry_tenor, maturity_tenor, strike):
    """Determine IFRS 13 level based on trade features."""
    is_level2 = (
        currency == "USD"
        and expiry_tenor in [2, 3]
        and maturity_tenor < 15
        and strike < 3.0
    )
    return "Level 2" if is_level2 else "Level 3"

def generate_trade(seq_id, force_usd_level2=False):
    trade_date = random_past_date()
    expiry_tenor = np.random.choice([2, 3, 5])
    maturity_tenor = np.random.choice([5, 10, 15, 20, 30])
    expiry_date = trade_date + timedelta(days=int(expiry_tenor * 365))
    maturity_date = trade_date + timedelta(days=int(maturity_tenor * 365))

    if force_usd_level2:
        strike = round(np.random.uniform(0.5, 2.9), 2)
        currency = "USD"
    else:
        strike = round(np.random.uniform(0.5, 5.0), 2)
        currency = np.random.choice(["EUR", "USD", "GBP", "JPY"])

    level = determine_ifrs13_level(currency, expiry_tenor, maturity_tenor, strike)

    return {
        "trade_id": f"HACKTRD{str(seq_id).zfill(4)}",
        "trade_id_type": "HackTradeID",
        "trade_version": np.random.randint(1, 5),
        "product_type": "IR Swaption",
        "currency": currency,
        "option_type": np.random.choice(["Payer", "Receiver"]),
        "notional": int(np.random.randint(10, 1000) * 100_000),
        "trade_date": trade_date,
        "strike": strike,
        "expiry_date": expiry_date,
        "maturity_date": maturity_date,
        "counterparty_id": f"CPTY{np.random.randint(1000, 9999)}",
        "expiry_tenor": expiry_tenor,
        "maturity_tenor": maturity_tenor,
        "ifrs13_level": level
    }

# ----------------------
# DATA GENERATION
# ----------------------
data = []

# USD Level 2 biased trades
for i in range(1, BIAS_COUNT + 1):
    data.append(generate_trade(i, force_usd_level2=True))

# General trades
for i in range(BIAS_COUNT + 1, NUM_TRADES + 1):
    data.append(generate_trade(i, force_usd_level2=False))

df = pd.DataFrame(data)

# ----------------------
# METADATA & CLEANUP
# ----------------------
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

# Fix trade_id from being auto-id'd
metadata.primary_key = None
metadata.update_column("trade_id", sdtype="categorical")

# Set datetime fields properly
metadata.update_column("trade_date", sdtype="datetime")
metadata.update_column("expiry_date", sdtype="datetime")
metadata.update_column("maturity_date", sdtype="datetime")

# ----------------------
# SDV SYNTHESIS
# ----------------------
synth = GaussianCopulaSynthesizer(metadata)
synth.fit(df)
synthetic_df = synth.sample(NUM_TRADES)

# Clean date columns (remove time)
for col in ["trade_date", "expiry_date", "maturity_date"]:
    synthetic_df[col] = pd.to_datetime(synthetic_df[col]).dt.date
# Round notional to nearest 100,000
synthetic_df["notional"] = synthetic_df["notional"].apply(lambda x: round(x / 100_000) * 100_000).astype(int)

# ----------------------
# OUTPUT
# ----------------------
output_file = "Synthetic_Swaption_Trades_With_IFRS13_Level.csv"
synthetic_df.to_csv(output_file, index=False)
print(f"✅ File saved: {output_file}")
print("Sample:")
print(synthetic_df[["trade_id", "currency", "strike", "notional", "expiry_tenor", "maturity_tenor", "ifrs13_level"]].head())
