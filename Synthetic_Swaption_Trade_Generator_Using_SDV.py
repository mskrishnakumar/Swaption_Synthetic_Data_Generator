import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

# ----------------------
# SYNTHETIC SWAPTION TRADE GENERATOR 
# Team FAIR & SQUARE
# ----------------------

np.random.seed(42)
NUM_TRADES = 1000

# ----------------------
# HELPERS
# ----------------------
def random_past_date(start_days_ago=730, end_days_ago=30):
    days_ago = np.random.randint(end_days_ago, start_days_ago)
    return (datetime.today() - timedelta(days=days_ago)).date()

def determine_ifrs13_level(currency, expiry_tenor, maturity_tenor, strike):
    is_level2 = (
        currency == "USD"
        and expiry_tenor in [2, 3]
        and maturity_tenor < 15
        and strike < 3.0
    )
    return "Level 2" if is_level2 else "Level 3"

def generate_trade(seq_id, force_usd_level2=False, force_level3=False):
    trade_date = random_past_date()
    expiry_tenor = np.random.choice([2, 3, 5])
    maturity_tenor = np.random.choice([5, 10, 15, 20, 30])
    expiry_date = trade_date + timedelta(days=int(expiry_tenor * 365))
    maturity_date = trade_date + timedelta(days=int(maturity_tenor * 365))

    if force_usd_level2:
        strike = round(np.random.uniform(0.5, 2.9), 2)
        currency = "USD"
    elif force_level3:
        strike = round(np.random.uniform(3.1, 5.0), 2)
        currency = np.random.choice(["EUR", "GBP", "JPY"])
        expiry_tenor = np.random.choice([1, 5])
        maturity_tenor = np.random.choice([15, 20, 30])
        expiry_date = trade_date + timedelta(days=int(expiry_tenor * 365))
        maturity_date = trade_date + timedelta(days=int(maturity_tenor * 365))
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
        "ifrs13_level": level,
        "Day2_Pnl_Above_Threshold": "No"  # Placeholder, updated below
    }

# ----------------------
# DATA GENERATION
# ----------------------
data = []

level2_count = int(NUM_TRADES * 0.8)
level3_count = NUM_TRADES - level2_count
level3_yes_count = int(level3_count * 0.15)
level3_no_count = level3_count - level3_yes_count

# Generate Level 2 trades (all with PnL = No)
for i in range(1, level2_count + 1):
    trade = generate_trade(i, force_usd_level2=True)
    trade["Day2_Pnl_Above_Threshold"] = "No"
    data.append(trade)

# Generate Level 3 trades
# 15% of Level 3 → PnL Yes
for i in range(level2_count + 1, level2_count + 1 + level3_yes_count):
    trade = generate_trade(i, force_level3=True)
    trade["Day2_Pnl_Above_Threshold"] = "Yes"
    data.append(trade)

# 85% of Level 3 → PnL No
for i in range(level2_count + 1 + level3_yes_count, NUM_TRADES + 1):
    trade = generate_trade(i, force_level3=True)
    trade["Day2_Pnl_Above_Threshold"] = "No"
    data.append(trade)

df = pd.DataFrame(data)

# ----------------------
# METADATA & SYNTHESIS
# ----------------------
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)
metadata.primary_key = None
metadata.update_column("trade_id", sdtype="categorical")
metadata.update_column("trade_date", sdtype="datetime")
metadata.update_column("expiry_date", sdtype="datetime")
metadata.update_column("maturity_date", sdtype="datetime")

synth = GaussianCopulaSynthesizer(metadata)
synth.fit(df)
synthetic_df = synth.sample(NUM_TRADES)

# Clean up
for col in ["trade_date", "expiry_date", "maturity_date"]:
    synthetic_df[col] = pd.to_datetime(synthetic_df[col]).dt.date
synthetic_df["notional"] = synthetic_df["notional"].apply(lambda x: round(x / 100_000) * 100_000).astype(int)

# ----------------------
# OUTPUT
# ----------------------
output_file = "Synthetic_Swaption_Trades_With_IFRS13_Level.csv"
synthetic_df.to_csv(output_file, index=False)
print(f"✅ File saved: {output_file}")
print("Sample:")
print(synthetic_df[["trade_id", "currency", "strike", "Day2_Pnl_Above_Threshold", "ifrs13_level"]].head())

# ----------------------
# DISTRIBUTION SUMMARY
# ----------------------
print("\n📊 [AFTER SYNTHESIS] IFRS13 Level Distribution (%):")
print(synthetic_df['ifrs13_level'].value_counts(normalize=True).mul(100).round(2).to_string())

print("\n📊 [AFTER SYNTHESIS] Day2_Pnl_Above_Threshold Distribution (%):")
print(synthetic_df['Day2_Pnl_Above_Threshold'].value_counts(normalize=True).mul(100).round(2).to_string())

print("\n📊 [AFTER SYNTHESIS] Cross Distribution (% by Day2_Pnl_Above_Threshold):")
cross_tab = pd.crosstab(
    synthetic_df["Day2_Pnl_Above_Threshold"],
    synthetic_df["ifrs13_level"],
    normalize="index"
).mul(100).round(2)
print(cross_tab.to_string())
