"""Country and currency overlay.

Country risk premium (CRP): extra return investors demand for a company's exposure to
a riskier country. Added to the cost of equity in the DCF (scaled by "lambda", the
company's exposure, 1 = fully exposed).

The defaults below are APPROXIMATE, rating-based estimates in the style of Aswath
Damodaran's annual country risk table (pages.stern.nyu.edu/~adamodar). Check them
against his latest update and edit them on the page; they are inputs, not facts.

Risk-free rates: only USD is live (10-year Treasury yield, ^TNX via yfinance). The
other currencies use placeholder defaults: set them to the current 10-year
government bond yield of that currency before relying on a valuation.
"""
from __future__ import annotations

# country -> (approximate CRP, currency)
COUNTRIES = {
    "United States": (0.000, "USD"), "Germany": (0.000, "EUR"), "Netherlands": (0.000, "EUR"),
    "Switzerland": (0.000, "CHF"), "Denmark": (0.000, "DKK"), "Sweden": (0.000, "SEK"), "Norway": (0.000, "NOK"),
    "Luxembourg": (0.000, "EUR"), "Canada": (0.000, "CAD"), "Australia": (0.000, "AUD"),
    "United Kingdom": (0.007, "GBP"), "France": (0.008, "EUR"), "Belgium": (0.008, "EUR"), "Austria": (0.006, "EUR"),
    "Finland": (0.006, "EUR"), "Ireland": (0.007, "EUR"), "Japan": (0.008, "JPY"),
    "Spain": (0.016, "EUR"), "Portugal": (0.019, "EUR"), "Italy": (0.025, "EUR"),
    "South Africa": (0.040, "ZAR"), "Brazil": (0.035, "BRL"), "India": (0.022, "INR"), "China": (0.008, "CNY"),
}
CRP_SOURCE = "approximate rating-based default (Damodaran-style); verify and edit"

# currency -> placeholder 10-year government bond yield (USD is replaced by live ^TNX when available)
RISK_FREE_DEFAULTS = {"USD": 0.043, "EUR": 0.027, "GBP": 0.045, "CHF": 0.007, "SEK": 0.025, "DKK": 0.025,
                      "NOK": 0.039, "ZAR": 0.100, "JPY": 0.015, "CAD": 0.032, "AUD": 0.043}
RF_SOURCE = "placeholder default; set to today's 10-year government bond yield"

MATURE_ERP = 0.045  # mature-market equity risk premium, editable on the page
ERP_SOURCE = "default mature-market ERP (Damodaran's implied ERP has ranged ~4-6%); edit"


def crp_for(country: str) -> float:
    return COUNTRIES.get(country, (0.0, None))[0]


def risk_free_default(currency: str) -> float:
    return RISK_FREE_DEFAULTS.get(currency, 0.04)
