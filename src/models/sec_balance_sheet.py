
from pydantic import BaseModel
from datetime import date
from typing import Any


class CurrentAssets(BaseModel):
    cash_and_cash_equivalents: float
    short_term_investments: float
    accounts_receivable: float
    inventory: float
    other_current_assets: float


class NonCurrentAssets(BaseModel):
    property_plant_equipment: float
    long_term_investments: float
    intangible_assets: float
    goodwill: float
    other_non_current_assets: float


class CurrentLiabilities(BaseModel):
    accounts_payable: float
    short_term_debt: float
    accrued_expenses: float
    other_current_liabilities: float


class NonCurrentLiabilities(BaseModel):
    long_term_debt: float
    deferred_tax_liabilities: float
    other_non_current_liabilities: float


class Equity(BaseModel):
    common_stock: float
    retained_earnings: float
    additional_paid_in_capital: float
    treasury_stock: float
    other_equity: float


class SecBalanceSheet(BaseModel):
    ticker: str
    company_name: str
    fiscal_year_start: date
    fiscal_year_end: date
    current_assets: CurrentAssets
    non_current_assets: NonCurrentAssets
    total_assets: float
    current_liabilities: CurrentLiabilities
    non_current_liabilities: NonCurrentLiabilities
    equity: Equity
    total_liabilities_and_equity: float

    @classmethod
    def from_edgar(cls, ticker: str, company_name: str, statement: Any) -> "SecBalanceSheet":
        """Build a serializable balance-sheet model from an Edgar Statement object."""
        raw_rows = statement.get_raw_data()

        def _safe_float(value: Any) -> float:
            if value is None:
                return 0.0
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        def _extract_date_key(value_key: str) -> date | None:
            # Keys are typically shaped like `instant_YYYY-MM-DD`.
            parts = value_key.split("_")
            if not parts:
                return None
            try:
                return date.fromisoformat(parts[-1])
            except ValueError:
                return None

        all_dates: set[date] = set()
        for row in raw_rows:
            values = row.get("values") or {}
            for key in values.keys():
                parsed = _extract_date_key(key)
                if parsed is not None:
                    all_dates.add(parsed)

        sorted_dates = sorted(all_dates)
        fiscal_year_end = sorted_dates[-1] if sorted_dates else date.today()
        fiscal_year_start = sorted_dates[-2] if len(sorted_dates) > 1 else fiscal_year_end

        def _pick_latest_value(row: dict[str, Any]) -> float:
            values = row.get("values") or {}
            dated_values: list[tuple[date, float]] = []
            for key, raw_value in values.items():
                parsed = _extract_date_key(str(key))
                if parsed is None:
                    continue
                dated_values.append((parsed, _safe_float(raw_value)))

            if not dated_values:
                return 0.0

            dated_values.sort(key=lambda x: x[0])
            return dated_values[-1][1]

        def _lookup_value(
            concept_candidates: list[str],
            label_contains: list[str] | None = None,
        ) -> float:
            for row in raw_rows:
                concept = str(row.get("concept") or "")
                if concept not in concept_candidates:
                    continue

                if label_contains:
                    label = str(row.get("label") or "").lower()
                    if not any(token in label for token in label_contains):
                        continue

                return _pick_latest_value(row)

            return 0.0

        current_assets = CurrentAssets(
            cash_and_cash_equivalents=_lookup_value([
                "us-gaap_CashAndCashEquivalentsAtCarryingValue",
                "us-gaap_CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            ]),
            short_term_investments=_lookup_value([
                "us-gaap_MarketableSecuritiesCurrent",
                "us-gaap_ShortTermInvestments",
            ]),
            accounts_receivable=_lookup_value([
                "us-gaap_AccountsReceivableNetCurrent",
                "us-gaap_AccountsReceivableNet",
            ]),
            inventory=_lookup_value([
                "us-gaap_InventoryNet",
            ]),
            other_current_assets=_lookup_value([
                "us-gaap_OtherAssetsCurrent",
            ]),
        )

        non_current_assets = NonCurrentAssets(
            property_plant_equipment=_lookup_value([
                "us-gaap_PropertyPlantAndEquipmentNet",
            ]),
            long_term_investments=_lookup_value([
                "us-gaap_MarketableSecuritiesNoncurrent",
                "us-gaap_LongTermInvestments",
            ]),
            intangible_assets=_lookup_value([
                "us-gaap_FiniteLivedIntangibleAssetsNet",
                "us-gaap_IntangibleAssetsNetExcludingGoodwill",
            ]),
            goodwill=_lookup_value([
                "us-gaap_Goodwill",
            ]),
            other_non_current_assets=_lookup_value([
                "us-gaap_OtherAssetsNoncurrent",
            ]),
        )

        current_liabilities = CurrentLiabilities(
            accounts_payable=_lookup_value([
                "us-gaap_AccountsPayableCurrent",
            ]),
            short_term_debt=_lookup_value([
                "us-gaap_ShortTermBorrowings",
                "us-gaap_CommercialPaper",
                "us-gaap_LongTermDebtCurrent",
            ]),
            accrued_expenses=_lookup_value([
                "us-gaap_AccruedLiabilitiesCurrent",
                "us-gaap_AccruedIncomeTaxesCurrent",
            ]),
            other_current_liabilities=_lookup_value([
                "us-gaap_OtherLiabilitiesCurrent",
            ]),
        )

        non_current_liabilities = NonCurrentLiabilities(
            long_term_debt=_lookup_value([
                "us-gaap_LongTermDebtNoncurrent",
                "us-gaap_LongTermDebt",
            ]),
            deferred_tax_liabilities=_lookup_value([
                "us-gaap_DeferredTaxLiabilitiesNoncurrent",
                "us-gaap_DeferredTaxLiabilities",
            ]),
            other_non_current_liabilities=_lookup_value([
                "us-gaap_OtherLiabilitiesNoncurrent",
            ]),
        )

        equity = Equity(
            common_stock=_lookup_value(
                ["us-gaap_CommonStockValue", "us-gaap_StockholdersEquity"],
                label_contains=["common stock"],
            ),
            retained_earnings=_lookup_value(
                ["us-gaap_RetainedEarningsAccumulatedDeficit", "us-gaap_StockholdersEquity"],
                label_contains=["retained earnings", "accumulated deficit"],
            ),
            additional_paid_in_capital=_lookup_value(
                ["us-gaap_AdditionalPaidInCapital", "us-gaap_StockholdersEquity"],
                label_contains=["additional paid-in capital", "common stock and additional paid-in capital"],
            ),
            treasury_stock=_lookup_value(
                ["us-gaap_TreasuryStockValue", "us-gaap_TreasuryStockCommonValue"],
            ),
            other_equity=_lookup_value(
                ["us-gaap_AccumulatedOtherComprehensiveIncomeLossNetOfTax", "us-gaap_StockholdersEquity"],
                label_contains=["other comprehensive income", "other comprehensive"],
            ),
        )

        total_assets = _lookup_value(["us-gaap_Assets"])
        total_liabilities = _lookup_value(["us-gaap_Liabilities"])
        total_equity = _lookup_value(["us-gaap_StockholdersEquity"], label_contains=["total shareholders", "total stockholders"])
        total_liabilities_and_equity = total_liabilities + total_equity

        return cls(
            ticker=ticker,
            company_name=company_name,
            fiscal_year_start=fiscal_year_start,
            fiscal_year_end=fiscal_year_end,
            current_assets=current_assets,
            non_current_assets=non_current_assets,
            total_assets=total_assets,
            current_liabilities=current_liabilities,
            non_current_liabilities=non_current_liabilities,
            equity=equity,
            total_liabilities_and_equity=total_liabilities_and_equity,
        )
