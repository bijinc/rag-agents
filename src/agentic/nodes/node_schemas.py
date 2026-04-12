"""Structured schemas for validating node-level LLM JSON outputs."""

from pydantic import BaseModel, ConfigDict, Field
from pydantic import field_validator
from typing import Literal

from src.finance_domain import (
    normalize_filing_type,
    normalize_fiscal_quarter,
    normalize_section_types,
    normalize_source_type,
    normalize_tickers,
)


class QueryAnalyzerResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sub_questions: list[str]
    tickers: list[str] | None = None
    source_type: Literal["sec_filing", "ect"] | None = None
    filing_type: Literal["10-K", "10-Q", "8-K"] | None = None
    section_types: list[Literal["risk_factors", "md&a", "financial_statements", "forward_looking_statements", "q_and_a", "prepared_remarks", "general"]] | None = None
    fiscal_year: int | None = None
    fiscal_quarter: Literal["Q1", "Q2", "Q3", "Q4"] | None = None
    canonical_period: str | None = None
    period_end_date_from: str | None = None
    period_end_date_to: str | None = None
    question_type: Literal["L1", "L2", "L3", "L4"] = "L1"

    @field_validator("source_type", mode="before")
    @classmethod
    def normalize_source_type_field(cls, value):
        if value is None:
            return None
        return normalize_source_type(value) if isinstance(value, str) else value

    @field_validator("filing_type", mode="before")
    @classmethod
    def normalize_filing_type_field(cls, value):
        if value is None:
            return None
        return normalize_filing_type(value) if isinstance(value, str) else value

    @field_validator("tickers", mode="before")
    @classmethod
    def normalize_tickers(cls, value):
        if value is None:
            return None
        if not isinstance(value, list):
            return value
        return normalize_tickers(value)

    @field_validator("section_types", mode="before")
    @classmethod
    def normalize_section_types(cls, value):
        if value is None:
            return None
        if not isinstance(value, list):
            return value
        return normalize_section_types(value)

    @field_validator("fiscal_quarter", mode="before")
    @classmethod
    def normalize_fiscal_quarter(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            return normalize_fiscal_quarter(value)
        return value


class QueryRefinerResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sub_questions: list[str]


class SufficiencyResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sufficient: bool
    reason: str = Field(default="")


class FaithfulnessResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    faithful: bool
    confidence: Literal["high", "low"] = Field(default="high")
    reasoning: str = Field(default="")
