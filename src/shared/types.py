"""
Shared type definitions for the EU AI Act Compliance Scanner.
All agents import from here to ensure consistent data structures.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RiskLevel(Enum):
    UNACCEPTABLE = "unacceptable"
    HIGH_RISK = "high_risk"
    LIMITED_RISK = "limited_risk"
    MINIMAL_RISK = "minimal_risk"


class Role(Enum):
    PROVIDER = "provider"
    DEPLOYER = "deployer"
    BOTH = "both"


class Priority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Article25Trigger:
    scenario_id: str  # "25_1_a", "25_1_b", "25_1_c"
    scenario_number: int
    short_name: str
    confidence: float  # 0.0 to 1.0
    reasoning: str


@dataclass
class AnnexIIIMatch:
    category_id: int
    subcategory_id: str
    category_name: str
    description: str
    confidence: float  # 0.0 to 1.0
    reasoning: str


@dataclass
class GPAITermsCheck:
    provider_name: str
    product_used: Optional[str]
    terms_violated: bool
    violated_terms: list[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class ObligationGap:
    obligation_id: str  # "P1", "P2", etc.
    article: str
    name: str
    description: str
    priority: Priority
    status: str  # "not_met", "partially_met", "met", "unknown"
    action_items: list[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Output of the Classifier Agent."""
    risk_level: RiskLevel
    annex_iii_match: Optional[AnnexIIIMatch]
    role: Role
    article_25_trigger: Optional[Article25Trigger]
    gpai_terms_check: Optional[GPAITermsCheck]
    transparency_obligations: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class ComplianceReport:
    """Output of the Report Writer Agent."""
    system_description: str
    classification: ClassificationResult
    obligation_gaps: list[ObligationGap] = field(default_factory=list)
    total_obligations: int = 0
    obligations_met: int = 0
    risk_score: float = 0.0  # 0-100
    executive_summary: str = ""
    next_steps: list[str] = field(default_factory=list)
    generated_at: str = ""
