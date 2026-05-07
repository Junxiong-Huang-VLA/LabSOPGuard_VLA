"""
Scene Understanding Engine for LabSOPGuard.

Provides experiment type taxonomy, chemistry entity definitions,
and SceneProfile data structures. Framework-ready for AI/VLM integration.

When AI is available, the SceneUnderstander will:
1. Analyze initial frames to classify experiment type
2. Identify visible containers, tools, and reagents
3. Generate expected step sequence and constraints
4. Output a SceneProfile that drives all downstream analysis.

Currently provides rule-based fallback classification and
structured data models for immediate use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Chemistry Entity Taxonomy
# ---------------------------------------------------------------------------

class ContainerType(str, Enum):
    BEAKER = "beaker"
    VOLUMETRIC_FLASK = "volumetric_flask"
    ERLENMEYER_FLASK = "erlenmeyer_flask"
    GRADUATED_CYLINDER = "graduated_cylinder"
    BURETTE = "burette"
    PIPETTE = "pipette"
    PIPETTE_AID = "pipette_aid"
    REAGENT_BOTTLE = "reagent_bottle"
    WASH_BOTTLE = "wash_bottle"
    CRUCIBLE = "crucible"
    WATCH_GLASS = "watch_glass"
    PETRI_DISH = "petri_dish"
    TEST_TUBE = "test_tube"
    FUNNEL = "funnel"
    SPATULA = "spatula"
    WEIGHT_BOAT = "weight_boat"
    OTHER = "other"


class ToolType(str, Enum):
    MAGNETIC_STIRRER = "magnetic_stirrer"
    HOT_PLATE = "hot_plate"
    BALANCE = "balance"
    pH_METER = "ph_meter"
    THERMOMETER = "thermometer"
    TIMER = "timer"
    VORTEX_MIXER = "vortex_mixer"
    CENTRIFUGE = "centrifuge"
    OTHER = "other"


class OperationType(str, Enum):
    POUR = "pour"
    TRANSFER = "transfer"
    DISPENSE = "dispense"
    TITRATE = "titrate"
    MEASURE = "measure"
    WEIGH = "weigh"
    STIR = "stir"
    HEAT = "heat"
    COOL = "cool"
    FILTER = "filter"
    DECANT = "decant"
    RINSE = "rinse"
    DRY = "dry"
    LABEL = "label"
    VERIFY = "verify"
    DISPOSE = "dispose"
    CLEAN = "clean"
    MIX = "mix"
    READ_VOLUME = "read_volume"
    CALIBRATE = "calibrate"
    UNKNOWN = "unknown"


class ExperimentType(str, Enum):
    ACID_BASE_TITRATION = "acid_base_titration"
    REDOX_TITRATION = "redox_titration"
    SOLUTION_PREPARATION = "solution_preparation"
    SERIAL_DILUTION = "serial_dilution"
    PIPETTING = "pipetting"
    WEIGHING = "weighing"
    FILTRATION = "filtration"
    EXTRACTION = "extraction"
    CRYSTALLIZATION = "crystallization"
    HEATING_REACTION = "heating_reaction"
    PH_MEASUREMENT = "ph_measurement"
    COLORIMETRY = "colorimetry"
    TITRATION_GENERAL = "titration_general"
    GENERAL_LAB_OPERATION = "general_lab_operation"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Expected steps per experiment type
# ---------------------------------------------------------------------------

EXPERIMENT_STEP_TEMPLATES: Dict[ExperimentType, List[Dict[str, Any]]] = {
    ExperimentType.ACID_BASE_TITRATION: [
        {"step_id": "prepare_ppe", "name_zh": "穿戴个人防护装备", "name_en": "Wear PPE",
         "keywords": ["glove", "goggles", "lab coat", "ppe", "防护", "手套", "护目镜"],
         "constraints": ["must_have_gloves", "must_have_goggles"]},
        {"step_id": "prepare_burette", "name_zh": "准备滴定管", "name_en": "Prepare burette",
         "keywords": ["burette", "rinse", "fill", "滴定管", "润洗", "装液"],
         "constraints": ["burette_must_be_rinsed"]},
        {"step_id": "measure_analyte", "name_zh": "量取待测液", "name_en": "Measure analyte",
         "keywords": ["pipette", "volumetric flask", "measure", "移液", "量取", "容量瓶"],
         "constraints": ["accurate_volume_required"]},
        {"step_id": "add_indicator", "name_zh": "加入指示剂", "name_en": "Add indicator",
         "keywords": ["indicator", "phenolphthalein", "methyl orange", "指示剂"],
         "constraints": []},
        {"step_id": "execute_titration", "name_zh": "执行滴定", "name_en": "Execute titration",
         "keywords": ["titrate", "drop", "endpoint", "color change", "滴定", "终点", "变色"],
         "constraints": ["slow_addition_near_endpoint"]},
        {"step_id": "read_volume", "name_zh": "读取体积", "name_en": "Read volume",
         "keywords": ["read", "meniscus", "volume", "读数", "液面", "体积"],
         "constraints": ["eye_level_reading"]},
        {"step_id": "record_data", "name_zh": "记录数据", "name_en": "Record data",
         "keywords": ["record", "note", "data", "记录", "数据"],
         "constraints": []},
        {"step_id": "clean_workspace", "name_zh": "清洁工作台", "name_en": "Clean workspace",
         "keywords": ["clean", "wash", "wipe", "清洁", "清洗", "擦拭"],
         "constraints": []},
        {"step_id": "dispose_waste", "name_zh": "处理废液", "name_en": "Dispose waste",
         "keywords": ["waste", "dispose", "废液", "处理"],
         "constraints": ["proper_waste_disposal"]},
    ],
    ExperimentType.SOLUTION_PREPARATION: [
        {"step_id": "prepare_ppe", "name_zh": "穿戴个人防护装备", "name_en": "Wear PPE",
         "keywords": ["glove", "goggles", "lab coat", "ppe", "防护"],
         "constraints": ["must_have_gloves"]},
        {"step_id": "calculate_amount", "name_zh": "计算用量", "name_en": "Calculate amount",
         "keywords": ["calculate", "formula", "molar", "计算", "公式"],
         "constraints": []},
        {"step_id": "weigh_solute", "name_zh": "称量溶质", "name_en": "Weigh solute",
         "keywords": ["balance", "weigh", "mass", "天平", "称量", "质量"],
         "constraints": ["balance_calibration_check"]},
        {"step_id": "dissolve_solute", "name_zh": "溶解溶质", "name_en": "Dissolve solute",
         "keywords": ["dissolve", "stir", "solvent", "溶解", "搅拌", "溶剂"],
         "constraints": []},
        {"step_id": "transfer_to_flask", "name_zh": "转移至容量瓶", "name_en": "Transfer to volumetric flask",
         "keywords": ["transfer", "volumetric flask", "funnel", "转移", "容量瓶"],
         "constraints": ["quantitative_transfer"]},
        {"step_id": "make_up_to_mark", "name_zh": "定容", "name_en": "Make up to mark",
         "keywords": ["mark", "dilute to", "meniscus", "刻度线", "定容"],
         "constraints": ["meniscus_at_mark"]},
        {"step_id": "mix_homogeneously", "name_zh": "混匀", "name_en": "Mix homogeneously",
         "keywords": ["invert", "mix", "shake", "倒转", "混匀"],
         "constraints": []},
        {"step_id": "label_store", "name_zh": "标记并储存", "name_en": "Label and store",
         "keywords": ["label", "concentration", "date", "标记", "浓度"],
         "constraints": ["must_label"]},
    ],
    ExperimentType.PIPETTING: [
        {"step_id": "prepare_ppe", "name_zh": "穿戴个人防护装备", "name_en": "Wear PPE",
         "keywords": ["glove", "goggles", "ppe", "防护"],
         "constraints": ["must_have_gloves"]},
        {"step_id": "select_pipette", "name_zh": "选择移液器", "name_en": "Select pipette",
         "keywords": ["pipette", "volume", "select", "移液器", "量程"],
         "constraints": ["correct_volume_range"]},
        {"step_id": "attach_tip", "name_zh": "安装吸头", "name_en": "Attach tip",
         "keywords": ["tip", "attach", "吸头", "安装"],
         "constraints": ["tip_must_be_secure"]},
        {"step_id": "aspirate", "name_zh": "吸液", "name_en": "Aspirate",
         "keywords": ["aspirate", "draw", "吸液", "吸取"],
         "constraints": ["no_bubbles", "slow_aspiration"]},
        {"step_id": "dispense", "name_zh": "排液", "name_en": "Dispense",
         "keywords": ["dispense", "release", "排液", "排出"],
         "constraints": ["touch_tip", "complete_dispense"]},
        {"step_id": "eject_tip", "name_zh": "退吸头", "name_en": "Eject tip",
         "keywords": ["eject", "waste", "退吸头", "废弃"],
         "constraints": ["proper_tip_disposal"]},
    ],
}

# Default template for unknown experiment types
DEFAULT_STEP_TEMPLATE = [
    {"step_id": "prepare_ppe", "name_zh": "穿戴个人防护装备", "name_en": "Wear PPE",
     "keywords": ["glove", "goggles", "lab coat", "ppe", "防护"],
     "constraints": ["must_have_gloves", "must_have_goggles"]},
    {"step_id": "prepare_materials", "name_zh": "准备实验材料", "name_en": "Prepare materials",
     "keywords": ["prepare", "setup", "准备", "布置"],
     "constraints": []},
    {"step_id": "execute_operation", "name_zh": "执行操作", "name_en": "Execute operation",
     "keywords": ["perform", "execute", "操作", "执行"],
     "constraints": []},
    {"step_id": "observe_record", "name_zh": "观察并记录", "name_en": "Observe and record",
     "keywords": ["observe", "record", "观察", "记录"],
     "constraints": []},
    {"step_id": "clean_workspace", "name_zh": "清洁工作台", "name_en": "Clean workspace",
     "keywords": ["clean", "wash", "清洁", "清洗"],
     "constraints": []},
    {"step_id": "dispose_waste", "name_zh": "处理废弃物", "name_en": "Dispose waste",
     "keywords": ["waste", "dispose", "废液", "处理"],
     "constraints": ["proper_waste_disposal"]},
]


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ChemistryEntity:
    """A recognized chemistry container, tool, or reagent."""
    entity_type: str  # "container", "tool", "reagent"
    name: str  # e.g., "beaker", "pipette", "hcl"
    name_zh: str  # Chinese name
    confidence: float = 0.0
    bbox: Optional[List[int]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)  # e.g., {"volume_ml": 250, "contains_liquid": True}


@dataclass
class SceneProfile:
    """Complete scene understanding result for an experiment video.

    When AI is available, this is populated by VLM analysis of initial frames.
    Without AI, rule-based fallback provides a best-effort classification.
    """
    experiment_type: ExperimentType
    experiment_type_zh: str
    confidence: float  # 0.0 - 1.0

    # Expected workflow
    expected_steps: List[Dict[str, Any]]  # From EXPERIMENT_STEP_TEMPLATES
    expected_step_ids: List[str]  # Ordered step_id list for step checker

    # Detected entities
    containers: List[ChemistryEntity] = field(default_factory=list)
    tools: List[ChemistryEntity] = field(default_factory=list)
    reagents: List[ChemistryEntity] = field(default_factory=list)

    # Safety constraints specific to this experiment
    safety_constraints: List[str] = field(default_factory=list)

    # Scene metadata
    scene_description_zh: str = ""
    scene_description_en: str = ""
    key_objects_of_interest: List[str] = field(default_factory=list)

    # AI readiness flag
    ai_analysis_ready: bool = False
    ai_analysis_unavailable_reason: str = "framework_only"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["experiment_type"] = self.experiment_type.value
        return d

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Rule-Based Fallback Classifier
# ---------------------------------------------------------------------------

def _keyword_match(text: str, keywords: List[str]) -> int:
    """Count keyword matches in text (case-insensitive)."""
    t = text.lower()
    return sum(1 for kw in keywords if kw.lower() in t)


def classify_experiment_by_keywords(text_summary: str) -> Tuple[ExperimentType, float]:
    """Rule-based experiment type classification from AI text summary.

    Returns (experiment_type, confidence).
    Confidence is based on keyword match density.
    """
    text = text_summary.lower()

    # Score each experiment type
    scores: Dict[ExperimentType, float] = {
        ExperimentType.ACID_BASE_TITRATION: 0.0,
        ExperimentType.SOLUTION_PREPARATION: 0.0,
        ExperimentType.PIPETTING: 0.0,
        ExperimentType.WEIGHING: 0.0,
        ExperimentType.FILTRATION: 0.0,
        ExperimentType.HEATING_REACTION: 0.0,
        ExperimentType.PH_MEASUREMENT: 0.0,
    }

    # Titration signals
    titration_kw = ["titrat", "burette", "buret", "endpoint", "indicator",
                     "滴定", "滴定管", "终点", "指示剂", "锥形瓶"]
    scores[ExperimentType.ACID_BASE_TITRATION] = min(1.0, _keyword_match(text, titration_kw) * 0.2)

    # Solution preparation signals
    prep_kw = ["volumetric flask", "dissolve", "make up to", "dilute to", "concentration",
               "容量瓶", "溶解", "定容", "稀释", "浓度", "配制"]
    scores[ExperimentType.SOLUTION_PREPARATION] = min(1.0, _keyword_match(text, prep_kw) * 0.17)

    # Pipetting signals
    pipette_kw = ["pipette", "tip", "aspirate", "dispense", "micropipette",
                  "移液", "吸头", "吸液", "排液", "移液器"]
    scores[ExperimentType.PIPETTING] = min(1.0, _keyword_match(text, pipette_kw) * 0.2)

    # Weighing signals
    weigh_kw = ["balance", "weigh", "mass", "gram", "天平", "称量", "质量", "克"]
    scores[ExperimentType.WEIGHING] = min(1.0, _keyword_match(text, weigh_kw) * 0.2)

    # Filtration signals
    filter_kw = ["filter", "filtration", "funnel", "filter paper", "过滤", "漏斗", "滤纸"]
    scores[ExperimentType.FILTRATION] = min(1.0, _keyword_match(text, filter_kw) * 0.2)

    # Heating signals
    heat_kw = ["heat", "boil", "hot plate", "bunsen", "加热", "煮沸", "酒精灯"]
    scores[ExperimentType.HEATING_REACTION] = min(1.0, _keyword_match(text, heat_kw) * 0.2)

    # pH signals
    ph_kw = ["ph", "indicator strip", "ph meter", "酸碱度", "ph计", "试纸"]
    scores[ExperimentType.PH_MEASUREMENT] = min(1.0, _keyword_match(text, ph_kw) * 0.25)

    # Find best match
    best_type = ExperimentType.GENERAL_LAB_OPERATION
    best_score = 0.0
    for etype, score in scores.items():
        if score > best_score:
            best_score = score
            best_type = etype

    # If no strong signal, default to general
    if best_score < 0.15:
        return ExperimentType.GENERAL_LAB_OPERATION, 0.3

    return best_type, best_score


def build_scene_profile_from_text(
    text_summary: str,
    existing_analysis: Optional[Dict[str, Any]] = None,
) -> SceneProfile:
    """Build a SceneProfile from AI text summary (or fallback to keyword matching).

    This is the main entry point for scene understanding.
    When AI is available, text_summary comes from VLM analysis of initial frames.
    Without AI, it comes from local visual analysis fallback text.
    """
    exp_type, confidence = classify_experiment_by_keywords(text_summary)

    # Get expected steps for this experiment type
    steps = EXPERIMENT_STEP_TEMPLATES.get(exp_type, DEFAULT_STEP_TEMPLATE)
    step_ids = [s["step_id"] for s in steps]

    # Chinese name mapping
    zh_names = {
        ExperimentType.ACID_BASE_TITRATION: "酸碱滴定",
        ExperimentType.REDOX_TITRATION: "氧化还原滴定",
        ExperimentType.SOLUTION_PREPARATION: "溶液配制",
        ExperimentType.SERIAL_DILUTION: "系列稀释",
        ExperimentType.PIPETTING: "移液操作",
        ExperimentType.WEIGHING: "称量操作",
        ExperimentType.FILTRATION: "过滤操作",
        ExperimentType.EXTRACTION: "萃取操作",
        ExperimentType.CRYSTALLIZATION: "结晶操作",
        ExperimentType.HEATING_REACTION: "加热反应",
        ExperimentType.PH_MEASUREMENT: "pH测量",
        ExperimentType.COLORIMETRY: "比色分析",
        ExperimentType.TITRATION_GENERAL: "滴定实验",
        ExperimentType.GENERAL_LAB_OPERATION: "一般实验室操作",
        ExperimentType.UNKNOWN: "未知实验",
    }

    # Safety constraints per experiment type
    safety_map = {
        ExperimentType.ACID_BASE_TITRATION: [
            "must_have_gloves", "must_have_goggles",
            "slow_addition_near_endpoint", "proper_acid_handling",
            "eye_level_reading", "proper_waste_disposal",
        ],
        ExperimentType.SOLUTION_PREPARATION: [
            "must_have_gloves", "quantitative_transfer",
            "balance_calibration_check", "proper_labeling",
        ],
        ExperimentType.PIPETTING: [
            "must_have_gloves", "no_bubbles",
            "slow_aspiration", "proper_tip_disposal",
        ],
        ExperimentType.HEATING_REACTION: [
            "must_have_gloves", "must_have_goggles",
            "no_flammable_near_heat", "proper_heat_control",
        ],
    }

    # Key objects of interest per experiment type
    objects_map = {
        ExperimentType.ACID_BASE_TITRATION: ["burette", "erlenmeyer_flask", "indicator", "clamp_stand"],
        ExperimentType.SOLUTION_PREPARATION: ["volumetric_flask", "balance", "wash_bottle", "funnel"],
        ExperimentType.PIPETTING: ["pipette", "pipette_tips", "sample_container", "waste_container"],
        ExperimentType.WEIGHING: ["balance", "weight_boat", "spatula", "desiccator"],
        ExperimentType.FILTRATION: ["funnel", "filter_paper", "flask", "glass_rod"],
        ExperimentType.HEATING_REACTION: ["hot_plate", "beaker", "thermometer", "stir_bar"],
    }

    return SceneProfile(
        experiment_type=exp_type,
        experiment_type_zh=zh_names.get(exp_type, "一般实验室操作"),
        confidence=confidence,
        expected_steps=steps,
        expected_step_ids=step_ids,
        safety_constraints=safety_map.get(exp_type, ["must_have_gloves", "must_have_goggles"]),
        scene_description_zh=f"基于文本分析识别为{zh_names.get(exp_type, '一般实验室操作')}场景。",
        scene_description_en=f"Identified as {exp_type.value.replace('_', ' ')} scene via text analysis.",
        key_objects_of_interest=objects_map.get(exp_type, ["general_equipment"]),
        ai_analysis_ready=False,
        ai_analysis_unavailable_reason="rule_based_fallback",
    )


def build_scene_profile_from_analysis(
    ai_analysis: Dict[str, Any],
) -> SceneProfile:
    """Build SceneProfile from AI/VLM analysis result.

    When AI becomes available, call this with the VLM response.
    The VLM should return structured JSON with:
    - experiment_type
    - containers_detected
    - tools_detected
    - reagents_detected
    - expected_steps
    - safety_notes
    """
    exp_type_str = str(ai_analysis.get("experiment_type", "general_lab_operation")).lower()
    try:
        exp_type = ExperimentType(exp_type_str)
    except ValueError:
        exp_type = ExperimentType.GENERAL_LAB_OPERATION

    steps = EXPERIMENT_STEP_TEMPLATES.get(exp_type, DEFAULT_STEP_TEMPLATE)
    step_ids = [s["step_id"] for s in steps]

    zh_names = {
        ExperimentType.ACID_BASE_TITRATION: "酸碱滴定",
        ExperimentType.SOLUTION_PREPARATION: "溶液配制",
        ExperimentType.PIPETTING: "移液操作",
        ExperimentType.GENERAL_LAB_OPERATION: "一般实验室操作",
    }

    containers = []
    for c in ai_analysis.get("containers_detected", []):
        containers.append(ChemistryEntity(
            entity_type="container",
            name=str(c.get("name", "unknown")),
            name_zh=str(c.get("name_zh", c.get("name", "未知"))),
            confidence=float(c.get("confidence", 0.5)),
            bbox=c.get("bbox"),
            attributes=c.get("attributes", {}),
        ))

    tools = []
    for t in ai_analysis.get("tools_detected", []):
        tools.append(ChemistryEntity(
            entity_type="tool",
            name=str(t.get("name", "unknown")),
            name_zh=str(t.get("name_zh", t.get("name", "未知"))),
            confidence=float(t.get("confidence", 0.5)),
            bbox=t.get("bbox"),
        ))

    reagents = []
    for r in ai_analysis.get("reagents_detected", []):
        reagents.append(ChemistryEntity(
            entity_type="reagent",
            name=str(r.get("name", "unknown")),
            name_zh=str(r.get("name_zh", r.get("name", "未知"))),
            confidence=float(r.get("confidence", 0.5)),
        ))

    return SceneProfile(
        experiment_type=exp_type,
        experiment_type_zh=zh_names.get(exp_type, exp_type.value.replace("_", " ")),
        confidence=float(ai_analysis.get("confidence", 0.7)),
        expected_steps=steps,
        expected_step_ids=step_ids,
        containers=containers,
        tools=tools,
        reagents=reagents,
        safety_constraints=ai_analysis.get("safety_notes", []),
        scene_description_zh=ai_analysis.get("description_zh", ""),
        scene_description_en=ai_analysis.get("description_en", ""),
        key_objects_of_interest=ai_analysis.get("key_objects", []),
        ai_analysis_ready=True,
        ai_analysis_unavailable_reason="",
    )
