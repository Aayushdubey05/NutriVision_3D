"""Small, replaceable nutrition lookup layer; values are per 100 g."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class NutritionFacts:
    calories_kcal: float
    protein_g: float
    carbohydrates_g: float
    fat_g: float

class NutritionLookup:
    def __init__(self, records: dict[str, dict[str, float]], default_key: str | None = None) -> None:
        self.records, self.default_key = records, default_key

    def lookup(self, label: str, weight_g: float) -> NutritionFacts:
        record = self.records.get(label) or (self.records.get(self.default_key) if self.default_key else None)
        if record is None:
            raise KeyError(f"No nutrition record for '{label}'.")
        if weight_g < 0:
            raise ValueError("Weight cannot be negative.")
        factor = weight_g / 100.0
        return NutritionFacts(record["calories_kcal"] * factor, record["protein_g"] * factor,
                              record["carbohydrates_g"] * factor, record["fat_g"] * factor)
