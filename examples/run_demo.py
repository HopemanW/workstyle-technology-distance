import argparse
from workstyle_distance import workstyle_technology_distance, simulate_adoption_panel, age_distance_benchmark

p=argparse.ArgumentParser(); p.add_argument("--transformer", action="store_true"); a=p.parse_args()
legacy=[
    "hardware engineers follow standardized validation procedures and reliability testing",
    "production managers emphasize process discipline, quality control, and stable routines",
]
technology=[
    "software teams iterate rapidly using experimentation, agile development, and flexible problem solving",
    "machine learning engineers emphasize creativity, model experimentation, and continuous deployment",
]
mode="transformer" if a.transformer else "tfidf"
d=workstyle_technology_distance(legacy,technology,mode=mode)
print(f"Workstyle-Technology Distance ({mode}): {d:.3f}")

df=simulate_adoption_panel()
b=age_distance_benchmark(df)
print("\nSynthetic economic benchmark")
print(b.round(4))
print("\nKey coefficient age_x_distance:", round(float(b["age_x_distance"]),4))
