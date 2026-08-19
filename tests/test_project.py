from workstyle_distance import workstyle_technology_distance, simulate_adoption_panel, age_distance_benchmark

def test_distance_bounded():
    d=workstyle_technology_distance(["stable routine process"],["creative flexible experimentation"])
    assert 0 <= d <= 1

def test_age_distance_friction_sign():
    df=simulate_adoption_panel(n_firms=1500, seed=5)
    b=age_distance_benchmark(df)
    assert b["age_x_distance"] < 0
