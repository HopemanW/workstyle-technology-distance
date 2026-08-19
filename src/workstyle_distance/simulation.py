from __future__ import annotations
import numpy as np
import pandas as pd

def simulate_adoption_panel(n_firms: int = 3000, seed: int = 20260819) -> pd.DataFrame:
    """Synthetic firm-technology panel with an organizational-conflict mechanism."""
    rng=np.random.default_rng(seed)
    age=np.exp(rng.normal(2.2,0.8,n_firms))
    log_age=np.log1p(age)
    size=np.exp(rng.normal(4.2,0.9,n_firms))
    productivity=rng.normal(size=n_firms)
    tech_exposure=np.clip(rng.beta(2,2,n_firms),0,1)
    wtd=np.clip(0.12 + 0.65*rng.beta(2,2,n_firms) + 0.08*tech_exposure,0,1)
    # Older firms are penalized specifically when workstyle distance is large.
    growth=(0.08 + 0.22*tech_exposure + 0.08*productivity + 0.02*np.log1p(size)
            -0.035*log_age -0.18*wtd -0.11*log_age*wtd + rng.normal(0,0.18,n_firms))
    adoption_latent=(0.6*tech_exposure -0.35*wtd -0.25*log_age*wtd +0.15*productivity + rng.normal(0,0.45,n_firms))
    adopted=(adoption_latent>0).astype(int)
    return pd.DataFrame({"firm_age":age,"log_age":log_age,"size":size,"productivity":productivity,
                         "technology_exposure":tech_exposure,"workstyle_distance":wtd,
                         "adopted":adopted,"future_growth":growth})
