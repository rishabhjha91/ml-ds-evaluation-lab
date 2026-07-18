"""Data-science analytics on a synthetic product-events table.

Demonstrates SQL analytics with DuckDB (conversion funnel, cohort retention,
data-quality checks) plus a two-proportion z-test for A/B evaluation. All data
is synthetic and reproducible via a fixed seed.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from scipy import stats


def make_events(n_users: int = 5000, random_state: int = 7) -> pd.DataFrame:
    """Generate a synthetic funnel/experiment events table.

    Columns: user_id, group (control/variant), cohort_month, visited,
    signed_up, purchased. The variant has a small, genuine uplift.
    """
    rng = np.random.default_rng(random_state)
    group = rng.choice(["control", "variant"], size=n_users, p=[0.5, 0.5])

    signup_p = np.where(group == "variant", 0.42, 0.38)
    signed_up = rng.random(n_users) < signup_p

    purchase_p = np.where(group == "variant", 0.24, 0.20)
    purchased = signed_up & (rng.random(n_users) < purchase_p)

    cohort_month = rng.choice(["2025-01", "2025-02", "2025-03"], size=n_users)

    return pd.DataFrame(
        {
            "user_id": np.arange(n_users),
            "group": group,
            "cohort_month": cohort_month,
            "visited": True,
            "signed_up": signed_up,
            "purchased": purchased,
        }
    )


def conversion_funnel(events: pd.DataFrame) -> pd.DataFrame:
    """Visitors -> signups -> purchases, with step conversion rates (via SQL)."""
    con = duckdb.connect()
    con.register("events", events)
    query = """
        WITH counts AS (
            SELECT
                COUNT(*)                                    AS visitors,
                SUM(CASE WHEN signed_up THEN 1 ELSE 0 END)  AS signups,
                SUM(CASE WHEN purchased THEN 1 ELSE 0 END)  AS purchases
            FROM events
        )
        SELECT
            visitors,
            signups,
            purchases,
            ROUND(signups::DOUBLE  / NULLIF(visitors, 0), 4) AS visit_to_signup,
            ROUND(purchases::DOUBLE / NULLIF(signups, 0), 4) AS signup_to_purchase,
            ROUND(purchases::DOUBLE / NULLIF(visitors, 0), 4) AS overall_conversion
        FROM counts
    """
    return con.execute(query).df()


def cohort_retention(events: pd.DataFrame) -> pd.DataFrame:
    """Per-cohort signup and purchase rates (via SQL)."""
    con = duckdb.connect()
    con.register("events", events)
    query = """
        SELECT
            cohort_month,
            COUNT(*)                                                       AS users,
            ROUND(AVG(CASE WHEN signed_up THEN 1 ELSE 0 END), 4)          AS signup_rate,
            ROUND(AVG(CASE WHEN purchased THEN 1 ELSE 0 END), 4)         AS purchase_rate
        FROM events
        GROUP BY cohort_month
        ORDER BY cohort_month
    """
    return con.execute(query).df()


def ab_test(events: pd.DataFrame, metric: str = "purchased") -> dict:
    """Two-proportion z-test comparing control vs variant on a binary metric."""
    control = events.loc[events["group"] == "control", metric].astype(int)
    variant = events.loc[events["group"] == "variant", metric].astype(int)

    n1, n2 = len(control), len(variant)
    x1, x2 = int(control.sum()), int(variant.sum())
    p1, p2 = x1 / n1, x2 / n2

    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    return {
        "metric": metric,
        "n_control": n1,
        "n_variant": n2,
        "control_rate": round(p1, 4),
        "variant_rate": round(p2, 4),
        "absolute_lift": round(p2 - p1, 4),
        "relative_lift": round((p2 - p1) / p1, 4) if p1 else None,
        "z_statistic": round(float(z), 4),
        "p_value": round(p_value, 4),
        "significant_at_0_05": bool(p_value < 0.05),
    }


def data_quality_checks(events: pd.DataFrame) -> dict:
    """Basic data-quality gates that a real pipeline would enforce."""
    valid_groups = {"control", "variant"}
    funnel_violations = int(((~events["signed_up"]) & (events["purchased"])).sum())
    null_user_ids = int(events["user_id"].isna().sum())
    duplicate_user_ids = int(events["user_id"].duplicated().sum())
    invalid_groups = int((~events["group"].isin(valid_groups)).sum())
    return {
        "null_user_ids": null_user_ids,
        "duplicate_user_ids": duplicate_user_ids,
        "invalid_groups": invalid_groups,
        "funnel_order_violations": funnel_violations,
        "passed": bool(
            funnel_violations == 0
            and duplicate_user_ids == 0
            and invalid_groups == 0
            and null_user_ids == 0
        ),
    }


def run_all(n_users: int = 5000, random_state: int = 7) -> dict:
    """Convenience runner returning every analytics result as plain objects."""
    events = make_events(n_users=n_users, random_state=random_state)
    return {
        "funnel": conversion_funnel(events).to_dict(orient="records")[0],
        "cohorts": cohort_retention(events).to_dict(orient="records"),
        "ab_test": ab_test(events),
        "data_quality": data_quality_checks(events),
    }
