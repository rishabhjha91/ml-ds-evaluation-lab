from mldseval.analytics import (
    ab_test,
    cohort_retention,
    conversion_funnel,
    data_quality_checks,
    make_events,
)


def test_funnel_is_monotonic():
    events = make_events(n_users=3000, random_state=1)
    funnel = conversion_funnel(events).iloc[0]
    assert funnel["visitors"] >= funnel["signups"] >= funnel["purchases"]
    assert 0.0 <= funnel["overall_conversion"] <= 1.0


def test_ab_test_reports_rates_and_pvalue():
    events = make_events(n_users=8000, random_state=2)
    result = ab_test(events)
    assert 0.0 <= result["control_rate"] <= 1.0
    assert 0.0 <= result["variant_rate"] <= 1.0
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["n_control"] + result["n_variant"] == 8000


def test_cohort_rows_present():
    events = make_events(n_users=2000, random_state=3)
    cohorts = cohort_retention(events)
    assert len(cohorts) >= 1
    assert {"cohort_month", "signup_rate", "purchase_rate"}.issubset(cohorts.columns)


def test_data_quality_passes_on_clean_events():
    events = make_events(n_users=1500, random_state=4)
    report = data_quality_checks(events)
    assert report["funnel_order_violations"] == 0
    assert report["passed"] is True
