"""Behavioural reproduction of the evaluation in the rater presentation."""

from scripts.reproduce_rater_evaluation import run_evaluation


def test_reproduces_rater_presentations_two_main_findings():
    rows = {
        row["informative_raters"]: row
        for row in run_evaluation(num_items=100, replicates=100, seed=20_260_817)
    }

    # With five equally informative raters, the full model pays for estimating unnecessary
    # confusion parameters and majority vote is slightly better.
    assert rows[5]["majority_accuracy"] > rows[5]["ds_accuracy_when_fit_passed"]

    # With two systematic spammers, the full model's rater-specific confusion estimates help.
    assert rows[3]["ds_accuracy_when_fit_passed"] > rows[3]["majority_accuracy"]
    assert rows[5]["fit_pass_rate"] == 1.0
    assert rows[3]["fit_pass_rate"] == 1.0
