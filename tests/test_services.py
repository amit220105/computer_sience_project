from math import isclose
from src.api.services import bayesian_score, apply_feedback


class DummyExhibit:
    """
    Minimal stand-in for the real Exhibit model.
    We only keep the fields that services.py needs.
    """
    def __init__(
        self,
        prior: float = 3.8,
        prior_weight: int = 20,
        score: float | None = None,
        avg_view_time_min: float | None = None,
    ):
        self.prior = prior
        self.prior_weight = prior_weight
        self.ratings_sum = 0.0
        self.ratings_count = 0
        self.score = prior if score is None else score
        self.avg_view_time_min = avg_view_time_min
        self.view_time_count = 0


# ---------- bayesian_score tests ----------

def test_bayesian_score_no_ratings_returns_prior():
    prior = 3.8
    prior_weight = 20
    ratings_sum = 0.0
    ratings_count = 0

    score = bayesian_score(prior, prior_weight, ratings_sum, ratings_count)

    # with 0 ratings, the score should equal the prior
    assert isclose(score, prior, rel_tol=1e-9)


def test_bayesian_score_with_ratings_moves_towards_data():
    prior = 3.8
    prior_weight = 20
    ratings_sum = 4 + 5  # two ratings: 4 and 5
    ratings_count = 2

    score = bayesian_score(prior, prior_weight, ratings_sum, ratings_count)

    # expected = (prior_weight * prior + ratings_sum) / (prior_weight + ratings_count)
    expected = (prior_weight * prior + ratings_sum) / (prior_weight + ratings_count)

    assert isclose(score, expected, rel_tol=1e-9)
    # also, it should be between prior and the sample mean (4.5)
    assert prior <= score <= 4.5 or 4.5 <= score <= prior


# ---------- apply_feedback tests ----------

def test_apply_feedback_first_rating_sets_score_and_time():
    ex = DummyExhibit(prior=3.8, prior_weight=20, avg_view_time_min=None)

    apply_feedback(ex, new_rating=5, new_view_seconds=300)  # 5 minutes

    # ratings updated
    assert ex.ratings_count == 1
    assert ex.ratings_sum == 5

    # view time updated (first value: EMA becomes exactly new value)
    assert isclose(ex.avg_view_time_min, 300 / 60.0, rel_tol=1e-9)
    assert ex.view_time_count == 1

    # bayesian score should match the formula
    expected_score = bayesian_score(
        ex.prior,
        ex.prior_weight,
        ex.ratings_sum,
        ex.ratings_count,
    )
    assert isclose(ex.score, expected_score, rel_tol=1e-9)


def test_apply_feedback_second_rating_updates_ema_and_score():
    # start with some existing state
    ex = DummyExhibit(prior=3.8, prior_weight=20, avg_view_time_min=5.0)
    ex.ratings_sum = 8.0      # e.g. two ratings: 4 and 4
    ex.ratings_count = 2
    ex.score = bayesian_score(ex.prior, ex.prior_weight, ex.ratings_sum, ex.ratings_count)
    ex.view_time_count = 2

    # new user spends 10 minutes and rates 5
    apply_feedback(ex, new_rating=5, new_view_seconds=600)

    # ratings updated
    assert ex.ratings_count == 3
    assert ex.ratings_sum == 13.0  # 8 + 5

    # EMA check: avg_new = alpha * 10 + (1-alpha) * 5
    alpha = 0.2
    expected_avg = alpha * (600 / 60.0) + (1 - alpha) * 5.0
    assert isclose(ex.avg_view_time_min, expected_avg, rel_tol=1e-9)
    assert ex.view_time_count == 3

    # score updated with new totals
    expected_score = bayesian_score(
        ex.prior,
        ex.prior_weight,
        ex.ratings_sum,
        ex.ratings_count,
    )
    assert isclose(ex.score, expected_score, rel_tol=1e-9)
