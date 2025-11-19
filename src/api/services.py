def bayesian_score(prior: float, prior_weight: int, ratings_sum: float, ratings_count: int) -> float:
    #If no ratings, return the prior
    if ratings_count == 0:
        return prior
    # Otherwise, compute the Bayesian average
    return (prior_weight * prior + ratings_sum) / (prior_weight + ratings_count)

def apply_feedback(exhibit, *, new_rating: int, new_view_seconds: int, alpha: float = 0.2):
    # ratings
    exhibit.ratings_sum += new_rating
    exhibit.ratings_count += 1
    exhibit.score = bayesian_score(exhibit.prior, exhibit.prior_weight, exhibit.ratings_sum, exhibit.ratings_count)
    # view time EMA (minutes)
    new_min = new_view_seconds / 60.0
    if exhibit.avg_view_time_min is None:
        exhibit.avg_view_time_min = new_min
    else:
        exhibit.avg_view_time_min = alpha * new_min + (1 - alpha) * exhibit.avg_view_time_min
    exhibit.view_time_count += 1
    return exhibit
