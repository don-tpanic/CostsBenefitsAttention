from scipy import stats

"""
Modulerised functions to compute metrics in
a standard way.
"""

def SDT_macmillan_correction(num_pos, hit_rate, fa_rate):
    """
    Compute d' and C based on
        one set of hit rate and fas rate
    And corrected by Macmillan approach.
    inputs:
    ------
        num_pos: number of positive examples given a target class
        hit_rate: per target class
        fa_rate: per target class
    
    returns:
    -------
        d_prime 
        c
    """
    half_hit = 0.5 / num_pos
    if hit_rate == 1:
        hit_rate_correct = 1 - half_hit
    elif hit_rate == 0:
        hit_rate_correct = half_hit
    else:
        hit_rate_correct = hit_rate

    half_fa = 0.5 / num_pos
    if fa_rate == 1:
        fa_rate_correct = 1 - half_fa
    elif fa_rate == 0:
        fa_rate_correct = half_fa
    else:
        fa_rate_correct = fa_rate

    d_prime = stats.norm.ppf(hit_rate_correct) - stats.norm.ppf(fa_rate_correct)
    c = -(stats.norm.ppf(hit_rate_correct) + stats.norm.ppf(fa_rate_correct)) / 2
    return d_prime, c
