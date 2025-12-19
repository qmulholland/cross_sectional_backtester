def apply_transaction_costs(returns, cost_bps=5):
    """
    Apply linear transaction costs in basis points.
    """
    cost = cost_bps / 10000
    return returns - cost
