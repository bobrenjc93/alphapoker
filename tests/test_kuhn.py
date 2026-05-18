from alphapoker.kuhn import BET, CALL, CHECK, FOLD, KuhnState


def test_terminal_payoffs_for_showdown_and_fold() -> None:
    state = KuhnState.initial((2, 0)).apply(CHECK).apply(CHECK)
    assert state.winner() == 0
    assert state.utility(0) == 1.0
    assert state.utility(1) == -1.0

    state = KuhnState.initial((0, 2)).apply(BET).apply(CALL)
    assert state.winner() == 1
    assert state.utility(0) == -2.0
    assert state.utility(1) == 2.0

    state = KuhnState.initial((0, 2)).apply(CHECK).apply(BET).apply(FOLD)
    assert state.winner() == 1
    assert state.utility(0) == -1.0
    assert state.utility(1) == 1.0


def test_legal_actions_follow_kuhn_betting_tree() -> None:
    root = KuhnState.initial((0, 1))
    assert root.legal_actions() == (CHECK, BET)
    assert root.apply(CHECK).legal_actions() == (CHECK, BET)
    assert root.apply(BET).legal_actions() == (CALL, FOLD)
    assert root.apply(CHECK).apply(BET).legal_actions() == (CALL, FOLD)

