import torch

from scripts.fig11_dynamic_runtime_validation import aggregate_omitted_scores, select_next_ng


def test_epistemic_trigger_controls_next_round_not_current_round():
    decision = select_next_ng(16, ale_score=1.0, epi_score=2.0, epi_threshold=1.5,
                              ale_target=1.0, ale_delta=0.1)
    assert decision["adaptation_trigger"] is True
    assert decision["full_feedback_fallback"] is True
    assert decision["next_ng"] == 1


def test_eq13_omitted_score_averages_subcarrier_map_over_omitted_set():
    uncertainty = torch.tensor([[[1.0, 3.0, 5.0, 7.0]] * 8])
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    ale, epi = aggregate_omitted_scores(uncertainty, uncertainty * 2.0, mask)
    assert ale == 10.0
    assert epi == 20.0
