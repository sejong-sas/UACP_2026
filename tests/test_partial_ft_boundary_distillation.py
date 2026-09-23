import torch

from scripts.partial_ft_boundary_distill import (
    id_uncertainty_distillation,
    one_sided_boundary_loss,
    InitialMagnitudeNormalizer,
)


def test_id_distillation_is_log_space_and_finite():
    teacher_epi = torch.tensor([1.0, 10.0])
    student_epi = torch.tensor([2.0, 5.0])
    teacher_ale = torch.tensor([0.5, 2.0])
    student_ale = torch.tensor([0.25, 4.0])
    value = id_uncertainty_distillation(student_epi, student_ale, teacher_epi, teacher_ale)
    expected = torch.mean((torch.log(student_epi + 1e-8) - torch.log(teacher_epi + 1e-8)) ** 2)
    expected += torch.mean((torch.log(student_ale + 1e-8) - torch.log(teacher_ale + 1e-8)) ** 2)
    assert torch.allclose(value, expected)
    assert torch.isfinite(value)


def test_boundary_loss_penalizes_only_student_confidence_increase():
    teacher = torch.tensor([1.0, 4.0, 2.0])
    student = torch.tensor([0.5, 8.0, 2.0])
    value = one_sided_boundary_loss(student, teacher)
    expected = torch.mean(torch.tensor([0.0, torch.log(torch.tensor(4.0 / 8.0)) ** 2, 0.0]))
    assert torch.allclose(value, expected)


def test_initial_magnitude_normalizer_is_fixed_after_first_observation():
    normalizer = InitialMagnitudeNormalizer()
    first = normalizer(torch.tensor(4.0))
    second = normalizer(torch.tensor(8.0))
    assert torch.equal(first, torch.tensor(1.0))
    assert torch.equal(second, torch.tensor(2.0))
    assert normalizer.initial == 4.0
