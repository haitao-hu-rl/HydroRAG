from hydrorag import __version__


def test_answer_evaluation_module_imports():
    from work.evaluate_answer_quality import (
        evaluate_fact_points,
        evaluate_numeric_facts,
        is_refusal,
    )
    from hydrorag.validation import audit_numeric_claims

    assert __version__
    assert evaluate_fact_points("覆盖1950年至2023年", [{"patterns": ["1950", "2023"]}])["score"] == 1.0
    numeric = evaluate_numeric_facts("误差为0.0403 m", [{"value": 0.0403, "unit": "m"}])
    assert numeric["value_unit_accuracy"] == 1.0
    assert evaluate_numeric_facts("覆盖1950年至2023年", [{"value": 1950, "unit": "年"}, {"value": 2023, "unit": "年"}])["value_unit_accuracy"] == 1.0
    assert is_refusal("现有资料不足，无法可靠回答。", False)
    assert is_refusal("资料未给出该参数，不能回答。", True)
    assert audit_numeric_claims("偏差为0.0403 m", "偏差为0.0403 m")["supported"]
    assert not audit_numeric_claims("偏差为0.0404 m", "偏差为0.0403 m")["supported"]
    assert evaluate_numeric_facts("流量为12 m³/s", [{"value": 12, "unit": "m3/s"}])["value_unit_accuracy"] == 1.0
    assert evaluate_numeric_facts("没有数值", [])['value_accuracy'] is None
