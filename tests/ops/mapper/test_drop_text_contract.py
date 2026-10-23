"""Regression tests for source-text removal in LLM extraction mappers."""

from unittest.mock import patch

import pytest
from datasets import Dataset

from data_juicer.core.data.dj_dataset import NestedDataset, nested_obj_factory
from data_juicer.ops.mapper import extract_event_mapper, extract_keyword_mapper, extract_support_text_mapper
from data_juicer.ops.mapper.drop_text import remove_text_from_dataset, remove_text_from_sample
from data_juicer.utils.constant import Fields


@pytest.fixture
def keyword_op():
    with patch.object(extract_keyword_mapper, "prepare_model", return_value="fake"), patch.object(
        extract_keyword_mapper, "get_model", return_value=lambda *args, **kwargs: '("content_keywords" "Alice, Beijing")'
    ):
        yield extract_keyword_mapper.ExtractKeywordMapper(api_model="fake", drop_text=True, try_num=1)


@pytest.mark.parametrize("text_key", ["text", "payload.text"])
@pytest.mark.parametrize("existing", [False, True])
def test_drop_text_direct_and_dataset(keyword_op, text_key, existing, tmp_path):
    keyword_op.text_key = text_key
    row = {"other": "keep", Fields.meta: {}}
    if text_key == "text":
        row["text"] = "Alice visited Beijing"
    else:
        row["payload"] = {"text": "Alice visited Beijing", "other": "nested keep"}
    if existing:
        row[Fields.meta]["keyword"] = ["already extracted"]

    direct = keyword_op.process_single(nested_obj_factory(row))
    assert direct["other"] == "keep"
    assert "text" not in (direct if text_key == "text" else direct["payload"])
    assert direct[Fields.meta]["keyword"]

    processed = keyword_op.run(NestedDataset(Dataset.from_list([row])))
    result = processed[0]
    assert result["other"] == "keep"
    assert "text" not in (result if text_key == "text" else result["payload"])
    if text_key != "text":
        assert result["payload"]["other"] == "nested keep"
    assert result[Fields.meta]["keyword"]
    output = tmp_path / "output.jsonl"
    processed.to_json(str(output))
    assert "Alice visited Beijing" not in output.read_text()


def test_drop_text_false_keeps_source(keyword_op):
    keyword_op.drop_text = False
    row = {"text": "Alice visited Beijing", Fields.meta: {}}
    result = keyword_op.run(NestedDataset(Dataset.from_list([row])))[0]
    assert result["text"] == row["text"]


@pytest.mark.parametrize("text_key", ["text", "payload.text"])
def test_drop_text_after_failed_model_attempts(keyword_op, text_key):
    keyword_op.text_key = text_key
    with patch.object(extract_keyword_mapper, "get_model", return_value=lambda *args, **kwargs: "invalid response"):
        row = {Fields.meta: {}, "text": "secret"} if text_key == "text" else {
            Fields.meta: {}, "payload": {"text": "secret", "other": "keep"}
        }
        result = keyword_op.run(NestedDataset(Dataset.from_list([row])))[0]
    assert "text" not in (result if text_key == "text" else result["payload"])
    assert "keyword" in result[Fields.meta]


def test_drop_text_matches_literal_first_nested_lookup():
    row = {"payload.text": "literal", "payload": {"text": "nested", "other": "keep"}}
    assert remove_text_from_sample(row.copy(), "payload.text")["payload"]["text"] == "nested"
    result = remove_text_from_dataset(NestedDataset(Dataset.from_list([row])), "payload.text")[0]
    assert result["payload"] == {"text": "nested", "other": "keep"}
    assert "payload.text" not in result


def test_drop_text_matches_nested_lookup_with_competing_prefixes():
    row = {"a.b": {"c": "keep"}, "a": {"b": {"c": "secret", "other": "keep"}}}
    assert nested_obj_factory(row)["a.b.c"] == "secret"
    sample = remove_text_from_sample(row, "a.b.c")
    assert sample["a.b"] == {"c": "keep"}
    assert sample["a"]["b"] == {"other": "keep"}
    result = remove_text_from_dataset(NestedDataset(Dataset.from_list([row])), "a.b.c")[0]
    assert result["a.b"] == {"c": "keep"}
    assert result["a"]["b"] == {"other": "keep"}


def test_drop_text_preserves_multiple_rows_and_nested_siblings(keyword_op):
    keyword_op.text_key = "payload.text"
    rows = [
        {"payload": {"text": "Alice visited Beijing", "other": "first"}, Fields.meta: {}},
        {"payload": {"text": "Alice visited Beijing", "other": "second"}, Fields.meta: {}},
    ]
    result = keyword_op.run(NestedDataset(Dataset.from_list(rows)))
    assert [row["payload"] for row in result] == [{"other": "first"}, {"other": "second"}]
    assert all(row[Fields.meta]["keyword"] for row in result)


def test_drop_text_invalid_input_support_mapper():
    with patch.object(extract_support_text_mapper, "prepare_model", return_value="fake"):
        op = extract_support_text_mapper.ExtractSupportTextMapper(api_model="fake", drop_text=True)
    with patch.object(extract_support_text_mapper, "get_model", return_value=lambda *args, **kwargs: ""):
        result = op.process_single({"text": "secret", Fields.meta: {}})
    assert "text" not in result


def test_drop_text_one_to_many_event_mapper():
    with patch.object(extract_event_mapper, "prepare_model", return_value="fake"):
        op = extract_event_mapper.ExtractEventMapper(api_model="fake", drop_text=True)
    op._process_single_sample = lambda text, rank=None: (["event one", "event two"], ["A", "B"])
    rows = [{"text": "secret", "other": "keep", Fields.meta: {}}]
    result = op.run(NestedDataset(Dataset.from_list(rows)))
    assert len(result) == 2
    assert "text" not in result.column_names
    assert [row["other"] for row in result] == ["keep", "keep"]


def test_drop_text_event_mapper_nested_text():
    with patch.object(extract_event_mapper, "prepare_model", return_value="fake"):
        op = extract_event_mapper.ExtractEventMapper(api_model="fake", drop_text=True, text_key="payload.text")
    op._process_single_sample = lambda text, rank=None: (["event"], ["A"])
    rows = [{"payload": {"text": "secret", "other": "keep"}, Fields.meta: {}}]
    result = op.run(NestedDataset(Dataset.from_list(rows)))
    assert len(result) == 1
    assert result[0]["payload"] == {"other": "keep"}


def test_drop_text_nested_nullable_parent():
    rows = [{"payload": {"text": "secret", "other": "keep"}}, {"payload": None}]
    result = remove_text_from_dataset(NestedDataset(Dataset.from_list(rows)), "payload.text")
    assert [row["payload"] for row in result] == [{"other": "keep"}, None]


def test_drop_text_list_of_structs(keyword_op):
    keyword_op.text_key = "payload.text"
    row = {"payload": [{"text": "secret", "other": "keep"}], Fields.meta: {"keyword": ["existing"]}}
    direct = keyword_op.process_single(nested_obj_factory(row))
    assert direct["payload"] == [{"other": "keep"}]
    result = keyword_op.run(NestedDataset(Dataset.from_list([row])))[0]
    assert result["payload"] == [{"other": "keep"}]
