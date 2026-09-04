import pytest
import json
import os
from tool_code import RefusalPhraseAnalyzer

# Fixture to create a dummy data.jsonl file for testing
@pytest.fixture
def dummy_data_file(tmp_path):
    data = [
        {"input": "I cannot provide medical advice.", "label": "professional_advice_limit", "note": ""},
        {"input": "This violates policy regarding harmful content.", "label": "policy_violation", "note": ""},
        {"input": "My capabilities do not include real-time browsing.", "label": "capabilities_limit", "note": ""},
        {"input": "I cannot generate hate speech.", "label": "safety_hate_speech", "note": ""},
        {"input": "I am an AI and do not have opinions.", "label": "capabilities_limit", "note": ""},
        {"input": "I cannot access personal data.", "label": "data_privacy", "note": ""},
        {"input": "I cannot provide legal advice.", "label": "professional_advice_limit", "note": ""}
    ]
    filepath = tmp_path / "data.jsonl"
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    return str(filepath)

# Test loading data
def test_load_data(dummy_data_file):
    analyzer = RefusalPhraseAnalyzer(dummy_data_file)
    data = analyzer.load_data()
    assert len(data) == 7
    assert data[0]['input'] == "I cannot provide medical advice."

def test_load_data_file_not_found():
    analyzer = RefusalPhraseAnalyzer('non_existent_file.jsonl')
    with pytest.raises(FileNotFoundError):
        analyzer.load_data()

# Test filtering by label
def test_get_refusals_by_label(dummy_data_file):
    analyzer = RefusalPhraseAnalyzer(dummy_data_file)
    medical_refusals = analyzer.get_refusals_by_label('professional_advice_limit')
    assert len(medical_refusals) == 2
    assert all(r['label'] == 'professional_advice_limit' for r in medical_refusals)
    assert medical_refusals[0]['input'] == 'I cannot provide medical advice.'

def test_get_refusals_by_label_no_match(dummy_data_file):
    analyzer = RefusalPhraseAnalyzer(dummy_data_file)
    non_existent_label = analyzer.get_refusals_by_label('non_existent_label')
    assert len(non_existent_label) == 0

# Test searching phrases
def test_search_refusal_phrase_case_insensitive(dummy_data_file):
    analyzer = RefusalPhraseAnalyzer(dummy_data_file)
    search_results = analyzer.search_refusal_phrase('cannot provide')
    assert len(search_results) == 2
    assert any(r['input'] == 'I cannot provide medical advice.' for r in search_results)
    assert any(r['input'] == 'I cannot provide legal advice.' for r in search_results)

def test_search_refusal_phrase_case_sensitive(dummy_data_file):
    analyzer = RefusalPhraseAnalyzer(dummy_data_file)
    search_results = analyzer.search_refusal_phrase('Cannot', case_sensitive=True)
    assert len(search_results) == 0 # 'Cannot' is not present, only 'cannot'

    search_results_correct_case = analyzer.search_refusal_phrase('cannot', case_sensitive=True)
    assert len(search_results_correct_case) == 6
    assert all('cannot' in r['input'] for r in search_results_correct_case)

def test_search_refusal_phrase_no_match(dummy_data_file):
    analyzer = RefusalPhraseAnalyzer(dummy_data_file)
    search_results = analyzer.search_refusal_phrase('xyz_non_existent')
    assert len(search_results) == 0

def test_get_all_labels(dummy_data_file):
    analyzer = RefusalPhraseAnalyzer(dummy_data_file)
    labels = analyzer.get_all_labels()
    expected_labels = [
        'capabilities_limit',
        'data_privacy',
        'policy_violation',
        'professional_advice_limit',
        'safety_hate_speech'
    ]
    assert sorted(labels) == sorted(expected_labels)
    assert len(labels) == len(expected_labels)

# Test with empty data file
def test_empty_data_file(tmp_path):
    filepath = tmp_path / "empty_data.jsonl"
    with open(filepath, 'w') as f:
        pass # Create an empty file

    analyzer = RefusalPhraseAnalyzer(str(filepath))
    data = analyzer.load_data()
    assert len(data) == 0
    assert analyzer.get_refusals_by_label('any_label') == []
    assert analyzer.search_refusal_phrase('any_query') == []
    assert analyzer.get_all_labels() == []
