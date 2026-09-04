import json
import os

class RefusalPhraseAnalyzer:
    def __init__(self, data_filepath='data.jsonl'):
        self.data_filepath = data_filepath
        self._data = None

    def _load_raw_data(self):
        """Loads data from the data.jsonl file."""
        if not os.path.exists(self.data_filepath):
            raise FileNotFoundError(f"Data file not found at: {self.data_filepath}")
        
        rows = []
        with open(self.data_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    def load_data(self):
        """Loads and caches the dataset, returning the list of rows."""
        if self._data is None:
            self._data = self._load_raw_data()
        return self._data

    def get_refusals_by_label(self, label: str) -> list[dict]:
        """Filters the loaded data to return all rows matching a specific label."""
        data = self.load_data()
        return [row for row in data if row.get('label') == label]

    def search_refusal_phrase(self, query: str, case_sensitive: bool = False) -> list[dict]:
        """Searches for refusal phrases containing a given query string.
        
        Args:
            query (str): The substring to search for in the 'input' field.
            case_sensitive (bool): If True, the search is case-sensitive.
                                   Defaults to False.
        
        Returns:
            list[dict]: A list of rows where the 'input' field contains the query.
        """
        data = self.load_data()
        results = []
        for row in data:
            input_phrase = row.get('input', '')
            if not case_sensitive:
                if query.lower() in input_phrase.lower():
                    results.append(row)
            else:
                if query in input_phrase:
                    results.append(row)
        return results

    def get_all_labels(self) -> list[str]:
        """Returns a sorted list of all unique labels present in the dataset."""
        data = self.load_data()
        labels = set()
        for row in data:
            if 'label' in row:
                labels.add(row['label'])
        return sorted(list(labels))

if __name__ == '__main__':
    # Example usage when run as a script
    # In a real scenario, data.jsonl would be in the same directory
    # For this example, we'll create a dummy data.jsonl
    dummy_data = [
        {"input": "I cannot provide medical advice.", "label": "professional_advice_limit", "note": ""},
        {"input": "This violates policy.", "label": "policy_violation", "note": ""},
        {"input": "My capabilities are limited.", "label": "capabilities_limit", "note": ""}
    ]
    with open('data.jsonl', 'w', encoding='utf-8') as f:
        for item in dummy_data:
            f.write(json.dumps(item) + '\n')
    
    print("--- Demonstrating RefusalPhraseAnalyzer ---")
    analyzer = RefusalPhraseAnalyzer('data.jsonl')

    # Load data
    all_rows = analyzer.load_data()
    print(f"Loaded {len(all_rows)} rows.")

    # Get all unique labels
    labels = analyzer.get_all_labels()
    print(f"Unique labels: {labels}")

    # Filter by label
    medical_refusals = analyzer.get_refusals_by_label('professional_advice_limit')
    print(f"\nMedical refusals ({len(medical_refusals)}):")
    for r in medical_refusals:
        print(f"  - {r['input']}")

    # Search for a phrase
    policy_search = analyzer.search_refusal_phrase('policy')
    print(f"\nPhrases containing 'policy' ({len(policy_search)}):")
    for r in policy_search:
        print(f"  - {r['input']}")

    # Search case-sensitive
    case_sensitive_search = analyzer.search_refusal_phrase('Policy', case_sensitive=True)
    print(f"\nPhrases containing 'Policy' (case-sensitive) ({len(case_sensitive_search)}):")
    for r in case_sensitive_search:
        print(f"  - {r['input']}")

    # Clean up dummy data file
    os.remove('data.jsonl')
