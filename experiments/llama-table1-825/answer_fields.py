"""Preserve the released answer-field fallback without nested backtracking."""

import json
import re


ANSWER_FIELD = re.compile(r'"answer"\s*:\s*"([^"]*)"')


def get_field_from_completion(completion_text, field, default=None):
    print(completion_text)
    try:
        data = json.loads(completion_text)
    except json.JSONDecodeError:
        if field != "answer":
            return default
        # The released greedy field consumes up to the first raw closing quote.
        # The nested escape repetition is redundant for any successful match.
        match = ANSWER_FIELD.search(completion_text)
        return match.group(1).encode().decode("unicode_escape") if match else default

    current = data
    for part in field.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def install():
    from src import universal_agent

    universal_agent.get_field_from_completion = get_field_from_completion
