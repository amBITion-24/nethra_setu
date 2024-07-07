from difflib import get_close_matches

def correct_anomalies(text, correction_dict):
    corrected_text = ''.join(correction_dict.get(char, char) for char in text)
    return corrected_text

def find_best_match(detected_text, values_list):
    matches = get_close_matches(detected_text, values_list, n=1, cutoff=0.55)
    return matches[0] if matches else detected_text
