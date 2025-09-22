# Test script to verify name extraction
from voice_app_realtime import extract_patient_info_with_gpt, extract_patient_info_fallback

# Test cases
test_cases = [
    "My name is Pasindu Dharmadasa, my date of birth is 1997 May 24",
    "My first name tree letters are P-A-S. My last name tree letters are D-H-A",
    "I am Pasindu Dharmadasa, DOB is 1997-05-24",
    "My name is Pasindu Dharmadasa and my date of birth is 1997 May 24"
]

print("Testing name extraction...")
print("=" * 50)

for i, test_case in enumerate(test_cases, 1):
    print(f"\nTest Case {i}: {test_case}")
    
    # Test GPT extraction
    dob_gpt, first_gpt, last_gpt = extract_patient_info_with_gpt(test_case)
    print(f"GPT-4o: DOB={dob_gpt}, First={first_gpt}, Last={last_gpt}")
    
    # Test fallback extraction
    dob_fb, first_fb, last_fb = extract_patient_info_fallback(test_case)
    print(f"Fallback: DOB={dob_fb}, First={first_fb}, Last={last_fb}")
    
    print("-" * 30)