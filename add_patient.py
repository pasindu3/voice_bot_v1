# Quick script to add sample patient data
# Run this in your terminal: python -c "exec(open('add_patient.py').read())"

from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Add sample patient data
sample_patient = {
    "first_name": "Pasendu",
    "last_name": "Bimsarai", 
    "dob": "1997-05-24",
    "phone_number": "+1234567890"
}

try:
    response = supabase.table('patients').insert(sample_patient).execute()
    if response.data:
        print(f"✅ Successfully added patient: {response.data[0]['first_name']} {response.data[0]['last_name']}")
        print(f"   DOB: {response.data[0]['dob']}")
        print(f"   ID: {response.data[0]['id']}")
    else:
        print("❌ Failed to add patient")
except Exception as e:
    print(f"❌ Error: {e}")
