import requests
import sys

def launch_attack(target_url, attack_type="fgsm"):
    # Map attack types to payloads that the n8n webhook can understand and forward to AISAP
    attack_payloads = {
        "fgsm": "Adversarial FGSM Attack",
        "pgd": "Adversarial PGD Attack",
        "bim": "Adversarial BIM Attack",
        "cw": "Adversarial Carlini-Wagner Attack"
    }
    
    payload = attack_payloads.get(attack_type.lower(), "Adversarial FGSM Attack")
    
    print(f"[*] Ξεκινάει χειροκίνητη επίθεση στο {target_url}...")
    print(f"[*] Attack Type: {attack_type.upper()}")
    print(f"[*] Payload: {payload}")
    
    try:
        response = requests.post(
            f"{target_url}/simulate_attack",
            json={"payload": payload, "attack_type": attack_type.lower()},
            timeout=10
        )
        print(f"\n[+] Αποτέλεσμα (Status Code: {response.status_code}):")
        
        try:
            # Ο επιτιθέμενος βλέπει μόνο αν το payload έγινε αποδεκτό ή όχι.
            pass
        except:
            pass
            
        if response.status_code == 400:
            print("\n[!] [FAIL] Η επιθεση πετυχε! Προσβαση στο συστημα (Status: 400)")
        elif response.status_code == 200:
            print("\n[OK] [SUCCESS] Η επιθεση αποκρουστηκε/αγνοηθηκε απο το Firewall/Defense (Status: 200)")
            
    except requests.exceptions.RequestException as e:
        print(f"[-] Σφάλμα σύνδεσης: {e}")

if __name__ == "__main__":
    target = "http://localhost:5001"
    attack = "fgsm"
    
    if len(sys.argv) > 1:
        attack = sys.argv[1]
    if len(sys.argv) > 2:
        target = sys.argv[2]
        
    launch_attack(target, attack)
