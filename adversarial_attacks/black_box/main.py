import json
import sys
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    config_path = 'black_box/config.json'
    config = load_config(config_path)

    type_of_attack = [a.lower() for a in config.get('type_of_attack', [])]

    # Any of these indicates we should use the black_box_attacks (your unified black-box module)
    black_box_names = {
        'zoo', 'hopskipjump', 'hsj', 'hop-skip-jump',
        'boundary', 'boundaryattack', 'ba',
        'square', 'square_attack', 'squareattack',
        'nes'
    }

    if any(a in black_box_names for a in type_of_attack):
        print(f"Black-box attack(s) selected: {type_of_attack}")
        import black_box_attacks  # <- renamed file
        black_box_attacks.main(config)
    else:
        print(f"Using {type_of_attack} attack(s) via black_box_attack_other.py")
        import black_box_attack_other
        black_box_attack_other.main(config)

if __name__ == "__main__":
    main()
