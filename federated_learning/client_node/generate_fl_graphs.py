import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import tensorflow as tf

def plot_attack_comparisons():
    # Δεδομένα από τις μετρικές (Pre-FL vs Post-FL)
    data = {
        'Attack Type': ['FGSM', 'FGSM', 'PGD', 'PGD', 'BIM', 'BIM'],
        'Model Phase': ['Pre-FL (Weak)', 'Post-FL (Robust)', 'Pre-FL (Weak)', 'Post-FL (Robust)', 'Pre-FL (Weak)', 'Post-FL (Robust)'],
        'Accuracy (%)': [87.5, 98.2, 99.51, 99.63, 99.51, 99.63]
    }
    
    df = pd.DataFrame(data)
    
    # Ρύθμιση στυλ με seaborn
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Δημιουργία Bar Plot
    ax = sns.barplot(
        data=df, 
        x='Attack Type', 
        y='Accuracy (%)', 
        hue='Model Phase',
        palette=['#e74c3c', '#2ecc71'] # Κόκκινο για Weak, Πράσινο για Robust
    )
    
    # Προσθήκη τίτλων
    plt.title('AMYNA-TN Scenario 2: FL Robustness vs Adversarial Attacks', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Adversarial Attack Type', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(80, 100) # Όριο για να φαίνονται καλύτερα οι διαφορές
    
    # Προσθήκη αριθμών πάνω στις μπάρες
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f') + '%', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontweight='bold')
    
    plt.legend(title='Defense Phase', loc='lower right')
    plt.tight_layout()
    
    # Αποθήκευση του γραφήματος
    output_path = 'FL_Robustness_Comparison.png'
    plt.savefig(output_path, dpi=300)
    print(f"✅ Το γράφημα αποθηκεύτηκε επιτυχώς στο: {output_path}")

def test_fl_model_loading():
    print("\n--- 🛠️ Έλεγχος Μοντέλων Federated Learning ---")
    model_paths = [
        "robust_model_fids-node-1.h5",
        "robust_model_fids-node-2.h5"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                # Φόρτωση του μοντέλου για να επιβεβαιώσουμε ότι είναι λειτουργικό
                model = tf.keras.models.load_model(path, compile=False)
                input_shape = model.input_shape
                output_shape = model.output_shape
                print(f"✅ Επιτυχία! Το μοντέλο '{path}' φορτώθηκε σωστά.")
                print(f"   🔹 Input Shape: {input_shape} | Output Shape: {output_shape}")
            except Exception as e:
                print(f"❌ Σφάλμα φόρτωσης του '{path}': {e}")
        else:
            print(f"⚠️ Προειδοποίηση: Το αρχείο '{path}' δεν βρέθηκε στον τρέχοντα φάκελο.")

if __name__ == "__main__":
    # Απόκρυψη TensorFlow Warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    
    print("==================================================")
    print("🛡️ AMYNA-TN FL Model Tester & Visualizer 🛡️")
    print("==================================================")
    
    # 1. Έλεγχος ότι τα FL μοντέλα τρέχουν
    test_fl_model_loading()
    
    # 2. Παραγωγή γραφημάτων
    print("\n--- 📊 Δημιουργία Γραφημάτων ---")
    plot_attack_comparisons()
    
    print("\n✅ Ολοκληρώθηκε!")
