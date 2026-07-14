import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def create_attack_graphs():
    # Δεδομένα από τον Πίνακα 4 του χρήστη
    data = {
        'Attack': ['FGSM', 'FGSM', 'PGD', 'PGD', 'BIM', 'BIM', 'Carlini L2', 'Carlini L2'],
        'Phase': ['Pre-FL (Αρχικό)', 'Post-FL (Ανθεκτικό)', 'Pre-FL (Αρχικό)', 'Post-FL (Ανθεκτικό)', 
                  'Pre-FL (Αρχικό)', 'Post-FL (Ανθεκτικό)', 'Pre-FL (Αρχικό)', 'Post-FL (Ανθεκτικό)'],
        'Accuracy (%)': [66.41, 98.46, 65.80, 97.90, 66.10, 98.10, 66.80, 83.01],
        'F1-Score': [0.60, 0.98, 0.58, 0.97, 0.59, 0.98, 0.60, 0.79]
    }
    
    df = pd.DataFrame(data)
    
    sns.set_theme(style="whitegrid")
    
    # --- ΓΡΑΦΗΜΑ 1: Accuracy ---
    plt.figure(figsize=(10, 6))
    ax1 = sns.barplot(
        data=df, x='Attack', y='Accuracy (%)', hue='Phase',
        palette=['#e74c3c', '#2ecc71']
    )
    plt.title('Επίδοση Μοντέλου: Accuracy vs. Adversarial Attacks (Eps=1.0)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Τύπος Επίθεσης', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(50, 105)
    
    # Προσθήκη ετικετών (νούμερα)
    for p in ax1.patches:
        height = p.get_height()
        if height > 0:
            ax1.annotate(f'{height:.2f}%', 
                         (p.get_x() + p.get_width() / 2., height), 
                         ha='center', va='bottom', 
                         xytext=(0, 5), textcoords='offset points', fontweight='bold')
            
    plt.legend(title='Φάση Μοντέλου')
    plt.tight_layout()
    plt.savefig('/app/Accuracy_Attacks_Comparison.png', dpi=300)
    plt.close()
    
    # --- ΓΡΑΦΗΜΑ 2: F1-Score ---
    plt.figure(figsize=(10, 6))
    ax2 = sns.barplot(
        data=df, x='Attack', y='F1-Score', hue='Phase',
        palette=['#e74c3c', '#2ecc71']
    )
    plt.title('Επίδοση Μοντέλου: F1-Score vs. Adversarial Attacks (Eps=1.0)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Τύπος Επίθεσης', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.ylim(0.4, 1.1)
    
    # Προσθήκη ετικετών
    for p in ax2.patches:
        height = p.get_height()
        if height > 0:
            ax2.annotate(f'{height:.2f}', 
                         (p.get_x() + p.get_width() / 2., height), 
                         ha='center', va='bottom', 
                         xytext=(0, 5), textcoords='offset points', fontweight='bold')
            
    plt.legend(title='Φάση Μοντέλου')
    plt.tight_layout()
    plt.savefig('/app/F1Score_Attacks_Comparison.png', dpi=300)
    plt.close()
    
    print("Τα γραφήματα δημιουργήθηκαν επιτυχώς στο /app!")

if __name__ == "__main__":
    create_attack_graphs()
