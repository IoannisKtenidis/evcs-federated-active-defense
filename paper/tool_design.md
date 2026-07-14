# Σχεδιασμός & Αρχιτεκτονική του Ενοποιημένου Εργαλείου (Fed-OCPP-Secure)

Αυτό το έγγραφο περιγράφει τη σχεδίαση, τις μαθηματικές εξισώσεις, τις στρατηγικές και τη δομή του ενοποιημένου εργαλείου **Fed-OCPP-Secure** (όπως διαμορφώθηκε μετά την εφαρμογή των Blueprints των εγγράφων «Ασφαλής Ομόσπονδη Μάθηση» και «Σχεδιασμός Federated Learning Testbed»).

---

## 1. Αρχιτεκτονικά Στοιχεία του Εργαλείου

Το εργαλείο αποτελεί ένα ολοκληρωμένο, αυτοματοποιημένο σύστημα **ενεργού άμυνας (active defense loop)** για δίκτυα OCPP (Electric Vehicle Charging Stations) και αποτελείται από τέσσερα βασικά επίπεδα:

```mermaid
graph TD
    subgraph n8n [Orchestrator Level]
        orchestrator[n8n Threat Mitigation Workflow]
    end

    subgraph AISAP [Adversarial Simulation Level]
        generator[AISAP API / ART Engine]
    end

    subgraph FIDS [Federated Retraining Level]
        Server[FIDS Aggregator Server]
        Client1[fids-node-1 (Client 1)]
        Client2[fids-node-2 (Client 2)]
        
        Server -- 1. gRPC FL Round --> Client1
        Server -- 1. gRPC FL Round --> Client2
        Client1 -- 2. Weight Upload --> Server
        Client2 -- 2. Weight Upload --> Server
    end

    %% Webhook Alerts
    Attacker((Attacker)) -- Evasion Attack --> Client1
    Client1 -- 3. compromised webhook --> orchestrator
    
    %% Orchestration Loop
    orchestrator -- 4. Trigger Evasion Dataset --> generator
    generator -- 5. Write adversarial_dataset.csv --> SharedVol[(Shared Volume /results)]
    orchestrator -- 6. Trigger /retrain --> Server
    
    %% Retraining Access
    Client1 -- 7. Load local partition + Adv data --> SharedVol
    Client2 -- 7. Load local partition + Adv data --> SharedVol
```

---

## 2. Μαθηματικές Φόρμουλες & Υλοποιημένες Τεχνικές

### Α. FedProx Proximal Regularization
Για την αντιμετώπιση του **Client Drift** που προκαλείται από Non-IID δεδομένα στα δίκτυα OCPP (διαφορετικά domains φορτιστών), ενσωματώθηκε ένας εγγύς όρος (proximal penalty term) στην τοπική συνάρτηση απώλειας (loss function):

$$L_{local}(w) = L_{base}(w) + \frac{\mu}{2} \sum_{l=1}^{L} \| w_l - w_l^{global} \|_2^2$$

Όπου:
* $L_{base}(w)$ είναι η τυπική Categorical Crossentropy απώλεια.
* $w_l$ είναι οι τοπικές εκπαιδευόμενες παράμετροι (βάρη) του layer $l$.
* $w_l^{global}$ είναι οι παγκόσμιες παράμετροι που ελήφθησαν από τον Server στην αρχή του γύρου.
* $\mu$ είναι η ρυθμιστική παράμετρος (`proximal_mu`), η οποία ελέγχει την ένταση του περιορισμού για να αποτρέψει την υπερβολική απόκλιση του client.

### Β. Τοπική Εχθρική Εκπαίδευση (Local Adversarial Training - AT)
Για τη θωράκιση του μοντέλου Keras έναντι evasion attacks (adversarial examples), κάθε client παράγει τοπικά εχθρικά δείγματα μέσω της βιβλιοθήκης **ART** και της μεθόδου **FGSM (Fast Gradient Sign Method)**:

$$X_{adv} = X + \epsilon \cdot \text{sign}(\nabla_X L(w, X, y))$$

Στη συνέχεια, τα δεδομένα εκπαίδευσης συνθέτονται σε αναλογία **50-50 (Robust Training Ratio)** για να διατηρηθεί υψηλή η ακρίβεια σε καθαρά δεδομένα:

$$X_{robust} = [X, X_{adv}], \quad y_{robust} = [y, y]$$

### Γ. Robust Aggregation Strategies
Ο server.py υποστηρίζει τρεις εναλλακτικές μεθόδους συγχώνευσης των βαρών:
1. **FedAvg (Baseline)**: Απλός σταθμισμένος μέσος όρος.
2. **FedProx**: Διαχειρίζεται το Non-IID statistical drift.
3. **FedMedian (Coordinate-wise Median)**: Αντιστέκεται σε Byzantine επιθέσεις (model poisoning), καθώς μεμονωμένες ακραίες τιμές που στέλνουν κακόβουλοι ή παραβιασμένοι κόμβοι φιλτράρονται αυτόματα μέσω του υπολογισμού της διαμέσου.

---

## 3. Λειτουργικότητα & Διαχείριση Διεργασιών Web Server

Για την αποτροπή σφαλμάτων **Port Already in Use (θύρα 8080)** και τη διατήρηση της σταθερότητας σε Docker περιβάλλοντα, ο server.py ενσωματώνει:
* **`psutil` Cleanup**: Πριν από κάθε έναρξη εκπαίδευσης, ο server σαρώνει τις συνδέσεις δικτύου και τερματίζει αναδρομικά οποιαδήποτε ορφανή διεργασία (parent και children) δεσμεύει τη θύρα 8080.
* **`multiprocessing` με Spawn**: Ο Flower Server εκτελείται σε απομονωμένη διεργασία (`multiprocessing.Process`) με τη μέθοδο `spawn` για την αποφυγή συγκρούσεων (deadlocks) με το TensorFlow, ενώ πραγματοποιείται σωστό `join`/`kill` και reaping για την αποφυγή **zombie/defunct** διεργασιών.
* **Aggregated Model Saving Callback**: Μετά το πέρας κάθε γύρου FL, ο server ανακτά τα συνενωμένα βάρη και ενημερώνει αυτόματα το κεντρικό αρχείο Keras μοντέλου `/shared_aisap/model/OCPP_model.h5`.

---

## 4. Δομή Δεδομένων & pre-processing

* **Dataset**: Χρησιμοποιείται το **OCPPFlow_meter** dataset με **55 στήλες** (22 στήλες στο αρχικό clean dataset των clients).
* **Dropped Columns**: Οι στήλες `['flow_id', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'timestamp']` αφαιρούνται ως μη πληροφοριακές.
* **Target Classes**: Υπάρχουν **5 κλάσεις** (4 είδη επιθέσεων OCPP + 1 normal κατάσταση), οι οποίες κωδικοποιούνται μέσω του `encoder_ocpp.joblib` και μετατρέπονται σε one-hot vectors.
* **Feature Scaling**: Εφαρμόζεται ο StandardScaler `scaler_ocpp.joblib` παράγοντας εισόδους shape `(None, 49)`.
