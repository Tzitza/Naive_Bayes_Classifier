import pandas as pd                  #Εισαγωγή της βιβλιοθήκης pandas για χειρισμό δεδομένων
import math                          #Εισαγωγή της βιβλιοθήκης math για μαθηματικές συναρτήσεις
from collections import defaultdict  #Εισαγωγή defaultdict για ευκολότερη αποθήκευση παραμέτρων
from sklearn.model_selection import train_test_split  #Εισαγωγή train_test_split

class NaiveBayesClassifier:  #Ορισμός της κλάσης του ταξινομητή Naive Bayes
    def __init__(self):  #Αρχικοποίηση του αντικειμένου
        self.classes = []  #Λίστα με τις κλάσεις
        self.attr_types = {}  #Λεξικό με τους τύπους χαρακτηριστικών (διακριτό ή συνεχές)
        self.class_prob = {}  #Λεξικό με τις πιθανότητες κάθε κλάσης
        self.attr_params = defaultdict(dict)  #Λεξικό με παραμέτρους για κάθε χαρακτηριστικό και κλάση
        self.laplace_alpha = 1.0  #Παράμετρος εξομάλυνσης Laplace

    def load_data(self, file_path):  #Συνάρτηση για φόρτωση των δεδομένων από CSV
        """Φόρτωση CSV με μορφή: 1η γραμμή = όνομα,TYPE, τελευταία στήλη = κλάση."""
        df = pd.read_csv(file_path, header=None)  #Ανάγνωση CSV χωρίς κεφαλίδες
        headers = [x.strip('"') for x in df.iloc[0]]  #Αφαίρεση εισαγωγικών από την 1η γραμμή

        self.attr_types = {headers[i]: headers[i+1] for i in range(0, len(headers)-1, 2)}  #Ορισμός τύπων χαρακτηριστικών
        column_names = [headers[i] for i in range(0, len(headers)-1, 2)] + [headers[-1]]  #Ονόματα στηλών

        df = pd.read_csv(file_path, skiprows=1, header=None, names=column_names).dropna()  #Φόρτωση δεδομένων χωρίς την 1η γραμμή
        self.classes = df[headers[-1]].unique()  #Ανάκτηση μοναδικών κλάσεων
        return df  #Επιστροφή του dataframe

    def train(self, file_path):  #Συνάρτηση εκπαίδευσης μοντέλου
        """Εκπαίδευση μοντέλου: υπολογισμός P(c) και P(x|c)."""
        df = self.load_data(file_path)  #Φόρτωση δεδομένων
        class_col = df.columns[-1]  #Ορισμός της στήλης κλάσης
        
        # Διαχωρισμός δεδομένων 70-30 με stratify για ισόρροπη κατανομή κλάσεων
        X = df.drop(class_col, axis=1)  #Χαρακτηριστικά
        y = df[class_col]  #Κλάσεις
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
        
        train_df = pd.concat([X_train, y_train], axis=1)  #Ενοποίηση train δεδομένων

        self.class_prob = train_df[class_col].value_counts(normalize=True).to_dict()  #Υπολογισμός πιθανοτήτων κλάσεων

        for attr in train_df.columns[:-1]:  #Για κάθε χαρακτηριστικό (εκτός κλάσης)
            attr_values = train_df[attr].unique() if self.attr_types[attr] != "C" else None  #Τιμές για διακριτά

            for c in self.classes:  #Για κάθε κλάση
                subset = train_df[train_df[class_col] == c][attr]  #Υποσύνολο για τη συγκεκριμένη κλάση

                if self.attr_types[attr] == "C":  #Αν είναι συνεχές χαρακτηριστικό
                    mean, std = subset.mean(), subset.std()  #Υπολογισμός μέσης τιμής και τυπικής απόκλισης
                    std = max(std, 0.01)  #Αποφυγή διαίρεσης με το μηδέν
                    self.attr_params[attr][c] = {"mean": mean, "std": std}  #Αποθήκευση παραμέτρων Gaussian
                else:  #Αν είναι διακριτό χαρακτηριστικό
                    value_counts = subset.value_counts().to_dict()  #Πλήθος κάθε τιμής
                    total_count = len(subset)  #Συνολικό πλήθος
                    unique_values = len(attr_values) if attr_values is not None else len(value_counts)  #Πλήθος μοναδικών τιμών

                    #Υπολογισμός πιθανοτήτων με Laplace
                    unique_values = len(attr_values) #Υπολογισμός πλήθους μοναδικών τιμών για το χαρακτηριστικό
                    self.attr_params[attr][c] = {val: (value_counts.get(val, 0) + self.laplace_alpha) /  #Αριθμός φορών που εμφανίζεται η τιμή val στην κλάση c + Laplace
                           (total_count + self.laplace_alpha * unique_values) #Συνολικός αριθμός τιμών στην κλάση c + (Laplace * μοναδικές τιμές)
                           for val in attr_values} #Επανάληψη για κάθε πιθανή τιμή του χαρακτηριστικού (val)
        
        test_df = pd.concat([X_test, y_test], axis=1)  #Ενοποίηση test δεδομένων
        return test_df  #Επιστροφή test δεδομένων

    def predict_proba(self, input_data):  #Συνάρτηση πρόβλεψης πιθανοτήτων P(c|x)
        """Πρόβλεψη: επιστροφή P(c|x) για κάθε κλάση."""
        posteriors = {}  #Λεξικό για τις τελικές πιθανότητες

        for c in self.classes:  #Για κάθε κλάση
            posterior = self.class_prob[c]  #Ξεκινάμε με την P(c)

            for attr, value in input_data.items():  #Για κάθε χαρακτηριστικό εισόδου
                if attr not in self.attr_types:
                    continue  #Παράλειψη άγνωστων χαρακτηριστικών

                if self.attr_types[attr] == "C":  #Αν είναι συνεχές χαρακτηριστικό
                    mean, std = self.attr_params[attr][c]["mean"], self.attr_params[attr][c]["std"]  #Παράμετροι Gaussian
                    exponent = -((value - mean) ** 2) / (2 * std ** 2)  #Εκθέτης Gaussian
                    prob = (1 / (math.sqrt(2 * math.pi) * std)) * math.exp(exponent)  #Πιθανότητα Gaussian
                else:  # Αν είναι διακριτό
                    prob = self.attr_params[attr][c].get(value, self.laplace_alpha)  #Πιθανότητα με Laplace

                posterior *= prob  #Πολλαπλασιασμός πιθανοτήτων

            posteriors[c] = posterior  #Αποθήκευση τελικής πιθανότητας για την κλάση

        total = sum(posteriors.values())  #Άθροισμα πιθανοτήτων για κανονικοποίηση
        if total > 0:
            return {c: p / total for c, p in posteriors.items()}  #Κανονικοποίηση
        else:
            return {c: 1.0 / len(self.classes) for c in self.classes}  #Ομοιόμορφη κατανομή σε περίπτωση αστάθειας

    def print_params(self):  #Συνάρτηση εκτύπωσης των παραμέτρων του μοντέλου
        """Εκτύπωση όλων των παραμέτρων του μοντέλου."""
        print("Πιθανότητες Κλάσεων P(c):")
        for c, p in self.class_prob.items():  #Εκτύπωση πιθανότητας κάθε κλάσης
            print(f"{c}:{p:.4f}")

        print("\nΠαράμετροι Χαρακτηριστικών:")
        for attr in self.attr_types:  #Για κάθε χαρακτηριστικό
            print(f"\n{attr} ({self.attr_types[attr]}):")
            for c in self.classes:  #Για κάθε κλάση
                if self.attr_types[attr] == "C":  #Αν είναι συνεχές
                    mean, std = self.attr_params[attr][c]["mean"], self.attr_params[attr][c]["std"]
                    print(f"{c}: μ = {mean:.2f}, σ = {std:.2f}")
                else:  #Αν είναι διακριτό
                    print(f"  Class {c}:")
                    for val, p in self.attr_params[attr][c].items():  #Εκτύπωση πιθανοτήτων για κάθε τιμή
                        print(f"{val}:{p:.4f}")

    def get_user_query(self):  #Συνάρτηση για εισαγωγή ερωτήματος από τον χρήστη
        """Ζητά από τον χρήστη να εισάγει τιμές για κάθε χαρακτηριστικό."""
        print("\nΕισάγετε τιμές για τα χαρακτηριστικά:")
        query = {}
        for attr, attr_type in self.attr_types.items():
            while True:
                try:
                    value = input(f"{attr} ({attr_type}): ")
                    if attr_type == "C":  #Συνεχές χαρακτηριστικό
                        query[attr] = float(value)
                    else:  #Διακριτό χαρακτηριστικό
                        query[attr] = value
                    break
                except ValueError:
                    print("Λάθος τιμή! Παρακαλώ εισάγετε αριθμό για συνεχές χαρακτηριστικό.")
        return query

if __name__ == "__main__": 
    nb = NaiveBayesClassifier() #Δημιουργία ταξινομητή
    test_data = nb.train("IRIS.csv")  #Εκπαίδευση μοντέλου με τα δεδομένα από το csv (επιστρέφει test δεδομένα)
    nb.print_params()  #Εκτύπωση παραμέτρων του μοντέλου
    
    query = nb.get_user_query()  #Ζητά από τον χρήστη να εισάγει τιμές
    
    print("\nPosterior Probabilities:")  #Εκτύπωση πιθανοτήτων P(c|x)
    for c, p in nb.predict_proba(query).items():
        print(f"P({c}|x) = {p:.6f}")
    
    # Υπολογισμός accuracy στο test set
    correct = 0
    total = len(test_data)
    class_col = test_data.columns[-1]

    for i, row in test_data.iterrows():
        input_data = row[:-1].to_dict()
        true_label = row[class_col]
        predicted = max(nb.predict_proba(input_data), key=lambda k: nb.predict_proba(input_data)[k])
        if predicted == true_label:
            correct += 1

    accuracy = correct / total
    print(f"\nAccuracy on test set: {accuracy:.2%}")