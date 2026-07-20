# CMP405 - Τεχνητή Νοημοσύνη
## Supervised, Unsupervised και Semi-Supervised Learning με MNIST

Το repository υλοποιεί πλήρως τα Μέρη Α1, Α2, Β, Γ1 και Γ2 της εργασίας CMP405 με κοινό dataset το MNIST.

## Κύρια αποτελέσματα

| Μέθοδος | Accuracy |
|---|---:|
| Supervised Random Forest (100% labels) | 97.07% |
| Νέα ψηφιακά χειρόγραφα/σχεδιασμένα ψηφία (ίδιο RF, χωρίς retraining) | 65.00% |
| K-Means, k=10 (post-hoc αντιστοίχιση clusters για αξιολόγηση) | 59.54% |
| Label Propagation (μόνο 5% labels) | 95.67% |
| K-Means + Cluster Labeling (μόνο 5% labels) | 59.54% |

Επιπλέον baseline με Random Forest εκπαιδευμένο μόνο στο ίδιο 5% των labels: **93.27%**. Αυτό δείχνει ότι το Label Propagation αξιοποίησε χρήσιμα τα unlabeled δεδομένα (+2.40 ποσοστιαίες μονάδες έναντι του 5%-only baseline).

## Δομή repository

```text
cmp405-mnist-learning/
├── data/
│   ├── mnist_standard.npz
│   ├── README.md
│   └── external_digits/
│       ├── raw/                 # 20 νέα ψηφιακά δείγματα
│       ├── processed/           # 28x28, grayscale, MNIST-compatible
│       └── external_digits.npz
├── report/
│   ├── CMP405_Report.pdf
│   └── CMP405_Report.docx
├── results/
│   ├── figures/
│   ├── tables/
│   ├── metrics.json
│   └── run_log.txt
├── src/
│   ├── prepare_data.py
│   └── run_all.py
├── DATA_SOURCES.md
├── FINAL_STEPS_BEFORE_SUBMISSION.md
├── SOS_SUMMARY_GR.md
├── requirements.txt
├── run_project.sh
└── README.md
```

## Εγκατάσταση

Προτείνεται Python 3.10+.

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Εγκατάσταση βιβλιοθηκών:

```bash
pip install -r requirements.txt
```

## Εκτέλεση

Το MNIST περιλαμβάνεται ήδη σε συμπιεσμένη μορφή (`data/mnist_standard.npz`), άρα το project μπορεί να τρέξει offline:

```bash
python src/run_all.py
```

ή σε Linux/macOS:

```bash
./run_project.sh
```

Τα αποτελέσματα γράφονται αυτόματα στους φακέλους `results/figures/`, `results/tables/` και στο `results/metrics.json`.

### Προαιρετική αναδημιουργία δεδομένων

Το αρχείο `src/prepare_data.py` μπορεί να ξαναδημιουργήσει τα 20 νέα ψηφία και, αν λείπει το τοπικό MNIST, να το κατεβάσει μέσω OpenML:

```bash
python src/prepare_data.py
```

Για εξαναγκασμένη νέα λήψη:

```bash
python src/prepare_data.py --force-download
```

## Μεθοδολογία

### Α1 - Supervised Learning
- Κανονικοποίηση pixels: `[0,255] -> [0,1]`.
- Random Forest, 200 trees, `max_features="sqrt"`, `random_state=42`.
- Αξιολόγηση: Accuracy, macro Precision, Recall, F1-score και Confusion Matrix.

### Α2 - Νέα ψηφία
- 20 ανεξάρτητα ψηφιακά σχεδιασμένα δείγματα, 2 ανά ψηφίο.
- Μετατροπή σε grayscale, inversion, crop, resize ώστε το ψηφίο να χωρά σε 20x20 και centering σε καμβά 28x28.
- Χρησιμοποιείται **το ίδιο ήδη εκπαιδευμένο Random Forest χωρίς retraining**.

### Β - Unsupervised Learning
- PCA 20 components ως unsupervised dimensionality reduction πριν το clustering.
- K-Means με `k=10`, k-means++, `n_init=3`.
- Labels δεν χρησιμοποιούνται στο fit.
- Μετρικές: Inertia/SSE, cohesion, separation, Silhouette Coefficient.
- Τα πραγματικά labels χρησιμοποιούνται μόνο εκ των υστέρων για cluster composition, purity και ερμηνεία.

### Γ - Semi-Supervised Learning
- Ακριβώς 5% labeled samples = 3,000 από τα 60,000 training samples.
- Επιλογή με **Stratified Sampling** ώστε να εκπροσωπούνται όλες οι κλάσεις.
- Τα υπόλοιπα 57,000 labels αποκρύπτονται κατά την εκπαίδευση.

**Γ1 Label Propagation**
- PCA 20 components.
- KNN kernel, 7 neighbors, max_iter=500.
- Test accuracy: 95.67%.

**Γ2 K-Means + Cluster Labeling**
- Τα clusters δημιουργούνται χωρίς labels.
- Η αντιστοίχιση cluster -> digit γίνεται μόνο από τα διαθέσιμα 5% labels με majority vote.
- Test accuracy: 59.54%.

## Αναπαραγωγιμότητα

Όπου υπάρχει τυχαιότητα χρησιμοποιείται `random_state=42`. Οι ακριβείς μετρικές της τελικής εκτέλεσης βρίσκονται στο `results/metrics.json`.

## GitHub URL για το report

Πριν την τελική υποβολή, δημιουργήστε GitHub repository από αυτόν τον φάκελο και αντικαταστήστε στο εξώφυλλο του report το placeholder:

`https://github.com/USERNAME/cmp405-mnist-learning`

με το πραγματικό URL του repository.
