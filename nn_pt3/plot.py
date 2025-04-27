import numpy as np
import matplotlib.pyplot as plt

# Μπορούμε να ορίσουμε ένα colormap ή απλώς μια λίστα από χρώματα για τις 4 κατηγορίες
colors = ['red', 'blue', 'green', 'orange']
labels = ['C1', 'C2', 'C3', 'C4']

# ----------------------------------------------------------------------------
# 1) Plot για train_data.txt
# ----------------------------------------------------------------------------
# Διαβάζουμε όλο το αρχείο, παραλείποντας την πρώτη γραμμή (skiprows=1)
train_data = np.loadtxt('train_data.txt', skiprows=1)
# Οι στήλες είναι: [ x1, x2, class ]
x1_train = train_data[:, 0]
x2_train = train_data[:, 1]
c_train = train_data[:, 2].astype(int)  # μετατρέπουμε σε ακέραιο για μάσκες

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for cls in range(4):
    # φτιάχνουμε μάσκα για τα σημεία της κατηγορίας cls
    mask = (c_train == cls)
    # Το όρισμα s καθορίζει το μέγεθος των κουκκίδων στο scatter
    plt.scatter(
        x1_train[mask],
        x2_train[mask],
        color=colors[cls],
        label=f'{labels[cls]} (train)',
        alpha=0.5,
        s=10  # <= Μικρότερες κουκκίδες
    )

plt.title('Train data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='best')
plt.grid(True)

# ----------------------------------------------------------------------------
# 2) Plot για test_data.txt
# ----------------------------------------------------------------------------
test_data = np.loadtxt('test_data.txt', skiprows=1)
# Οι στήλες είναι: [ x1, x2, class ]
x1_test = test_data[:, 0]
x2_test = test_data[:, 1]
c_test = test_data[:, 2].astype(int)

plt.subplot(1, 2, 2)
for cls in range(4):
    mask = (c_test == cls)
    plt.scatter(
        x1_test[mask],
        x2_test[mask],
        color=colors[cls],
        label=f'{labels[cls]} (test)',
        alpha=0.5,
        s=10  # <= Μικρότερες κουκκίδες
    )

plt.title('Test data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='best')
plt.grid(True)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# 3) Προαιρετικό plot: classified_test_points.txt
#    Δείχνει την προβλεφθείσα κατηγορία και αν είναι σωστή ή λάθος
# ----------------------------------------------------------------------------

classified_data = np.loadtxt('classified_test_points.txt')
x1_c = classified_data[:, 0]
x2_c = classified_data[:, 1]
true_c = classified_data[:, 2].astype(int)
pred_c = classified_data[:, 3].astype(int)
correct = classified_data[:, 4].astype(int)

plt.figure(figsize=(12, 5))

# (α) Αριστερά: χρωματίζουμε με βάση την *πραγματική* κατηγορία
plt.subplot(1, 2, 1)
for cls in range(4):
    mask = (true_c == cls)
    plt.scatter(
        x1_c[mask],
        x2_c[mask],
        color=colors[cls],
        label=f'True: {labels[cls]}',
        alpha=0.5,
        s=10  # <= Μικρότερες κουκκίδες
    )
plt.title('Test data (True classes)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='best')
plt.grid(True)

# (β) Δεξιά: χρωματίζουμε με βάση την *προβλεφθείσα* κατηγορία,
# και χρησιμοποιούμε διαφορετικό marker για σωστές/λάθος προβλέψεις
plt.subplot(1, 2, 2)
for cls in range(4):
    # Σωστές προβλέψεις
    mask_correct = (pred_c == cls) & (correct == 1)
    # Λάθος προβλέψεις
    mask_incorrect = (pred_c == cls) & (correct == 0)
    
    plt.scatter(
        x1_c[mask_correct],
        x2_c[mask_correct],
        color=colors[cls],
        marker='o',
        alpha=0.7,
        label=f'Pred: {labels[cls]} (correct)',
        s=10  # <= Μικρότερες κουκκίδες
    )
    plt.scatter(
        x1_c[mask_incorrect],
        x2_c[mask_incorrect],
        color=colors[cls],
        marker='x',
        alpha=0.7,
        label=f'Pred: {labels[cls]} (wrong)',
        s=30  # Π.χ. λίγο μεγαλύτερο για να ξεχωρίζουν τα 'x'
    )

plt.title('Test data (Predicted classes)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='best')
plt.grid(True)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# 4) Plot για το error (loss) που αποθηκεύσαμε στο αρχείο loss_per_epoch.txt
# ----------------------------------------------------------------------------
loss_data = np.loadtxt('loss_per_epoch.txt')
epochs = loss_data[:, 0]
loss = loss_data[:, 1]

plt.figure(figsize=(8, 4))
plt.plot(epochs, loss, 'b-o', markersize=5, alpha=0.7, label='Training Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.grid(True)
plt.show()
