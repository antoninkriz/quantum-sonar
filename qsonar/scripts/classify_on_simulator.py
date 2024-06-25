import sys
module_path = "/Users/eliskakratka/documents/quantum-stuff/"
sys.path.append(module_path)
import qsonar.load
from qsonar.data_objects.project_directories import ProjectDirectories
from qsonar.data_objects.svm_configuration import SVMConfiguration
from qsonar.svm.svm import SVM

if len(sys.argv) != 4:
	print("Usage: python3 classify_on_simulator.py <n_qubits> <depth> <shots>")
	sys.exit(1)

# Parse command-line arguments
n_qubits = int(sys.argv[1])
depth = int(sys.argv[2])
shots = int(sys.argv[3])

df = qsonar.load.load_data_from_internet()
X_train, X_test, y_train, y_test = qsonar.load.scale_and_split(df, pca_components=4)
train_size = X_train.shape[0]
test_size = X_test.shape[0]

# Load dataset
working_dir = module_path
directories = ProjectDirectories(working_dir, results_dir="results/simulator/")

# Construct output file name for the saving the results based on script parameters
output_file_name = f"svm_{train_size}_train_{test_size}_test_{n_qubits}_qubits_{depth}_depth_{shots}_shots.txt"
output_file_path = directories.working_dir + directories.dataset_dir + directories.results_dir + output_file_name

# Redirect standard output to the file
try:
	with open(output_file_path, 'w') as f:
		sys.stdout = f
		# Set configuration and create SVM object
		config = SVMConfiguration(
			train_size,
			test_size,
			n_qubits,
			depth,
			shots=shots,
			train_samples=X_train,
			test_samples=X_test,
			train_labels=y_train,
			test_labels=y_test
		)
		svm = SVM(directories, config)

		# Classify
		feature_maps = ["zz", "pauli", "zzPhi", "z"]
		classical_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
		svm.classify(feature_maps=feature_maps)
except FileNotFoundError as e:
	print(f"Error: File not found - {e}")
except PermissionError:
	print("Permission denied.")
except OSError as e:
	print(f"Error: Unable to write to file - {e}")
except Exception as e:
	print(f"Unexpected error: {e}")

# Reset standard output to the console
sys.stdout = sys.__stdout__

print(f"Results saved to {output_file_path}")
