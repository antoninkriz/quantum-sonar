import sys
module_path = '/root/dp/ni-dip/'
sys.path.append(module_path)
from src.svm.svm import SVM
from src.data_objects.project_directories import ProjectDirectories
from src.data_objects.svm_configuration import SVMConfiguration

if len(sys.argv) != 9 and len(sys.argv) != 10:
	print("Usage: python3 classify_on_hardware.py <train_size> <test_size> <n_features> <n_qubits> <depth> <shots> <feature_map> <runtime_jobs_completed> <ibm_backend>")
	sys.exit(1)

# Parse command-line arguments
train_size = int(sys.argv[1])
test_size = int(sys.argv[2])
n_features = int(sys.argv[3])
n_qubits = int(sys.argv[4])
depth = int(sys.argv[5])
shots = int(sys.argv[6])
feature_map = str(sys.argv[7])

if len(sys.argv) == 10:
	runtime_jobs_completed = bool(int(sys.argv[8]))
	ibm_backend = str(sys.argv[9])
elif len(sys.argv) == 9:
	runtime_jobs_completed = bool(int(sys.argv[8]))
	ibm_backend = None
else:
	runtime_jobs_completed = None
	ibm_backend = None

# Load dataset
working_dir = "/root/dp/ni-dip/"
directories = ProjectDirectories(working_dir, results_dir="results/hardware/")

# Construct output file name for the saving the results based on script parameters
output_file_name = f"svm_{train_size}_train_{test_size}_test_{n_features}_features_{n_qubits}_qubits_{depth}_depth_{shots}_shots_{feature_map}_feature_map_{ibm_backend}_backend.txt"
output_file_path = directories.working_dir + directories.dataset_dir + directories.results_dir + output_file_name

# Redirect standard output to the file
try:
	with open(output_file_path, 'w') as f:
		sys.stdout = f
		# Set configuration and create SVM object
		config = SVMConfiguration(
			train_size,
			test_size,
			n_features,
			n_qubits,
			depth,
			shots,
			simulator=False,
			ibm_backend=ibm_backend,
			runtime_jobs_completed=runtime_jobs_completed
		)
		svm = SVM(directories, config)

		# Classify
		svm.classify(feature_maps=[feature_map])
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
