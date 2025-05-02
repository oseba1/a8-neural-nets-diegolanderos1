from neural import *
import csv

raw_data = []

with open('Bean.csv', 'r') as file:
    csvreader = csv.reader(file, delimiter='\t')

    header = next(csvreader)  # Skip header

    for row in csvreader:
        area = float(row[0])
        perimeter = float(row[1])
        solidity = float(row[2])
        roundness = float(row[3])
        compactness = float(row[4])
        raw_data.append((area, perimeter, solidity, compactness, roundness))


def min_max_scale(column):
    min_val = min(column)
    max_val = max(column)
    return [(x - min_val) / (max_val - min_val) for x in column], min_val, max_val

columns = list(zip(*raw_data))  # Transpose to columns
scaled_columns = []
scalers = []

for col in columns:
    scaled_col, min_val, max_val = min_max_scale(col)
    scaled_columns.append(scaled_col)
    scalers.append((min_val, max_val))


training_data = []

for i in range(len(raw_data)):
    inputs = [scaled_columns[0][i], scaled_columns[1][i], scaled_columns[2][i], scaled_columns[3][i]]  # All but roundness
    target = [scaled_columns[4][i]]  # Roundness
    training_data.append((inputs, target))


print("<<<<<<<<<<<<<< Predicting Roundness >>>>>>>>>>>>>>\n")
bean = NeuralNet(4, 6, 1)
print("Starting training...")
bean.train(training_data, learning_rate=0.7, iters=100, print_interval=10)
print("Training finished.")


print("\nTesting on training data:")
print(bean.test_with_expected(training_data))

def normalize_input(area, perimeter, solidity, compactness):
    a_min, a_max = scalers[0]
    p_min, p_max = scalers[1]
    s_min, s_max = scalers[2]
    c_min, c_max = scalers[3]
    return [
        (area - a_min) / (a_max - a_min),
        (perimeter - p_min) / (p_max - p_min),
        (solidity - s_min) / (s_max - s_min),
        (compactness - c_min) / (c_max - c_min),
    ]

def predict_and_denormalize(area, perimeter, solidity, compactness):
    norm_input = normalize_input(area, perimeter, solidity, compactness)
    pred_norm = bean.evaluate(norm_input)[0]
    r_min, r_max = scalers[4]
    return pred_norm * (r_max - r_min) + r_min

print("\nPredictions (Actual Roundness):")
print(predict_and_denormalize(28395, 610.291, 0.988856, 0.913358))  # ~0.958
print(predict_and_denormalize(29380, 624.11, 0.989559, 0.908774))   # ~0.948
print(predict_and_denormalize(30008, 645.884, 0.976696, 0.928329))  # ~0.904
print(predict_and_denormalize(25000, 590.0, 0.975, 0.900))