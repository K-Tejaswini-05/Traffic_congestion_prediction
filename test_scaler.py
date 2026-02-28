import pickle

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("Scaler loaded successfully!")
print(type(scaler))