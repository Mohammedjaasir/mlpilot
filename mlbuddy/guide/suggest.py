from sklearn.metrics import accuracy_score


def suggest(guided_model):
    acc = guided_model.accuracy

    print("\n💬 mlbuddy suggestions:\n")
    print(f"  Your model accuracy: {acc * 100:.1f}%\n")

    if acc < 0.60:
        print("  ⚠ Accuracy is quite low. Here's what to try:\n")
        print("    1. Get more training data.")
        print("    2. Check your labels — are they correct?")
        if guided_model.model_name != "random_forest":
            print("    3. Try model='random_forest' — often works better.")
        else:
            print("    3. Try model='svm' or model='logistic' instead.")
        print("    4. Your features might not be informative enough.")

    elif acc < 0.80:
        print("  🟡 Decent start! Here's how to improve:\n")
        print("    1. Try model='random_forest'.")
        print("    2. Add more features if you have them.")
        print("    3. Check for class imbalance.")
        print("    4. Try tuning hyperparameters.")

    elif acc < 0.90:
        print("  🟢 Good accuracy! To push further:\n")
        print("    1. Try model='svm'.")
        print("    2. Engineer new features from existing ones.")
        print("    3. Collect more diverse training data.")

    else:
        print("  ✅ Excellent accuracy! A few things to check:\n")
        print("    1. Make sure you're not overfitting.")
        print("    2. Check for duplicated rows in your dataset.")
        print("    3. If all looks good — you're ready to deploy!")

    print()
    print("  📘 Quick guide:")
    print("    - Accuracy < 60%  → data quality or quantity problem")
    print("    - Accuracy 60-80% → try different models or more features")
    print("    - Accuracy > 90%  → check for overfitting\n")