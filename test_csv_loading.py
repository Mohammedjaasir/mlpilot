import mlbuddy as ml

# Load the real CSV dataset
print('Loading real dataset from CSV...')
X_train, X_test, y_train, y_test = ml.load_csv('sample_data.csv', target_column='hired')

print('\n✅ Dataset loaded successfully!')
print(f'Training set shape: {X_train.shape}')
print(f'Test set shape: {X_test.shape}')
print(f'Features: age, income, score')
print(f'Target: hired (0=no, 1=yes)')

# Now try compare and train
print('\n' + '='*50)
print('Comparing all models on real data...')
print('='*50)
results = ml.compare(X_train, X_test, y_train, y_test)

# Train the best model
best_model = max(results, key=results.get)
print(f'\n' + '='*50)
print(f'Training the best model: {best_model}')
print('='*50)
model = ml.train(X_train, X_test, y_train, y_test, model=best_model)
