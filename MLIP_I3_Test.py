import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from giskard import Dataset, Model, scan, testing

# Synthetic dataset creation for the recommendation system
import numpy as np
from faker import Faker

fake = Faker()
num_users = 1000
num_movies = 500

# Generate users dataframe
users = pd.DataFrame({
    'user_id': range(1, num_users + 1),
    'age': np.random.randint(18, 70, num_users),
    'subscription_type': np.random.choice(['Free', 'Premium'], num_users)
})

# Generate movies dataframe
movies = pd.DataFrame({
    'movie_id': range(1, num_movies + 1),
    'genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi'], num_movies),
})

# Generate interactions
interactions = pd.DataFrame({
    'user_id': np.random.randint(1, num_users + 1, 10000),
    'movie_id': np.random.randint(1, num_movies + 1, 10000),
    'rating': np.random.randint(1, 6, 10000),  # User's rating (target variable)
})

# Merge to create a dataset
data = interactions.merge(users, on='user_id').merge(movies, on='movie_id')

# Encode categorical features
data['is_premium'] = data['subscription_type'].apply(lambda x: 1 if x == 'Premium' else 0)
data = pd.get_dummies(data, columns=['genre'], drop_first=True)

# Select features and target
features = ['user_id', 'movie_id', 'age', 'is_premium'] + [col for col in data.columns if 'genre_' in col]
X = data[features]
y = data['rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Prepare raw_data for Giskard
raw_data = pd.concat([X_test, y_test.rename("rating")], axis=1)

giskard_dataset = Dataset(
    df=raw_data,
    target='rating',
    name="Movie Recommendation Dataset",
    cat_columns=['user_id', 'movie_id', 'is_premium'] + [col for col in raw_data.columns if 'genre_' in col]
)

giskard_model = Model(
    model=model,
    model_type="classification",
    name="Movie Recommendation Classifier",
    feature_names=X_test.columns.tolist()
)
# Scan the model for vulnerabilities
results = scan(giskard_model, giskard_dataset)
print(results)

# Generate test suite from scan results
test_suite = results.generate_test_suite("Movie Recommendation Test Suite")
test_suite.run()
