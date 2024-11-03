# Enhancing Movie Recommendation Systems with Giskard: A Comprehensive Testing Approach
In the rapidly evolving landscape of machine learning, deploying models into production environments demands more than just high accuracy. Ensuring reliability, fairness, and robustness is paramount to maintaining user trust and delivering consistent performance. This is where Giskard, an open-source MLOps tool, steps in to revolutionize the way we test and validate ML models. In this blog post, we'll explore how Giskard addresses critical challenges in building production-ready machine learning systems, illustrated through a movie recommendation system example.

## The Challenge: Ensuring Model Reliability in Production

Deploying an ML model involves navigating a myriad of challenges beyond initial training and validation. Models can inadvertently harbor biases, suffer from data leakage, or perform inconsistently across different user segments. Traditional evaluation metrics like accuracy or R² scores, while essential, often fall short in uncovering these nuanced issues. To bridge this gap, robust testing frameworks are needed to systematically identify and mitigate potential vulnerabilities in ML models.

## Introduction of Giskard

Giskard is an MLOps tool designed to streamline the testing, debugging, and deployment of machine learning models by providing a comprehensive suite of functionalities focused on model reliability, fairness, and robustness. Built with a user-friendly interface and detailed documentation, Giskard offers specialized modules to address various aspects of machine learning workflows—from model testing and bias detection to deployment and monitoring in CI/CD pipelines. It provides a suite of tools for:
- Automated Vulnerability Detection: Identify performance biases, data leakage, ethical issues, and more.
- Comprehensive Test Suite Generation: Automatically create and curate tests that go beyond basic accuracy metrics.
- Model Explainability: Understand and interpret model predictions to ensure they align with desired outcomes.
- Collaboration and Reporting: Share insights and test results seamlessly with your team.

### Key Features and Functionalities

- **Quickstart and Setup Guides:** Giskard provides easy-to-follow Quickstart guides across multiple domains, including Large Language Models (LLMs), NLP, vision, and tabular data, which can help users get up and running quickly based on their specific use case.
  ![image](https://github.com/user-attachments/assets/f730661f-95ab-440e-8783-e541e6cbf923)


- **Detailed Guides on different tasks:**
    - Installation and Dependency Management: The installation guide includes details on setting up the Giskard library with Python, covering dependency handling and installation commands. This helps users avoid conflicts with other libraries, like pandas, and ensures a smooth setup process.
    - LLM Client Setup: Giskard supports multiple LLM clients, including OpenAI GPT, Azure, and custom models, with detailed instructions on configuring API keys and endpoints. This feature is especially beneficial for organizations working with LLMs in production and seeking a unified way to connect their models with Giskard.
    - Model Scanning: Giskard's model scanning functionality automatically detects potential vulnerabilities in machine learning models. It enables users to assess model reliability, fairness, and robustness through a series of steps, including data wrapping, model integration, and comprehensive scanning. This functionality is available across different domains like tabular, NLP, and vision, making it versatile for various applications.
    - RAG Evaluation Toolkit: The RAG (Retrieval-Augmented Generation) Evaluation Toolkit is a unique feature in Giskard that helps create test sets for evaluating RAG agents, automating the process of question generation and evaluation. This toolkit can generate question-answer pairs from a knowledge base, allowing users to validate RAG agent performance and identify weaknesses.
    - Customizable Tests: Giskard allows users to create and execute tests tailored to their specific use case, such as drift tests, performance tests, and statistical tests. It also offers a catalog of predefined tests to streamline the process, making it easier for teams to build robust test suites without starting from scratch.
    - Integrate Tests: Giskard also allows you to integrate it into existing workflows, making it possible to automate the testing and logging of your model quality.
      ![image](https://github.com/user-attachments/assets/0f581209-a3cf-4447-a9e6-1244d48d27ce)

- **Notebook Tutorials:** Giskard provides a range of notebook tutorials to guide users through real-world machine learning tasks, such as churn prediction, fraud detection, drug classification, and more. These tutorials offer hands-on experience and are valuable for users looking to understand Giskard’s application in practical settings.
  ![image](https://github.com/user-attachments/assets/d2fef3d7-cd2d-417e-9061-5612ef243842)

- **Integration with Popular MLOps Tools:** Giskard’s integration with CI/CD platforms like GitHub, MLflow, and Weights & Biases makes it easier to incorporate model testing and evaluation into continuous integration workflows. This ensures that every model update is tested automatically, helping teams avoid introducing vulnerabilities into production models.
  ![image](https://github.com/user-attachments/assets/5618b19b-2789-4787-b740-0e7f8f00a3bd)



## Using Giskard with a Movie Recommendation System
### Giskard as a MLOps tool for Model Testing

**Warning: This section involves the code implementation of creating test datasets and modeling. You can skip the code blocks if not interested.**

To illustrate Giskard's capabilities, let's walk through its integration with a movie recommendation system. We'll cover data preparation, model training, Giskard integration, running scans, and interpreting the results.

### Data Preparation 

For this example, we created a synthetic dataset representing user interactions with a movie streaming platform. The dataset includes:
- Users: Attributes like user_id, age, and subscription_type (Free or Premium).
- Movies: Attributes like movie_id and genre.
- Interactions: User ratings for movies (rating), serving as the target variable.

```
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

```

### Model Training

We trained a RandomForestClassifier to predict user ratings based on the features.

```
# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
```

### Integrating Giskard
With the model trained, we integrated Giskard to perform comprehensive testing.

**Wrapping the Dataset**

First, we wrapped our dataset using Giskard’s `Dataset` class to prepare it for vulnerability scanning.

```
# Prepare raw_data for Giskard
raw_data = pd.concat([X_test, y_test.rename("rating")], axis=1)

giskard_dataset = Dataset(
    df=raw_data,
    target='rating',
    name="Movie Recommendation Dataset",
    cat_columns=['user_id', 'movie_id', 'is_premium'] + [col for col in raw_data.columns if 'genre_' in col]
)
```


**Wrapping the Model**

Next, we wrapped our trained model using Giskard’s `Model` class, specifying it as a classification model.

```
giskard_model = Model(
    model=model,
    model_type="classification",
    name="Movie Recommendation Classifier",
    feature_names=X_test.columns.tolist()
)
```

### Running Scans and Analyzing Output

With both the dataset and model wrapped, we initiated a scan to detect potential vulnerabilities.

```
# Scan the model for vulnerabilities
results = scan(giskard_model, giskard_dataset)
print(results)
```

### Understanding the Scan Output

The scan output provides detailed logs of various detectors that assess different aspects of the model. Here's an interpretation of the key sections from the scan log:

- Casting and Prediction Logs:
    - Casting dataframe columns: Ensures that the data types of each column are consistent and compatible with the model's expectations.
    - Predicted dataset with shape: Indicates the shape of the dataset used for predictions during the scan and the time taken for each prediction.
    - Sample output:![image](https://github.com/user-attachments/assets/50178137-94b5-4ad1-a9ac-e1c171426778)


- Detectors(example included):
    - DataLeakageDetector:
        - Purpose: Detects if any target information inadvertently leaks into the features, leading to inflated performance metrics.
        - Result: 0 issue detected indicates no data leakage was found.
    - EthicalBiasDetector:
        - Purpose: Evaluate the model for potential ethical biases by simulating scenarios like switching user demographics.
        - Result: 0 issue detected suggests no significant ethical biases were identified.
    - TextPerturbationDetector:
        - Purpose: Assesses the model's robustness against minor text changes, such as typos or case transformations.
        - Result: 0 issue detected means the model's predictions remain stable despite text perturbations.
    - OverconfidenceDetector:
        - Purpose: Identifies if the model is excessively confident in its predictions for specific data slices.
        - Result: 2 issues detected in slices like age >= 30.5 AND age < 34.5 and genre_Sci-Fi == True, indicating areas where the model's confidence surpasses acceptable thresholds.
    - UnderconfidenceDetector:
        - Purpose: Checks if the model exhibits too much uncertainty in its predictions for certain data slices.
        - Result: 1 issue detected in the slice is_premium == 0, indicating underconfidence in predicting for non-premium users.
    - SpuriousCorrelationDetector:
        - Purpose: Detects irrelevant or weakly associated features that the model might be relying on for predictions.
        - Result: 0 issue detected confirms that the model's predictions aren't influenced by spurious correlations.
    - PerformanceBiasDetector:
        - Purpose: Evaluates the model's performance across different user groups or genres to identify biases.
        - Result: 2 issues detected in slices like genre_Horror == True and is_premium == 1, where precision and recall metrics deviate from the global performance.
    - StochasticityDetector:
        - Purpose: Ensures the model provides consistent outputs for identical inputs, avoiding unpredictability.
        - Result: 0 issue detected confirms the model's predictions are deterministic and reliable.

Sample output:
   ![image](https://github.com/user-attachments/assets/9204de8c-1030-41b3-8be7-4194b089a28a)

- Scan Summary:
    - Indicates a total of 5 issues detected across all detectors, highlighting areas for model improvement.
  ![image](https://github.com/user-attachments/assets/72bb7445-a378-49c8-b516-f64c3ca8f806)

- Failed Test Details:
    - Overconfidence Tests:
        - Slices like `age >= 30.5 AND age < 34.5 and genre_Sci-Fi == True` exceeded the overconfidence thresholds.
        - Metric: Represents the model's confidence level, with higher values indicating overconfidence.
    - Underconfidence Test:
        - Slice `is_premium == 0` fell below the underconfidence threshold.
        - Metric: Lower values indicate insufficient confidence in predictions.
    - Precision Tests:
        - Slice `genre_Horror == True` showed a precision below the expected threshold.
        - Metric: Measures the accuracy of positive predictions, with lower values indicating potential bias.

## Strengths of Giskard

- Comprehensive Testing: Giskard's suite of detectors covers a wide range of potential model vulnerabilities, from ethical biases to performance inconsistencies.
- Automated Vulnerability Detection: Automates the process of identifying issues, saving time and ensuring thorough evaluation.
- Ease of Integration: Seamlessly integrates with existing ML pipelines, supporting various model types and datasets.
- Actionable Insights: Provides detailed logs and metrics, enabling developers to pinpoint and address specific issues effectively.
- Open-Source and Extensible: Being open-source, Giskard allows for customization and extension to fit unique project requirements.
- Seamless CI/CD Integration: With integration options for GitHub, MLflow, and Weights & Biases, Giskard fits easily into modern MLOps workflows, enabling continuous testing and monitoring. This helps ML teams ensure that every model version meets quality standards before deployment.
- Specific Tools for RAG Evaluation: Giskard’s RAG Evaluation Toolkit is particularly useful for validating Retrieval-Augmented Generation systems, which are increasingly used in NLP and LLM-based applications. By automating test set generation, Giskard helps streamline the evaluation process for these complex systems.

## Limitations and Considerations

- Synthetic Data Constraints: In this example, we used synthetic data, which may not capture the complexities of real-world datasets. Results may vary with real user interaction data.
- Threshold Configuration: Detectors rely on predefined thresholds, which might need fine-tuning based on specific use cases and model requirements.
- Performance Overhead: Running comprehensive scans can be time-consuming, especially with large datasets or complex models.
- Limited Detector Scope: While Giskard covers many areas, some niche vulnerabilities or domain-specific issues might require additional custom detectors.
- Interpretation of Results: Understanding and acting upon the scan results requires a good grasp of the model's domain and the implications of detected issues.
- Dependency Sensitivity: Giskard's setup requires careful handling of dependencies, which could cause issues for users with complex project environments. The tool provides guidance on resolving dependency conflicts (e.g., with pandas), but users may need to uninstall and reinstall certain libraries, which could disrupt workflows.
- Limited Customization for Specialized Use Cases: While Giskard supports customization of tests, highly specialized use cases may require significant adaptation. For instance, users working with unconventional data types or complex model architectures might find Giskard’s standard testing functionalities somewhat limiting and may need to create extensive custom tests.

## Conclusion

Giskard emerges as a powerful tool in the MLOps arsenal, offering robust mechanisms to ensure that machine learning models are not only accurate but also fair, reliable, and free from hidden vulnerabilities. By integrating Giskard into a movie recommendation system, we were able to uncover specific areas where the model's performance could be improved, such as addressing overconfidence in certain genres and enhancing prediction reliability for non-premium users.

While Giskard provides comprehensive testing capabilities, it's essential to complement its automated processes with domain expertise and continuous monitoring to maintain model integrity in dynamic production environments. Embracing tools like Giskard paves the way for building trustworthy and high-performing machine-learning systems that stand the test of real-world challenges.

## Reference

https://docs.giskard.ai/en/stable/index.html




