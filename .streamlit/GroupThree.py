import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Function to calculate accuracy and predictions
def calculate_accuracy_and_predictions(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    return accuracy, y_pred

# Create a Streamlit app
st.title('Crop Yield Prediction Dashboard')

# Add widgets to the app
st.sidebar.title('Model Selection')
selected_model = st.sidebar.selectbox('Select Model', ['Naive Bayes', 'Decision Tree', 'Random Forest'])

# Load your datasets
crop_wether_soils_fertiliser_w_df = pd.read_csv("crop_wether_soils_fertiliser_w_df.csv")
crop_wether_soils_fertiliser_b_df = pd.read_csv("crop_wether_soils_fertiliser_b_df.csv")
crop_wether_soils_fertiliser_m_df = pd.read_csv("crop_wether_soils_fertiliser_m_df.csv")

# Split the data and calculate accuracy for each crop
X_w = crop_wether_soils_fertiliser_w_df.drop(['Wheat Tonnes', 'Category'], axis=1)
y_w = crop_wether_soils_fertiliser_w_df['Category']
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_w, y_w, test_size=0.25, random_state=42)

X_b = crop_wether_soils_fertiliser_b_df.drop(['Barley Tonnes', 'Category'], axis=1)
y_b = crop_wether_soils_fertiliser_b_df['Category']
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_b, y_b, test_size=0.25, random_state=42)

X_m = crop_wether_soils_fertiliser_m_df.drop(['Maize grain Tonnes', 'Category'], axis=1)
y_m = crop_wether_soils_fertiliser_m_df['Category']
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_m, y_m, test_size=0.25, random_state=42)

# Display accuracy for each crop based on the selected model
if selected_model == 'Naive Bayes':
    accuracy_w, y_pred_w = calculate_accuracy_and_predictions(X_train_w, X_test_w, y_train_w, y_test_w, GaussianNB())
    accuracy_b, y_pred_b = calculate_accuracy_and_predictions(X_train_b, X_test_b, y_train_b, y_test_b, GaussianNB())
    accuracy_m, y_pred_m = calculate_accuracy_and_predictions(X_train_m, X_test_m, y_train_m, y_test_m, GaussianNB())

    # Display accuracy for each crop with a frame
    st.markdown('### Results')  # Header for the results
    st.write(f'Test Accuracy for Wheat: {accuracy_w}')
    st.write(f'Test Accuracy for Barley: {accuracy_b}')
    st.write(f'Test Accuracy for Maize Grain: {accuracy_m}')

    # Plot histogram
    all_accuracies = [accuracy_w, accuracy_b, accuracy_m]
    colors = ['blue', 'green', 'orange']
    fig, ax = plt.subplots()
    for i, (accuracy, color) in enumerate(zip(all_accuracies, colors)):
        ax.bar(['Wheat', 'Barley', 'Maize Grain'][i], accuracy, color=color, alpha=0.7)
    ax.set_title('Test Accuracy Distribution for Naive Bayes')
    ax.set_xlabel('Crop')
    ax.set_ylabel('Accuracy')
    st.pyplot(fig)

    # Create dataset with actual and predicted categories
    crop_types = ['Wheat', 'Barley', 'Maize Grain']
    results_df_w = pd.DataFrame({'Actual': y_test_w, 'Predicted': y_pred_w})
    results_df_w['Crop_type'] = crop_types[0]

    results_df_b = pd.DataFrame({'Actual': y_test_b, 'Predicted': y_pred_b})
    results_df_b['Crop_type'] = crop_types[1]

    results_df_m = pd.DataFrame({'Actual': y_test_m, 'Predicted': y_pred_m})
    results_df_m['Crop_type'] = crop_types[2]

    # Display the table on the left sidebar
    st.sidebar.write(results_df_w)
    st.sidebar.write(results_df_b)
    st.sidebar.write(results_df_m)


elif selected_model == 'Decision Tree':
    accuracy_w, y_pred_w = calculate_accuracy_and_predictions(X_train_w, X_test_w, y_train_w, y_test_w, DecisionTreeClassifier(max_depth=3, random_state=42))
    accuracy_b, y_pred_b = calculate_accuracy_and_predictions(X_train_b, X_test_b, y_train_b, y_test_b, DecisionTreeClassifier(max_depth=3, random_state=42))
    accuracy_m, y_pred_m = calculate_accuracy_and_predictions(X_train_m, X_test_m, y_train_m, y_test_m, DecisionTreeClassifier(max_depth=3, random_state=42))
    # Display accuracy for each crop with a frame
    st.markdown('### Results')  # Header for the results
    st.write(f'Test Accuracy for Wheat: {accuracy_w}')
    st.write(f'Test Accuracy for Barley: {accuracy_b}')
    st.write(f'Test Accuracy for Maize Grain: {accuracy_m}')

    # Plot histogram
    all_accuracies = [accuracy_w, accuracy_b, accuracy_m]
    colors = ['blue', 'green', 'orange']
    fig, ax = plt.subplots()
    for i, (accuracy, color) in enumerate(zip(all_accuracies, colors)):
        ax.bar(['Wheat', 'Barley', 'Maize Grain'][i], accuracy, color=color, alpha=0.7)
    ax.set_title('Test Accuracy Distribution for Decision Tree')
    ax.set_xlabel('Crop')
    ax.set_ylabel('Accuracy')
    st.pyplot(fig)

    # Create dataset with actual and predicted categories
    crop_types = ['Wheat', 'Barley', 'Maize Grain']
    results_df_w = pd.DataFrame({'Actual': y_test_w, 'Predicted': y_pred_w})
    results_df_w['Crop_type'] = crop_types[0]

    results_df_b = pd.DataFrame({'Actual': y_test_b, 'Predicted': y_pred_b})
    results_df_b['Crop_type'] = crop_types[1]

    results_df_m = pd.DataFrame({'Actual': y_test_m, 'Predicted': y_pred_m})
    results_df_m['Crop_type'] = crop_types[2]

    # Display the table on the left sidebar
    st.sidebar.write(results_df_w)
    st.sidebar.write(results_df_b)
    st.sidebar.write(results_df_m)

elif selected_model == 'Random Forest':
    accuracy_w, y_pred_w = calculate_accuracy_and_predictions(X_train_w, X_test_w, y_train_w, y_test_w, RandomForestClassifier(n_estimators=100, random_state=42))
    accuracy_b, y_pred_b = calculate_accuracy_and_predictions(X_train_b, X_test_b, y_train_b, y_test_b, RandomForestClassifier(n_estimators=100, random_state=42))
    accuracy_m, y_pred_m = calculate_accuracy_and_predictions(X_train_m, X_test_m, y_train_m, y_test_m, RandomForestClassifier(n_estimators=100, random_state=42))
    # Display accuracy for each crop with a frame
    st.markdown('### Results')  # Header for the results
    st.write(f'Test Accuracy for Wheat: {accuracy_w}')
    st.write(f'Test Accuracy for Barley: {accuracy_b}')
    st.write(f'Test Accuracy for Maize Grain: {accuracy_m}')

    # Plot histogram
    all_accuracies = [accuracy_w, accuracy_b, accuracy_m]
    colors = ['blue', 'green', 'orange']
    fig, ax = plt.subplots()
    for i, (accuracy, color) in enumerate(zip(all_accuracies, colors)):
        ax.bar(['Wheat', 'Barley', 'Maize Grain'][i], accuracy, color=color, alpha=0.7)
    ax.set_title('Test Accuracy Distribution for Random Forest')
    ax.set_xlabel('Crop')
    ax.set_ylabel('Accuracy')
    st.pyplot(fig)

    # Create dataset with actual and predicted categories
    crop_types = ['Wheat', 'Barley', 'Maize Grain']
    results_df_w = pd.DataFrame({'Actual': y_test_w, 'Predicted': y_pred_w})
    results_df_w['Crop_type'] = crop_types[0]

    results_df_b = pd.DataFrame({'Actual': y_test_b, 'Predicted': y_pred_b})
    results_df_b['Crop_type'] = crop_types[1]

    results_df_m = pd.DataFrame({'Actual': y_test_m, 'Predicted': y_pred_m})
    results_df_m['Crop_type'] = crop_types[2]

    # Display the table on the left sidebar
    st.sidebar.write(results_df_w)
    st.sidebar.write(results_df_b)
    st.sidebar.write(results_df_m)
