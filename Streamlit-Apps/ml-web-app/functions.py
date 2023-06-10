import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


@st.cache_data
def load_data(csv_file):
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        return df
    
    
def body(csv):
    df = load_data(csv)
    filtered_df = filter_clean_data(df)
    st.subheader("Data")
    st.write(filtered_df.head())

    # input_features(filtered_df)
    sidebar(filtered_df)
    train_models(filtered_df)


def sidebar(df):
    choose_explore_option(df)
    visualize_data(df)





def filter_clean_data(df):
    st.sidebar.header("Filter & Clean Data")

    columns = df.columns.tolist()
    selected_columns = st.sidebar.multiselect("Select columns", columns, columns)

    # Filter DataFrame based on selected columns
    filtered_df = df[selected_columns]

    # Drop nan values
    drop_nan = st.sidebar.checkbox('Drop NaN')

    if drop_nan:
        filtered_df.dropna()

    return filtered_df


def choose_explore_option(df):
    st.sidebar.header("Exploration & Visualization")

    explore_ops = st.sidebar.multiselect('Explore Data',
        ['shape', 'cols', 'describe', 'dtypes', 'nan_vals'],
        ['shape'])

    if "shape" in explore_ops:
        st.subheader("Shape")
        st.write(df.shape)

    if "cols" in explore_ops:
        st.subheader("Columns")
        st.write(df.columns)

    if "describe" in explore_ops:
        st.subheader("Describe")
        st.write(df.describe())

    if "dtypes" in explore_ops:
        st.subheader("dtypes")
        st.write(df.dtypes)

    if "nan_vals" in explore_ops:
        st.subheader("NaN Values")
        st.write(df.isnull().sum())

def visualize_data(df):

    visualize_ops = st.sidebar.multiselect(
        'Visualize Data',
        ['correlation_matrix', 'histogram', 'density'])
    
    if "correlation_matrix" in visualize_ops:

        st.subheader("Correlation Matrix")

        correlation_matrix = df.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(correlation_matrix.values, cmap='coolwarm')

        ax.set_xticks(np.arange(len(correlation_matrix.columns)))
        ax.set_yticks(np.arange(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns)
        ax.set_yticklabels(correlation_matrix.columns)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_title("Correlation Matrix")
        plt.colorbar(im)
        plt.show()
        st.pyplot(fig)

    if "histogram" in visualize_ops:
        st.subheader("Histogram")

        # Calculate the number of rows and columns for subplots
        num_columns = df.shape[1]
        num_rows = (num_columns + 2) // 3  # Adjust the number of columns as needed

        # Set the figure size
        fig, axes = plt.subplots(num_rows, 3, figsize=(12, 8))

        # Iterate over each column in the dataset
        for i, column in enumerate(df.columns):
            # Calculate the row and column index for the current subplot
            row_idx = i // 3
            col_idx = i % 3

            # Select the appropriate axis for the current subplot
            if num_rows > 1:
                ax = axes[row_idx, col_idx]
            else:
                ax = axes[col_idx]

            # Plot the histogram for the current column
            sns.histplot(df[column], bins=10, ax=ax)  # Adjust the number of bins as needed

            # Set plot title and labels
            ax.set_title(column)
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)



    if "density" in visualize_ops:
        st.subheader("Density")

        # Calculate the number of rows and columns for subplots
        num_columns = df.shape[1]
        num_rows = (num_columns + 2) // 3  # Adjust the number of columns as needed

        # Set the figure size
        fig, axes = plt.subplots(num_rows, 3, figsize=(12, 8))

        # Iterate over each column in the dataset
        for i, column in enumerate(df.columns):
            # Calculate the row and column index for the current subplot
            row_idx = i // 3
            col_idx = i % 3

            # Select the appropriate axis for the current subplot
            if num_rows > 1:
                ax = axes[row_idx, col_idx]
            else:
                ax = axes[col_idx]

            # Plot the density plot for the current column
            sns.kdeplot(df[column], shade=True, ax=ax)

            # Set plot title and labels
            ax.set_title(column)
            ax.set_xlabel(column)
            ax.set_ylabel("Density")

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)



def split_data(df):
    st.subheader("Train Test Split")
    st.sidebar.header("Split Data")
    columns = df.columns.tolist()
    target = st.sidebar.multiselect("Select target", columns, max_selections=1, default=columns[-1])

    X = df.drop(target, axis=1)
    y = df[target]

    # Normalize and Standardize Data
    norm_standardize = st.sidebar.radio(
    "Norm/Standardize Data",
    ('Normalize', 'Standardize', 'None'))

    if norm_standardize == 'Normalize':
        scaler = MinMaxScaler()
        df_normalized = scaler.fit_transform(X)
        df = pd.DataFrame(df_normalized)
    
    elif norm_standardize == 'Standardize':
        scaler = StandardScaler()
        df_standardize = scaler.fit_transform(X)
        df = pd.DataFrame(df_standardize)
    else:
        scaler = None

    test_size = st.sidebar.number_input("test size", step=0.01, max_value=1.0, value=0.2)
    random_state = st.sidebar.number_input("random state", step=1, max_value=200, value=42)

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=test_size, random_state=random_state)

    st.write("Training set - X_train shape:", X_train.shape)
    st.write("Training set - y_train shape:", y_train.shape)
    st.write("Testing set - X_test shape:", X_test.shape)
    st.write("Testing set - y_test shape:", y_test.shape)

    return X_train, X_test, y_train, y_test, scaler




def train_models(df):

    # split data
    X_train, X_test, y_train, y_test, scaler = split_data(df)

    st.sidebar.header("Train Model")
    classifiers = ['RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier', 'LogisticRegression']
    model_ops = st.sidebar.radio('Choose Model', classifiers)

    # choose hyperparameters
    st.sidebar.write(f"-- {model_ops} Hyperparameters --")
    hyperparameters = {}

    if model_ops == 'RandomForestClassifier':
        hyperparameters = {'n_estimators': None, 'max_features': None, 'max_depth': None, 'min_samples_split': None}
    elif model_ops == 'XGBClassifier':
        hyperparameters = {'learning_rate': None, 'max_depth': None}
    elif model_ops == 'LGBMClassifier':
        hyperparameters = {'learning_rate': None, 'max_depth': None, 'num_leaves': None}
    elif model_ops == 'LogisticRegression':
        hyperparameters = {'max_iter': None}

    for hyperparam in hyperparameters:
        if hyperparam == 'learning_rate':
            number = st.sidebar.number_input(f"{hyperparam}", step=0.01, max_value=1.0, value=0.01, key=f"{model_ops}-{hyperparam}")
        elif hyperparam == 'max_iter':
            number = st.sidebar.number_input(f"{hyperparam}", step=1, max_value=10000,value=100, key=f"{model_ops}-{hyperparam}")
        else:
            number = st.sidebar.number_input(f"{hyperparam}", step=1, max_value=1000,value=10, key=f"{model_ops}-{hyperparam}")
        hyperparameters[hyperparam] = number

    st.write('\n')


    if "RandomForestClassifier" in model_ops:
        st.subheader("Classifier: RandomForestClassifier")
        model = RandomForestClassifier(**hyperparameters)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.subheader("Evaluation")
        st.table(report_df)

    if "XGBClassifier" in model_ops:
        st.subheader("Classifier: XGBClassifier")
        model = XGBClassifier(**hyperparameters)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.subheader("Evaluation")
        st.table(report_df)


    if "LGBMClassifier" in model_ops:
        st.subheader("Classifier: LGBMClassifier")
        model = lgb.LGBMClassifier(**hyperparameters)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.subheader("Evaluation")
        st.table(report_df)


    if "LogisticRegression" in model_ops:
        st.subheader("Classifier: LogisticRegression")
        model = LogisticRegression(**hyperparameters)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.subheader("Evaluation")
        st.table(report_df)


    st.subheader("Prediction")
    predict(df, scaler, model)




def predict(df, scaler, model):
    st.sidebar.header("Predict")
    inputs = input_features(df)
    inputs = scaler.transform(inputs)

    pred = model.predict(inputs)
    st.success(f"Predicted Value: {pred[0]}")

    st.header("Created by: Abdullah Khaled")
    st.subheader("Phone: +201557504902")





def input_features(df):
    inputs = []
    
    for col in df.columns[:-1]:
        col_min = df[col].min().item()
        col_max = df[col].max().item()
        col_mean = df[col].mean().item()

        if df[col].dtype == 'float':
            value = st.sidebar.slider(f'Enter {col}', float(col_min), float(col_max), float(col_mean), step=(col_max - col_min) / 100, key=f'{col}_slider')
            value = round(value, 2)
        else:
            value = st.sidebar.slider(f'Enter {col}', int(col_min), int(col_max), int(col_mean), step=1, key=f'{col}_slider')

        inputs.append(value)
    
    inputs = np.reshape(inputs, (1, -1))

    return inputs
