
import streamlit as st
import joblib


# Function to load the trained model from a pickle file
def load_model(filename):
    try:
        model = joblib.load(filename)
        return model
    except Exception as e:
        # Print the exception if any, it will help us to know what went wrong
        print(f"An error occurred while loading the model: {e}")
        raise


# Function to make predictions based on model and user input
def predict_with_model(model, user_input):
    try:
        prediction = model.predict([user_input])
        return prediction[0]
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        raise


def main():
    model = load_model('Best_class.pkl')
    user_input = [0,1,0,1,1,1,0,1,0,0,0,1]  # Example input
    prediction = predict_with_model(model, user_input)
    print(f"The predicted output is: {prediction}")


if __name__ == "__main__":
    main()
    