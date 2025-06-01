import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random

# Google collab link : https://colab.research.google.com/drive/1vG5VU261K6UBBDIU11dOXu63M9L1bbkQ?usp=sharing


class PersonalityDetector:
    dataset_path = "personality_dataset.csv"

    def train(self):
        raw_df = self._load_dataset()
        df = self._normalize_dataset(raw_df)
        self._train_model(df)

    def predict(self, data):
        model = tf.keras.models.load_model("personality_detector_model.h5")
        predictions = self._predict(model, data)
        return predictions

    def _load_dataset(self):
        return pd.read_csv(self.dataset_path)

    @staticmethod
    def _normalize_dataset(raw_df):
        str_cols_missing_value = [
            "Stage_fear",
            "Drained_after_socializing"
        ]
        float_cols_missing_value = [
            "Time_spent_Alone",
            "Social_event_attendance",
            "Going_outside",
            "Friends_circle_size",
            "Post_frequency",
        ]
        all_cols_missing_value = str_cols_missing_value + float_cols_missing_value

        df = raw_df.copy()
        for col in all_cols_missing_value:
            if col in str_cols_missing_value:
                df[col] = df[col].str.upper().map({'YES': 1, 'NO': 0})
            df[col] = df[col].fillna(df[col].mean())

        df['Personality'] = df['Personality'].map({'Introvert': 0, 'Extrovert': 1})

        return df

    @staticmethod
    def _predict(model, data_test):
        predictions = model.predict(data_test)
        return np.where(predictions > 0.5, "Extrovert", "Introvert").flatten().tolist()

    def _train_model(self,df):
        X = df[df.columns[:-1]].values
        y = df[df.columns[-1]].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])
        evaluation = model.evaluate(X_train, y_train)
        print(f'Model evaluation result: {evaluation}')

        print("--Before training--")
        print(self._predict(model=model,data_test=X_test[:5]))

        model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(X_test, y_test))
        model.save("personality_detector_model.h5")

        print("--After training--")
        print(self._predict(model=model,data_test=X_test[:5]))
        return model
    

def train_model():
    model = PersonalityDetector()
    model.train()


def test_data():
    model = PersonalityDetector()
    # Data from the test dataset
    test_data = [
        [10, 1, 3, 3, 1, 5, 3],  # Should be Introvert
        [4.505816, 0.0, 3.96335447, 5, 0, 14, 5],  # Should be Extrovert
        [5,1, 0, 0,1,0,2], # Introvert
        [10,1,2,2,1,2,1] # Introvert
    ]
    
    array = np.array(test_data)
    predictions = model.predict(array)

    print("Prediction", predictions) 

    for i, data in enumerate(predictions):
        print(f"Data {test_data[i]} => prediction :  {data}")



def conversational_prediction():
    print("Welcome to the Personality Detector!\n\n")
    lang = input("Choose a language (fr, eng): ").strip().lower()
    if lang not in ["fr", "eng"]:
        print("Invalid language. Defaulting to English.")
        lang = "eng"

    questions = [
        {
            "key": "Time_spent_Alone",
            "question": {
                "eng": "How many hours do you spend alone daily (0–11)?",
                "fr": "Combien d'heures passez-vous seul par jour (0–11) ?"
            },
            "range": (0, 11)
        },
        {
            "key": "Stage_fear",
            "question": {
                "eng": "Do you experience stage fright? (1 for YES, 0 for NO)",
                "fr": "Avez-vous le trac sur scène ? (1 pour OUI, 0 pour NON)"
            },
            "range": (0, 1)
        },
        {
            "key": "Social_event_attendance",
            "question": {
                "eng": "How often do you attend social events (0–10)?",
                "fr": "À quelle fréquence participez-vous à des événements sociaux (0–10) ?"
            },
            "range": (0, 10)
        },
        {
            "key": "Going_outside",
            "question": {
                "eng": "How often do you go outside per week (0–7)?",
                "fr": "À quelle fréquence sortez-vous par semaine (0–7) ?"
            },
            "range": (0, 7)
        },
        {
            "key": "Drained_after_socializing",
            "question": {
                "eng": "Do you feel drained after socializing? (1 for YES, 0 for NO)",
                "fr": "Vous sentez-vous épuisé après avoir socialisé ? (1 pour OUI, 0 pour NON)"
            },
            "range": (0, 1)
        },
        {
            "key": "Friends_circle_size",
            "question": {
                "eng": "How many close friends do you have (0–15)?",
                "fr": "Combien d'amis proches avez-vous (0–15) ?"
            },
            "range": (0, 15)
        },
        {
            "key": "Post_frequency",
            "question": {
                "eng": "How often do you post on social media (0–10)?",
                "fr": "À quelle fréquence publiez-vous sur les réseaux sociaux (0–10) ?"
            },
            "range": (0, 10)
        }
    ]

    retrievedData = {
        "Time_spent_Alone": None,
        "Stage_fear": None,
        "Social_event_attendance": None,
        "Going_outside": None,
        "Drained_after_socializing": None,
        "Friends_circle_size": None,
        "Post_frequency": None
    }

    conversation_history = ""

    while not all(value is not None for value in retrievedData.values()):
        
        unfilled_keys = [q for q in questions if retrievedData[q["key"]] is None]
        if not unfilled_keys:
            break
        question_data = random.choice(unfilled_keys)
        key = question_data["key"]
        question = question_data["question"][lang]
        valid_range = question_data["range"]

        print("BOT:", question)
        userMessage = input("You: ")

        try:
            # Validate user input
            value = userMessage.strip()
            if value.lower() in ["none", ""]:
                print("BOT: Please provide a valid number.")
                continue

            value = int(value)  # Convert to integer
            if not (valid_range[0] <= value <= valid_range[1]):
                print(f"BOT: Please enter a number between {valid_range[0]} and {valid_range[1]}.")
                continue

            # Update retrieved data
            retrievedData[key] = value
            conversation_history += f"User: {userMessage}\nBOT: {question}\n"

        except ValueError:
            print("BOT: Invalid input. Please enter a valid number.")
            continue

    personalityDetector = PersonalityDetector()
    prediction = personalityDetector.predict(np.array([list(retrievedData.values())]))
    prefix = "D'après cette conversation, vous êtes un/une" if lang == "fr" else "Based on the conversation, you are an"
    result = prediction[0].capitalize()
    print("\n" + "=" * 50)
    print(f"{prefix.center(50)}")
    print(f"{('🔹 ' + result + ' 🔹').center(50)}")
    print("=" * 50 + "\n")

    return retrievedData




if __name__ == "__main__":
    # train_model()
    # test_data()
    conversational_prediction()