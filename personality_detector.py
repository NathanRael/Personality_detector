import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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


if __name__ == "__main__":
    # train_model()
    test_data()