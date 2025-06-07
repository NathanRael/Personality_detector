# RALAIVOAVY Natanael - 035I23 - DA2I L3

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import base64
import io
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import random
# Google collab link : https://colab.research.google.com/drive/1vG5VU261K6UBBDIU11dOXu63M9L1bbkQ?usp=sharing
# Github Repo : https://github.com/NathanRael/Personality_detector


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

        results = []
        for prob in predictions:
            p = float(prob[0])
            if p > 0.5:
                label = "Extrovert"
                confidence = p * 100
            else:
                label = "Introvert"
                confidence = (1 - p) * 100

            results.append(f"{label} ({confidence:.2f}%)")

        return results

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

class PersonalityDetectorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🧠 Détecteur de Personnalité")
        self.root.geometry("1200x800")
        self.root.configure(bg='#0f172a')
        
        self.questions = [
            ("🕒", "Heures passées seul par jour (0-11)?", 0, 11),
            ("🎭", "Avez-vous le trac sur scène?", 0, 1, True),
            ("🎉", "Fréquence des événements sociaux (0-10)?", 0, 10),
            ("🚶", "Sorties par semaine (0-7)?", 0, 7),
            ("😴", "Vous sentez-vous épuisé après socialiser?", 0, 1, True),
            ("👥", "Nombre d'amis proches (0-15)?", 0, 15),
            ("📱", "Publications sur réseaux sociaux (0-10)?", 0, 10)
        ]
        
        self.answers = []
        self.current_q = 0
        
        self.create_ui()
        self.show_question()
    
    def create_ui(self):
        self.main_frame = tk.Frame(self.root, bg='#1a1a2e')
        self.main_frame.pack(fill='both', expand=True, padx=40, pady=40)
        
        self.title = tk.Label(self.main_frame, text="✨ Détecteur de Personnalité ✨", 
                             font=('Segoe UI', 28, 'bold'), bg='#1a1a2e', fg='#00d4ff')
        self.title.pack(pady=30)
        
        self.question_frame = tk.Frame(self.main_frame, bg='#16213e', relief='flat', bd=0)
        self.question_frame.pack(fill='both', expand=True, pady=30)
    
    def show_question(self):
        for widget in self.question_frame.winfo_children():
            widget.destroy()
        
        if self.current_q >= len(self.questions):
            self.show_result()
            return
        
        q = self.questions[self.current_q]
        icon, text, min_val, max_val = q[0], q[1], q[2], q[3]
        is_boolean = len(q) > 4
        
        progress = tk.Label(self.question_frame, 
                           text=f"Question {self.current_q + 1} sur {len(self.questions)} 🎯",
                           font=('Segoe UI', 14, 'italic'), bg='#16213e', fg='#8892b0')
        progress.pack(pady=15)
        
        question_label = tk.Label(self.question_frame, text=f"{icon}\n{text}",
                                 font=('Segoe UI', 18, 'bold'), bg='#16213e', fg='#ccd6f6')
        question_label.pack(pady=40)
        
        if is_boolean:
            btn_frame = tk.Frame(self.question_frame, bg='#16213e')
            btn_frame.pack(pady=30)
            
            yes_btn = tk.Button(btn_frame, text="✨ OUI", font=('Segoe UI', 16, 'bold'),
                               bg='#64ffda', fg='#0a192f', padx=40, pady=15, relief='flat', bd=0,
                               command=lambda: self.answer(1))
            yes_btn.pack(side='left', padx=20)
            
            no_btn = tk.Button(btn_frame, text="❌ NON", font=('Segoe UI', 16, 'bold'),
                              bg='#ff6b9d', fg='white', padx=40, pady=15, relief='flat', bd=0,
                              command=lambda: self.answer(0))
            no_btn.pack(side='left', padx=20)
        else:
            self.scale_var = tk.IntVar(value=min_val)
            scale = tk.Scale(self.question_frame, from_=min_val, to=max_val,
                           orient='horizontal', length=300, font=('Segoe UI', 14, 'bold'),
                           bg='#233554', fg='#64ffda', troughcolor='#0f172a', variable=self.scale_var)
            scale.pack(pady=25)
            
            next_btn = tk.Button(self.question_frame, text="Suivant ➤",
                               font=('Segoe UI', 16, 'bold'), bg='#00d4ff', fg='#0a192f',
                               padx=50, pady=15, relief='flat', bd=0, command=lambda: self.answer(self.scale_var.get()))
            next_btn.pack(pady=25)
    
    def answer(self, value):
        self.answers.append(value)
        self.current_q += 1
        self.show_question()
    
    def show_result(self):
        for widget in self.question_frame.winfo_children():
            widget.destroy()
        
        model = PersonalityDetector()
        personality = model.predict(np.array([self.answers]))[0]
        print(f"Personality prediction: {personality}")
        color = "#ff6b9d" if "Extrovert" in personality else "#64ffda"
        
        result_label = tk.Label(self.question_frame, 
                               text=f"🎊 Vous êtes 🎊\n\n💫 {personality} 💫",
                               font=('Segoe UI', 24, 'bold'), bg='#16213e', fg=color)
        result_label.pack(pady=50)
        
        restart_btn = tk.Button(self.question_frame, text="🚀 Recommencer",
                               font=('Segoe UI', 16, 'bold'), bg='#a855f7', fg='white',
                               padx=40, pady=18, relief='flat', bd=0, command=self.restart)
        restart_btn.pack(pady=30)
    
    def restart(self):
        self.answers = []
        self.current_q = 0
        self.show_question()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    def run_gui():
        app = PersonalityDetectorGUI()
        app.run()
    run_gui()

    test = PersonalityDetector()
    
    # personality_detector_cli()
    
    # train_model()
    # test_data()