import tkinter as tk
from tkinter import ttk, messagebox
import random
import numpy as np
from PIL import Image, ImageTk
import io
import base64
from personality_detector import PersonalityDetector

class PersonalityDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Personality Detector")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a2e')
        
        # Language selection
        self.lang = "eng"
        
        # Question data
        self.questions = [
            {
                "key": "Time_spent_Alone",
                "question": {
                    "eng": "How many hours do you spend alone daily?",
                    "fr": "Combien d'heures passez-vous seul par jour ?"
                },
                "range": (0, 11),
                "icon": "🕒"
            },
            {
                "key": "Stage_fear",
                "question": {
                    "eng": "Do you experience stage fright?",
                    "fr": "Avez-vous le trac sur scène ?"
                },
                "range": (0, 1),
                "type": "boolean",
                "icon": "🎭"
            },
            {
                "key": "Social_event_attendance",
                "question": {
                    "eng": "How often do you attend social events?",
                    "fr": "À quelle fréquence participez-vous à des événements sociaux ?"
                },
                "range": (0, 10),
                "icon": "🎉"
            },
            {
                "key": "Going_outside",
                "question": {
                    "eng": "How often do you go outside per week?",
                    "fr": "À quelle fréquence sortez-vous par semaine ?"
                },
                "range": (0, 7),
                "icon": "🚶"
            },
            {
                "key": "Drained_after_socializing",
                "question": {
                    "eng": "Do you feel drained after socializing?",
                    "fr": "Vous sentez-vous épuisé après avoir socialisé ?"
                },
                "range": (0, 1),
                "type": "boolean",
                "icon": "😴"
            },
            {
                "key": "Friends_circle_size",
                "question": {
                    "eng": "How many close friends do you have?",
                    "fr": "Combien d'amis proches avez-vous ?"
                },
                "range": (0, 15),
                "icon": "👥"
            },
            {
                "key": "Post_frequency",
                "question": {
                    "eng": "How often do you post on social media?",
                    "fr": "À quelle fréquence publiez-vous sur les réseaux sociaux ?"
                },
                "range": (0, 10),
                "icon": "📱"
            }
        ]
        
        self.retrieved_data = {q["key"]: None for q in self.questions}
        self.current_question_index = 0
        
        self.setup_styles()
        self.create_widgets()
        self.show_welcome_screen()
    
    def setup_styles(self):
        """Setup custom styles for the GUI"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Title.TLabel', 
                       font=('Arial', 24, 'bold'),
                       background='#1a1a2e',
                       foreground='#ffffff')
        
        style.configure('Question.TLabel',
                       font=('Arial', 16),
                       background='#16213e',
                       foreground='#ffffff',
                       padding=20)
        
        style.configure('Custom.TButton',
                       font=('Arial', 12, 'bold'),
                       padding=10,
                       background='#0f3460',
                       foreground='#ffffff')
        
        style.map('Custom.TButton',
                 background=[('active', '#e94560'),
                           ('pressed', '#e94560')])
    
    def create_widgets(self):
        """Create the main GUI widgets"""
        # Main container
        self.main_frame = tk.Frame(self.root, bg='#1a1a2e')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        self.title_label = tk.Label(
            self.main_frame,
            text="🧠 Personality Detector",
            font=('Arial', 28, 'bold'),
            bg='#1a1a2e',
            fg='#ffffff'
        )
        self.title_label.pack(pady=20)
        
        # Content frame
        self.content_frame = tk.Frame(self.main_frame, bg='#1a1a2e')
        self.content_frame.pack(fill=tk.BOTH, expand=True)
    
    def show_welcome_screen(self):
        """Display the welcome screen with language selection"""
        self.clear_content()
        
        welcome_frame = tk.Frame(self.content_frame, bg='#16213e', relief=tk.RAISED, bd=2)
        welcome_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)
        
        # Welcome message
        welcome_text = tk.Label(
            welcome_frame,
            text="Welcome to the Personality Detector!\n\nDiscover whether you're an introvert or extrovert\nthrough our interactive questionnaire.",
            font=('Arial', 16),
            bg='#16213e',
            fg='#ffffff',
            justify=tk.CENTER
        )
        welcome_text.pack(pady=30)
        
        # Language selection
        lang_frame = tk.Frame(welcome_frame, bg='#16213e')
        lang_frame.pack(pady=20)
        
        tk.Label(
            lang_frame,
            text="Choose your language:",
            font=('Arial', 14, 'bold'),
            bg='#16213e',
            fg='#ffffff'
        ).pack(pady=10)
        
        # Language buttons
        button_frame = tk.Frame(lang_frame, bg='#16213e')
        button_frame.pack()
        
        eng_btn = tk.Button(
            button_frame,
            text="🇺🇸 English",
            font=('Arial', 12, 'bold'),
            bg='#0f3460',
            fg='#ffffff',
            padx=20,
            pady=10,
            command=lambda: self.set_language('eng'),
            cursor='hand2'
        )
        eng_btn.pack(side=tk.LEFT, padx=10)
        
        fr_btn = tk.Button(
            button_frame,
            text="🇫🇷 Français",
            font=('Arial', 12, 'bold'),
            bg='#0f3460',
            fg='#ffffff',
            padx=20,
            pady=10,
            command=lambda: self.set_language('fr'),
            cursor='hand2'
        )
        fr_btn.pack(side=tk.LEFT, padx=10)
    
    def set_language(self, lang):
        """Set the language and start the questionnaire"""
        self.lang = lang
        self.show_question()
    
    def show_question(self):
        """Display the current question"""
        if self.current_question_index >= len(self.questions):
            self.show_results()
            return
        
        self.clear_content()
        
        question_data = self.questions[self.current_question_index]
        question_text = question_data["question"][self.lang]
        question_range = question_data["range"]
        icon = question_data["icon"]
        
        # Question frame
        question_frame = tk.Frame(self.content_frame, bg='#16213e', relief=tk.RAISED, bd=2)
        question_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)
        
        # Progress bar
        progress_frame = tk.Frame(question_frame, bg='#16213e')
        progress_frame.pack(fill=tk.X, padx=20, pady=10)
        
        progress = tk.Label(
            progress_frame,
            text=f"Question {self.current_question_index + 1} of {len(self.questions)}",
            font=('Arial', 12),
            bg='#16213e',
            fg='#888888'
        )
        progress.pack()
        
        progress_bar = ttk.Progressbar(
            progress_frame,
            length=300,
            mode='determinate'
        )
        progress_bar['value'] = ((self.current_question_index + 1) / len(self.questions)) * 100
        progress_bar.pack(pady=5)
        
        # Question icon and text
        question_label = tk.Label(
            question_frame,
            text=f"{icon}\n\n{question_text}",
            font=('Arial', 18, 'bold'),
            bg='#16213e',
            fg='#ffffff',
            justify=tk.CENTER
        )
        question_label.pack(pady=30)
        
        # Input frame
        input_frame = tk.Frame(question_frame, bg='#16213e')
        input_frame.pack(pady=20)
        
        if question_data.get("type") == "boolean":
            # Yes/No buttons for boolean questions
            btn_frame = tk.Frame(input_frame, bg='#16213e')
            btn_frame.pack()
            
            yes_text = "Oui" if self.lang == "fr" else "Yes"
            no_text = "Non" if self.lang == "fr" else "No"
            
            yes_btn = tk.Button(
                btn_frame,
                text=f"✓ {yes_text}",
                font=('Arial', 14, 'bold'),
                bg='#27ae60',
                fg='#ffffff',
                padx=30,
                pady=15,
                command=lambda: self.answer_question(1),
                cursor='hand2'
            )
            yes_btn.pack(side=tk.LEFT, padx=10)
            
            no_btn = tk.Button(
                btn_frame,
                text=f"✗ {no_text}",
                font=('Arial', 14, 'bold'),
                bg='#e74c3c',
                fg='#ffffff',
                padx=30,
                pady=15,
                command=lambda: self.answer_question(0),
                cursor='hand2'
            )
            no_btn.pack(side=tk.LEFT, padx=10)
        else:
            # Scale input for numeric questions
            scale_label = tk.Label(
                input_frame,
                text=f"Range: {question_range[0]} - {question_range[1]}",
                font=('Arial', 12),
                bg='#16213e',
                fg='#888888'
            )
            scale_label.pack(pady=5)
            
            self.scale_var = tk.IntVar(value=question_range[0])
            scale = tk.Scale(
                input_frame,
                from_=question_range[0],
                to=question_range[1],
                orient=tk.HORIZONTAL,
                length=300,
                font=('Arial', 12, 'bold'),
                bg='#16213e',
                fg='#ffffff',
                highlightbackground='#16213e',
                variable=self.scale_var
            )
            scale.pack(pady=10)
            
            submit_btn = tk.Button(
                input_frame,
                text="Next ➤" if self.lang == "eng" else "Suivant ➤",
                font=('Arial', 14, 'bold'),
                bg='#e94560',
                fg='#ffffff',
                padx=30,
                pady=15,
                command=lambda: self.answer_question(self.scale_var.get()),
                cursor='hand2'
            )
            submit_btn.pack(pady=20)
    
    def answer_question(self, value):
        """Process the answer and move to next question"""
        question_key = self.questions[self.current_question_index]["key"]
        self.retrieved_data[question_key] = value
        self.current_question_index += 1
        self.show_question()
    
    def show_results(self):
        """Display the personality prediction results"""
        self.clear_content()
        
        # Calculate prediction
        personality_detector = PersonalityDetector()
        prediction = personality_detector.predict(np.array([list(self.retrieved_data.values())]))
        result = prediction[0].capitalize()
        
        # Results frame
        results_frame = tk.Frame(self.content_frame, bg='#16213e', relief=tk.RAISED, bd=2)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)
        
        # Results title
        title_text = "Résultats de l'analyse" if self.lang == "fr" else "Analysis Results"
        results_title = tk.Label(
            results_frame,
            text=title_text,
            font=('Arial', 24, 'bold'),
            bg='#16213e',
            fg='#ffffff'
        )
        results_title.pack(pady=20)
        
        # Personality result
        prefix = "D'après cette analyse, vous êtes un/une" if self.lang == "fr" else "Based on this analysis, you are an"
        
        result_label = tk.Label(
            results_frame,
            text=f"{prefix}\n\n🔹 {result.upper()} 🔹",
            font=('Arial', 20, 'bold'),
            bg='#16213e',
            fg='#e94560',
            justify=tk.CENTER
        )
        result_label.pack(pady=30)
        
        # Data summary
        summary_frame = tk.Frame(results_frame, bg='#0f3460', relief=tk.RAISED, bd=1)
        summary_frame.pack(fill=tk.X, padx=40, pady=20)
        
        summary_title = tk.Label(
            summary_frame,
            text="Your Responses:" if self.lang == "eng" else "Vos Réponses:",
            font=('Arial', 14, 'bold'),
            bg='#0f3460',
            fg='#ffffff'
        )
        summary_title.pack(pady=10)
        
        for i, question in enumerate(self.questions):
            key = question["key"]
            value = self.retrieved_data[key]
            icon = question["icon"]
            
            if question.get("type") == "boolean":
                display_value = "Yes/Oui" if value == 1 else "No/Non"
            else:
                display_value = str(value)
            
            response_label = tk.Label(
                summary_frame,
                text=f"{icon} {display_value}",
                font=('Arial', 11),
                bg='#0f3460',
                fg='#ffffff'
            )
            response_label.pack(pady=2)
        
        # Restart button
        restart_text = "Recommencer" if self.lang == "fr" else "Start Over"
        restart_btn = tk.Button(
            results_frame,
            text=f"🔄 {restart_text}",
            font=('Arial', 14, 'bold'),
            bg='#27ae60',
            fg='#ffffff',
            padx=30,
            pady=15,
            command=self.restart_quiz,
            cursor='hand2'
        )
        restart_btn.pack(pady=20)
    
    def restart_quiz(self):
        """Restart the personality quiz"""
        self.retrieved_data = {q["key"]: None for q in self.questions}
        self.current_question_index = 0
        self.show_welcome_screen()
    
    def clear_content(self):
        """Clear the content frame"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()

def personality_detector_gui():
    root = tk.Tk()
    app = PersonalityDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    personality_detector_gui()