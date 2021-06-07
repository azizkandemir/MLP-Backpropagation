from tkinter import ttk, filedialog, Tk, N, W, E, S, RIDGE, Label, DISABLED, ACTIVE, END
from tkinter import messagebox as msg

from functions import ActivationFunction
from mlp_model import MLPClassification, MLPRegression
from utils import Utils


class MLPGuiTest:
    def __init__(self, trained_mlp):
        self.window = Tk()
        self.mlp = trained_mlp
        self.test_file = None
        self.output_path = ''
        self.input_frame_entry_box = None  # Input File Path
        self.run_button = None
        self.browse_input_button = None
        self.create_widgets()

    def create_widgets(self):
        window = self.window

        style = ttk.Style()
        style.configure('TLabelframe.Label', font=('courier', 18, 'bold'))

        window.title("MLP Test")
        mainframe = ttk.Frame(window)
        mainframe.grid(column=10, row=10, sticky=(N, W, E, S))

        # Title Frame
        title_frame = ttk.LabelFrame(window, relief=RIDGE, padding="50 10 50 10")
        title_frame.grid(column=1, row=0, columnspan=2, sticky=N, padx=100, pady=20)
        # Title Label
        title_label = Label(title_frame, text=" MLP - BackPropagation", font=("Arial Bold", 25))
        title_label.grid(column=2, row=1, sticky=N)

        author_label = Label(title_frame, text="Aziz Mahmut Kandemir | Kuntur Soveatin")
        author_label.grid(column=2, row=2, sticky=N)

        # Main Frame

        main_frame = ttk.LabelFrame(window, text='Test MLP', padding="30 10 30 10", style="TLabelframe")
        main_frame.grid(column=1, row=1, sticky=N, padx=20, pady=20, ipady=14)

        # Intro Label
        intro_label = Label(main_frame, text="Provide test dataset file in the form of an external csv file.\n\n "
                                             "Provided dataset will be used to test MLP model.", font=("Arial", 12))
        intro_label.grid(column=1, row=0, sticky=N)

        # Input Frame
        input_frame = ttk.LabelFrame(main_frame, text='Load Test Dataset', padding="30 10 30 10", style="TLabelframe")
        input_frame.grid(column=1, row=1, sticky=W, padx=20, pady=20, ipady=14)

        # Input Frame Label
        input_frame_label = Label(input_frame, text="Dataset File Path: ", font=("Arial Bold", 14))
        input_frame_entry_box = ttk.Entry(input_frame, width=40)

        input_frame_label.grid(column=1, row=3, sticky=W)
        input_frame_entry_box.grid(column=2, row=3, sticky=W)
        self.input_frame_entry_box = input_frame_entry_box

        browse_input_button = ttk.Button(master=input_frame, text="BROWSE", command=self.browse_input)
        browse_input_button.grid(column=2, row=4, sticky=E, pady=10, ipadx=2)
        self.browse_input_button = browse_input_button

        run_button = ttk.Button(master=main_frame, text="TEST", command=self.run)
        run_button.grid(column=1, row=9, sticky=N, pady=10, ipadx=5)
        run_button['state'] = DISABLED
        self.run_button = run_button

        window.protocol("WM_DELETE_WINDOW", self.on_closing)
        window.mainloop()

    def browse_input(self):
        path = filedialog.askopenfilename(title="Select file")
        if not path.endswith('.csv'):
            msg.showinfo(title="Error", message="File type must be .csv")
        else:
            self.input_frame_entry_box.delete(0, END)
            self.input_frame_entry_box.insert(0, path)
            self.test_file = InputFile(path)
            self.run_button['state'] = ACTIVE

    def run(self):
        mlp_model = self.mlp
        mlp_model.test(self.test_file.get_file_path())

    def terminate(self):
        self.window.destroy()

    def on_closing(self):
        if msg.askokcancel("Quit", "Do you want to quit?"):
            self.terminate()


class MLPGuiTrain:
    def __init__(self):
        self.window = Tk()
        self.train_file = None
        self.input_frame_entry_box = None  # Input File Path
        self.hidden_layer_num_entry_box = None  # Number of Vectors
        self.hidden_layer_size_entry_box = None  # Size of Vectors
        self.hidden_layer_activation_function_combobox = None  # Hidden Layer Activation Function
        self.output_layer_activation_function_combobox = None  # Output Layer Activation Function
        self.bias_presence_combobox = None  # Bias Presence
        self.batch_size_entry_box = None  # Batch Size
        self.number_of_epochs_entry_box = None  # Num of Epochs
        self.learning_rate_entry_box = None  # Learning Rate
        self.momentum_entry_box = None  # Momentum
        self.problem_type_combobox = None  # Problem Type: Classification / Regression
        self.run_button = None
        self.browse_input_button = None
        self.generate_input_button = None
        self.trained_model = None
        self.create_widgets()

    def create_widgets(self):
        window = self.window

        style = ttk.Style()
        style.configure('TLabelframe.Label', font=('courier', 18, 'bold'))

        window.title("MLP Train")
        mainframe = ttk.Frame(window)
        mainframe.grid(column=10, row=10, sticky=(N, W, E, S))

        # Title Frame
        title_frame = ttk.LabelFrame(window, relief=RIDGE, padding="50 10 50 10")
        title_frame.grid(column=1, row=0, columnspan=2, sticky=N, padx=100, pady=20)
        # Title Label
        title_label = Label(title_frame, text=" MLP - BackPropagation", font=("Arial Bold", 25))
        title_label.grid(column=2, row=1, sticky=N)

        author_label = Label(title_frame, text="Aziz Mahmut Kandemir | Kuntur Soveatin")
        author_label.grid(column=2, row=2, sticky=N)

        # Main Frame

        main_frame = ttk.LabelFrame(window, text='Train MLP', padding="30 10 30 10", style="TLabelframe")
        main_frame.grid(column=1, row=1, sticky=N, padx=20, pady=20, ipady=14)

        # Intro Label
        intro_label = Label(main_frame, text="Provide train dataset file in the form of an external csv file.\n\n "
                                             "Provided dataset will be used to train MLP model.", font=("Arial", 12))
        intro_label.grid(column=1, row=0, sticky=N)

        # Input Frame
        input_frame = ttk.LabelFrame(main_frame, text='Load Train Dataset', padding="30 10 30 10", style="TLabelframe")
        input_frame.grid(column=1, row=1, sticky=W, padx=20, pady=20, ipady=14)

        # Input Frame Label
        input_frame_label = Label(input_frame, text="Dataset File Path: ", font=("Arial Bold", 14))
        input_frame_entry_box = ttk.Entry(input_frame, width=40)

        input_frame_label.grid(column=1, row=3, sticky=W)
        input_frame_entry_box.grid(column=2, row=3, sticky=W)
        self.input_frame_entry_box = input_frame_entry_box

        browse_input_button = ttk.Button(master=input_frame, text="BROWSE", command=self.browse_input)
        browse_input_button.grid(column=2, row=4, sticky=E, pady=10, ipadx=2)
        self.browse_input_button = browse_input_button
    
        # Input Frame GUI
        input_frame_gui = ttk.LabelFrame(main_frame, text='Input Parameters', padding="30 10 30 10", style="TLabelframe")
        input_frame_gui.grid(column=1, row=2, sticky=N, padx=20, pady=20)

        hidden_layer_num_label = Label(input_frame_gui, text="Number of Hidden Layers: ", font=("Arial Bold", 14))
        hidden_layer_num_entry_box = ttk.Entry(input_frame_gui, width=14)

        hidden_layer_num_label.grid(column=3, row=3, sticky=E)
        hidden_layer_num_entry_box.grid(column=4, row=3, sticky=E)
        self.hidden_layer_num_entry_box = hidden_layer_num_entry_box

        hidden_layer_size_label = Label(input_frame_gui, text="Hidden Layer Size: ", font=("Arial Bold", 14))
        hidden_layer_size_entry_box = ttk.Entry(input_frame_gui, width=14)

        hidden_layer_size_label.grid(column=3, row=4, sticky=E)
        hidden_layer_size_entry_box.grid(column=4, row=4, sticky=E)
        self.hidden_layer_size_entry_box = hidden_layer_size_entry_box

        hidden_layer_activation_function_label = Label(input_frame_gui, text="Hidden Layer Activation Function: ", font=("Arial Bold", 14))
        hidden_layer_activation_function_combobox = ttk.Combobox(master=input_frame_gui, width=12, state="readonly")
        hidden_layer_activation_function_combobox['values'] = ("Sigmoid", "tanh", "ReLU", "Linear")
        hidden_layer_activation_function_combobox.current(0)

        hidden_layer_activation_function_label.grid(column=3, row=5, sticky=E)
        hidden_layer_activation_function_combobox.grid(column=4, row=5, sticky=E)
        self.hidden_layer_activation_function_combobox = hidden_layer_activation_function_combobox

        output_layer_activation_function_label = Label(input_frame_gui, text="Output Layer Activation Function: ", font=("Arial Bold", 14))
        output_layer_activation_function_combobox = ttk.Combobox(master=input_frame_gui, width=12, state="readonly")
        output_layer_activation_function_combobox['values'] = ("Sigmoid", "tanh", "ReLU", "Linear")
        output_layer_activation_function_combobox.current(0)

        output_layer_activation_function_label.grid(column=3, row=6, sticky=E)
        output_layer_activation_function_combobox.grid(column=4, row=6, sticky=E)
        self.output_layer_activation_function_combobox = output_layer_activation_function_combobox

        bias_presence_label = Label(input_frame_gui, text="Bias Presence: ", font=("Arial Bold", 14))
        bias_presence_combobox = ttk.Combobox(master=input_frame_gui, width=12, state="readonly")
        bias_presence_combobox['values'] = ("Yes", "No")
        bias_presence_combobox.current(0)

        bias_presence_label.grid(column=3, row=7, sticky=E)
        bias_presence_combobox.grid(column=4, row=7, sticky=E)
        self.bias_presence_combobox = bias_presence_combobox

        batch_size_label = Label(input_frame_gui, text="Batch Size: ", font=("Arial Bold", 14))
        batch_size_entry_box = ttk.Entry(input_frame_gui, width=14)

        batch_size_label.grid(column=3, row=8, sticky=E)
        batch_size_entry_box.grid(column=4, row=8, sticky=E)
        self.batch_size_entry_box = batch_size_entry_box

        number_of_epochs_label = Label(input_frame_gui, text="Number of Epochs: ", font=("Arial Bold", 14))
        number_of_epochs_entry_box = ttk.Entry(input_frame_gui, width=14)

        number_of_epochs_label.grid(column=3, row=9, sticky=E)
        number_of_epochs_entry_box.grid(column=4, row=9, sticky=E)
        self.number_of_epochs_entry_box = number_of_epochs_entry_box

        learning_rate_label = Label(input_frame_gui, text="Learning Rate: ", font=("Arial Bold", 14))
        learning_rate_entry_box = ttk.Entry(input_frame_gui, width=14)

        learning_rate_label.grid(column=3, row=10, sticky=E)
        learning_rate_entry_box.grid(column=4, row=10, sticky=E)
        self.learning_rate_entry_box = learning_rate_entry_box

        momentum_label = Label(input_frame_gui, text="Momentum: ", font=("Arial Bold", 14))
        momentum_entry_box = ttk.Entry(input_frame_gui, width=14)

        momentum_label.grid(column=3, row=11, sticky=E)
        momentum_entry_box.grid(column=4, row=11, sticky=E)
        self.momentum_entry_box = momentum_entry_box

        problem_type_label = Label(input_frame_gui, text="Problem Type: ", font=("Arial Bold", 14))
        problem_type_combobox = ttk.Combobox(master=input_frame_gui, width=12, state="readonly")
        problem_type_combobox['values'] = ("Classification", "Regression")
        problem_type_combobox.current(0)

        problem_type_label.grid(column=3, row=12, sticky=E)
        problem_type_combobox.grid(column=4, row=12, sticky=E)
        self.problem_type_combobox = problem_type_combobox

        run_button = ttk.Button(master=main_frame, text="TRAIN", command=self.run)
        run_button.grid(column=1, row=9, sticky=N, pady=10, ipadx=5)
        run_button['state'] = DISABLED
        self.run_button = run_button

        window.protocol("WM_DELETE_WINDOW", self.on_closing)
        window.mainloop()

    def browse_input(self):
        path = filedialog.askopenfilename(title="Select file")
        if not path.endswith('.csv'):
            msg.showinfo(title="Error", message="File type must be .csv")
        else:
            self.input_frame_entry_box.delete(0, END)
            self.input_frame_entry_box.insert(0, path)
            self.train_file = InputFile(path, is_train_dataset=True)
            self.run_button['state'] = ACTIVE

    def run(self):
        if (hidden_layer_count := Utils.cast_int(self.hidden_layer_num_entry_box.get())) is None:
            msg.showerror(title="ERROR", message="NUMBER OF HIDDEN LAYERS MUST BE AN INTEGER!")
        elif (hidden_layer_size := Utils.cast_int(self.hidden_layer_size_entry_box.get())) is None:
            msg.showerror(title="ERROR", message="HIDDEN LAYER SIZE MUST BE AN INTEGER!")
        elif (batch_size := Utils.cast_int(self.batch_size_entry_box.get())) is None:
            msg.showerror(title="ERROR", message="BATCH SIZE MUST BE AN INTEGER!")
        elif (epochs := Utils.cast_int(self.number_of_epochs_entry_box.get())) is None:
            msg.showerror(title="ERROR", message="NUMBER OF EPOCHS MUST BE AN INTEGER!")
        elif (learning_rate := Utils.cast_float(self.learning_rate_entry_box.get())) is None:
            msg.showerror(title="ERROR", message="LEARNING RATE MUST BE A NUMERIC VALUE!")
        elif (momentum := Utils.cast_float(self.momentum_entry_box.get())) is None:
            msg.showerror(title="ERROR", message="MOMENTUM SIZE MUST BE A NUMERIC VALUE!")
        else:
            hidden_layer_activation_function = ActivationFunction.determine_function(self.hidden_layer_activation_function_combobox.get())
            output_layer_activation_function = ActivationFunction.determine_function(self.output_layer_activation_function_combobox.get())
            bias_presence = self.bias_presence_combobox.get()
            problem_type = self.problem_type_combobox.get()
            train_file_path = self.train_file.get_file_path()
            if problem_type.lower() == 'classification':
                solution_model = MLPClassification
            else:
                solution_model = MLPRegression

            mlp_model = solution_model(hidden_layer_count, hidden_layer_size, hidden_layer_activation_function,
                                       output_layer_activation_function, train_file_path,
                                       epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                       bias_presence=bias_presence, momentum=momentum)
            mlp_model.train()
            self.trained_model = mlp_model
            # self.terminate()

    def terminate(self):
        self.window.destroy()

    def on_closing(self):
        if msg.askokcancel("Quit", "Do you want to quit?"):
            self.terminate()


class InputFile:
    def __init__(self, input_file_path, is_train_dataset=False):
        self.file_path = input_file_path

    def get_file_path(self):
        return self.file_path


if __name__ == "__main__":
    train_mlp = MLPGuiTrain()
    if train_mlp.trained_model:
        test_mlp = MLPGuiTest(train_mlp.trained_model)
