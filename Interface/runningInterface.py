import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import HCI
import joblib

# Loading models
svm = joblib.load('../svmACDCT_model.pkl')
rf = joblib.load('../RandomForest_ACDCT_model.pkl')
svm1 = joblib.load('../svm_wavelet_model.pkl')
rfw = joblib.load('../RandomForest_wavelet_model.pkl')

def main():
    root = tk.Tk()
    root.title("ECG Signal Uploader")
    
    # Set window size and position to full screen
    root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
    
    # Set background color
    root.configure(bg="#ADD8E6")  # Light blue color
    
    # Welcome message
    welcome_label = tk.Label(root, text="Welcome to ECG Signal Signal Identifier Interface!", font=("Helvetica", 25), bg="#ADD8E6")
    welcome_label.pack(pady=(root.winfo_screenheight()//4, 20))
    
    # Label for file selection
    file_label = tk.Label(root, text="Please select your ECG signal file:", font=("Helvetica", 20), bg="#ADD8E6")
    file_label.pack(anchor='center')
    
    # Button to select file
    file_path_var = tk.StringVar()
    file_button = ttk.Button(root, text="Browse", command=lambda: browse_file(file_path_var))
    file_button.pack(pady=10, anchor='center')
    
    # Input field to display selected file path
    file_path_entry = tk.Entry(root, textvariable=file_path_var, state='readonly', width=50)
    file_path_entry.pack(pady=5, anchor='center')
    
    # Button to submit
    submit_button = ttk.Button(root, text="Submit", command=lambda: handle_submit(file_path_var, root))
    submit_button.pack(pady=10, anchor='center')
    
    root.mainloop()

def browse_file(file_path_var):
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    file_path_var.set(file_path)

def submit_file(file_path_var):
    freq = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    file_path = file_path_var.get()
    if file_path:
        # Process the file here (e.g., load and analyze the ECG signal)
        sub1X, sub1Y = HCI.loadSubjectSamples(file_path)
        filtered_signal_sub_1 = HCI.filterSignal(sub1Y)
        segmented_signal_sub_1 = HCI.SegmentHeartBeats(filtered_signal_sub_1, sub1X)
        segmented_signal_sub_1 = HCI.concatenate_beats(segmented_signal_sub_1)

        dct1 = HCI.AutoCorrelate(segmented_signal_sub_1)
        dct1 = HCI.Normalize_Features(dct1)
        sub1_dct = HCI.Unify_Electrodes(dct1)

        wl1 = HCI.Wavelet(segmented_signal_sub_1)
        sub1_Wavelet = HCI.Unify_Electrodes(wl1)

        tmp = HCI.identify_subject(svm, sub1_dct, 0.5)
        freq[tmp] = freq[tmp] + 1
        tmp = HCI.identify_subject(rf, sub1_dct, 0.5)
        freq[tmp] = freq[tmp] + 1
        tmp = HCI.identify_subject(svm1, sub1_Wavelet, 0.5)
        freq[tmp] = freq[tmp] + 1
        tmp = HCI.identify_subject(rfw, sub1_Wavelet, 0.5)
        freq[tmp] = freq[tmp] + 1
        key = max(freq, key=lambda k: freq[k])
        print(freq[5])
        if freq[key]>2: return key
        else: return 5
    else:
        print("Please select a file.")

def handle_submit(file_path_var, root):
    result = submit_file(file_path_var)
    if result in range(1, 5):
        # Display picture and welcome message
        image_path = f"subject{result}.png"  # Assuming images are named subject1.png, subject2.png, etc.
        display_image_and_message(image_path, root)
    elif result == 5:
        # Inform user that they are not identified
        display_not_identified_message(root)
    else:
        # Handle unexpected result
        print("Unexpected result received.")


from PIL import Image, ImageTk

def display_image_and_message(image_path, root):
    try:
        # Create a new window
        image_window = tk.Toplevel(root)
        image_window.title("Welcome")
        image_window.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))

        # Set background color
        image_window.configure(bg="#ADD8E6")  # Light blue color

        # Load image with PIL
        image_pil = Image.open(image_path)
        window_width = root.winfo_screenwidth()
        window_height = root.winfo_screenheight()
        image_width = int(window_width * 0.4)  # Resize image to 40% of window width
        image_height = int(image_width * image_pil.height / image_pil.width)
        image_resized = image_pil.resize((image_width, image_height))
        photo = ImageTk.PhotoImage(image_resized)

        # Display image
        image_label = tk.Label(image_window, image=photo)
        image_label.pack(pady=(window_height * 0.1, window_height * 0.05))  # Center image vertically

        # Keep reference to avoid garbage collection
        image_label.image = photo

        # Display welcome message
        welcome_message_label = tk.Label(image_window, text="Welcome Dear User!", font=("Helvetica", 20), bg="#ADD8E6")
        welcome_message_label.pack()
        welcome_message_label.place(relx=0.5, rely=0.87, anchor=tk.CENTER)  # Center welcome message horizontally and place below image

    except Exception as e:
        print(f"Error loading image '{image_path}': {str(e)}")



def display_not_identified_message(root):
    # Create a new window
    not_identified_window = tk.Toplevel(root)
    not_identified_window.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
    not_identified_window.title("Not Identified")
    
    # Set background color
    not_identified_window.configure(bg="#ADD8E6")  # Light blue color
    
    # Display message
    not_identified_message_label = tk.Label(not_identified_window, text="You are not identified, you imposter", font=("Helvetica", 20), bg="#ADD8E6")
    not_identified_message_label.pack()

if __name__ == "__main__":
    main()
