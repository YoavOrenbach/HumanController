import tkinter as tk
#from tkinter import ttk
from tkinter import messagebox as mb
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tokenize import String
from PIL import Image, ImageTk
from cv2 import cv2
import os
import shutil
import re
from create_dataset import collect_key_data, create_data_folder
from train_model import controller_model
from predict_and_play import print_poses


KEYS = [
    [
        # =========================================
        # ===== Keyboard Configurations ===========
        # =========================================

        [
            # Layout Name
            ("Function_Keys"),

            # Layout Frame Pack arguments
            ({'side':'top','expand':'yes','fill':'both'}),
            [
                # list of Keys
                ('esc'," ", 'F1', 'F2','F3','F4'," ",'F5','F6','F7','F8'," ",'F9','F10','F11','F12')
            ]
        ],

        [
            ("Character_Keys"),
            ({'side':'top','expand':'yes','fill':'both'}),
            [
                ('`','1','2','3','4','5','6','7','8','9','0','-','=','Backspace'),
                ('Tab','Q','W','E','R','T','Y','U','I','O','P','[',']','\\'),
                ('CapsLock','A','S','D','F','G','H','J','K','L',';',"'","\tEnter"),
                ("Shift\t",'Z','X','C','V','B','N','M',',','.','/',"\tshift"),
                ("Ctrl", "Win",'Alt','\t\t\tSpace\t\t\t','alt','win','menu','ctrl')
            ]
        ]
    ],
    [
        [
            ("System_Keys"),
            ({'side':'top','expand':'yes','fill':'both'}),
            [
                (
                    "Print\nScreen",
                    "Scroll\nLock",
                    "Pause"
                )
            ]
        ],
        [
            ("Editing_Keys"),
            ({'side':'top','expand':'yes','fill':'both'}),
            [
                (
                    "Insert",
                    "Home",
                    "Page\nUp"
                ),
                ( "Delete",
                  "End",
                  "Page\nDown"
                  ),
            ]
        ],

        [
            ("Navigation_Keys"),
            ({'side':'top','expand':'yes','fill':'both'}),
            [
                (
                    "Up",
                ),
                ( "Left",
                  "Down",
                  "Right"
                  ),
            ]
        ],

    ],
    [

        [
            ("Numeric_Keys"),
            ({'side':'top','expand':'yes','fill':'both'}),
            [
                ("Num\nLock","/\n","*\n"),
                ("7\n","8\n","9\n","-\n"),
                ("4\n","5\n","6\n","+\n"),
                ("1\n","2\n","3\n"," "),
                ("0\n",".\n","Enter\n")
            ]
        ],

    ]

]

switcher = {
    '\\': 'backslash',
    '.': 'period',
    '/': 'slash',
    'Shift\t': 'LShift',
    '\tshift': 'RShift',
    'Ctrl': 'LCtrl',
    'ctrl': 'RCtrl',
    'Win': 'LWin',
    'win': 'Rwin',
    'Alt': 'LAlt',
    'alt': 'RAlt',
    '/\n': 'Divide',
    '*\n': 'multiply',
    '-\n': 'subtract',
    '+\n': 'add',
    'Enter\n': 'EnterNumpad',
    '.\n': 'DecimalNumpad',
    '0\n': '0Numpad',
    '1\n': '1Numpad',
    '2\n': '2Numpad',
    '3\n': '3Numpad',
    '4\n': '4Numpad',
    '5\n': '5Numpad',
    '6\n': '6Numpad',
    '7\n': '7Numpad',
    '8\n': '8Numpad',
    '9\n': '9Numpad'
}


class Pose:
    def __init__(self, id: int, pose_name: String, keys: String):
        self.id = id
        self.name = pose_name
        self.key = keys

    def __str__(self):
        return str(self.id) + ' ' + self.name + ' ' + self.key + '\n'


class Controller:
    def __init__(self, root, tab_id):
        # class members:
        self.root = root
        self.webcam_variable = None
        self.poses = []
        self.tree = None
        self.timer = tk.IntVar()
        self.timer.set(5)
        self.images = []
        self.selected_keys = []
        self.dataset_path = "datasets/dataset" + str(tab_id)
        self.log_path = "logs/log" + str(tab_id) + ".txt"
        self.model_path = "saved_models/model" + str(tab_id)

        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)
        self.root.rowconfigure(3, weight=1)

        # Webcam section:
        self.webcam_frame = ttk.LabelFrame(self.root, text="Webcam settings")
        self.webcam_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10, ipady=5)
        self.webcam_settings()

        # Tree section:
        self.load_data()
        self.tree_frame = ttk.LabelFrame(self.root, text="Step 1: Add poses for keys")
        self.tree_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)
        self.create_tree()

        # Options section:
        self.options_frame = ttk.LabelFrame(self.root)
        self.options_frame.grid(row=1, column=1, rowspan=3, sticky='nsew', padx=10, pady=10)
        self.edit_pose(self.poses[0])

        # Train section:
        self.train_frame = ttk.LabelFrame(self.root, text="Step 2: Train model")
        self.train_frame.grid(row=2, column=0, sticky='nsew', padx=10, pady=10, ipady=5)
        self.train()

        # Play section
        self.play_frame = ttk.LabelFrame(self.root, text="Step 3: Play games with poses")
        self.play_frame.grid(row=3, column=0, sticky='nsew', padx=10, pady=10, ipady=5)
        self.play()

    def webcam_settings(self):
        menu_frame = ttk.Frame(self.webcam_frame)
        menu_frame.pack(fill=tk.X, pady=5)
        preview_frame = ttk.Frame(self.webcam_frame)
        preview_frame.pack(fill=tk.X, pady=5)

        camera_ports = find_camera_ports()
        self.webcam_variable = tk.StringVar(self.root)
        if len(camera_ports) > 1:
            webcam_lbl = ttk.Label(menu_frame, text="Choose which webcam you want to use:")
            webcam_lbl.pack(side=tk.LEFT, anchor='w', padx=10)
            camera_ports = ["webcam " + str(port) for port in camera_ports]
            dropdown = ttk.OptionMenu(menu_frame, self.webcam_variable, camera_ports[0], *camera_ports,
                                      bootstyle=(LIGHT, OUTLINE))
            dropdown.pack(side=tk.LEFT)
        else:
            camera_ports = ["webcam 0"]
            webcam_lbl = ttk.Label(menu_frame, text="default webcam is used")
            webcam_lbl.pack(side=tk.LEFT)
        self.webcam_variable.set(camera_ports[0])

        webcam_button = ttk.Button(preview_frame, text="view webcam preview", command=self.camera_preview,
                                   bootstyle=(LIGHT, OUTLINE))
        webcam_button.pack()

    def camera_preview(self):
        webcam_window = ttk.Toplevel(title="webcam preview")
        webcam_window.grab_set()
        width, height = 640, 480
        camera_port = int(self.webcam_variable.get()[-1])
        camera = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        lmain = tk.Label(webcam_window)
        lmain.pack()
        try:
            self.show_frame(camera, lmain)
        except cv2.error as e:
            mb.showerror("Webcam Error", "Webcam is not set properly")
        center(webcam_window)
        webcam_window.protocol("WM_DELETE_WINDOW", lambda: close_window(webcam_window))

    def show_frame(self, camera, lmain):
        _, frame = camera.read()
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, self.show_frame, camera, lmain)

    def load_data(self):
        create_data_folder(self.dataset_path)
        if not os.path.isfile(self.log_path):
            with open(self.log_path, 'w') as f:
                f.writelines(["--not trained--\n", "1 Normal pose Normal\n", "2 Stop pose Stop\n"])
        with open(self.log_path, 'r') as f:
            data = f.readlines()
        for i in range(1, len(data)):
            pose_info = data[i].split()
            self.poses.append(Pose(int(pose_info[0]), ' '.join(pose_info[1:-1]), pose_info[-1]))

    def create_tree(self):
        tree_lbl = ttk.Label(self.tree_frame, text="Click any row in the table to edit its pose")
        tree_lbl.pack(anchor='w', padx=10, pady=10)

        add_row = ttk.Button(self.tree_frame, text="Add a pose", command=self.add_pose)
        add_row.pack(side=tk.BOTTOM, padx=10, pady=10)

        scroll = ttk.Scrollbar(self.tree_frame)
        scroll.pack(side=tk.LEFT, fill=tk.Y)

        cols = ('Pose ID', 'Pose name', 'Pose key')
        self.tree = ttk.Treeview(self.tree_frame, columns=cols, show='headings',
                                 yscrollcommand=scroll.set, bootstyle=PRIMARY)
        self.tree.column("#0", width=0,  stretch=tk.NO)
        self.tree.heading("#0", text="", anchor=tk.CENTER)
        for col in cols:
            self.tree.column(col, anchor=tk.CENTER)
            self.tree.heading(col, text=col, anchor=tk.CENTER)
        for pose in self.poses:
            self.tree.insert(parent='', index='end', iid=pose.id, text='', values=(pose.id, pose.name, pose.key))
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.table_click)  # <ButtonRelease-1>
        scroll.config(command=self.tree.yview)

    def table_click(self, event):
        item = self.tree.focus()
        selected_pose = None
        for pose in self.poses:
            if pose.id == int(self.tree.item(item, 'values')[0]):
                selected_pose = pose
        self.edit_pose(selected_pose)

    def add_pose(self):
        pose_id = self.poses[-1].id + 1
        pose = Pose(pose_id, "Pose " + str(pose_id), "--")
        self.tree.insert(parent='', index='end', iid=pose.id, text='', values=(pose.id, pose.name, pose.key))
        self.poses.append(pose)
        with open(self.log_path, 'r') as f:
            data = f.readlines()
        data[0] = "--not trained--\n"
        data.append(str(pose))
        with open(self.log_path, 'w') as f:
            f.writelines(data)
        self.tree.focus(pose_id)
        self.tree.selection_set(pose_id)
        self.edit_pose(pose)
        #with open('log.txt', 'a') as f:
        #    f.write(str(pose))

    def update_pose(self, pose):
        self.tree.item(pose.id, values=(pose.id, pose.name, pose.key))
        for i in range(len(self.poses)):
            if pose.id == self.poses[i].id:
                self.poses[i] = pose
                break
        with open(self.log_path, 'r') as f:
            data = f.readlines()
        data[0] = "--not trained--\n"
        data[pose.id] = str(pose)
        with open(self.log_path, 'w') as f:
            f.writelines(data)

    def delete_pose(self, pose, pose_dir):
        if check_pose_directory(pose_dir):
            self.remove_pose_data(pose, pose_dir)
        for i in range(pose.id, len(self.poses)):
            self.poses[i].id -= 1
            if self.poses[i].name == "Pose " + str(i+1):
                self.poses[i].name = "Pose " + str(i)
            self.tree.item(i, values=(self.poses[i].id, self.poses[i].name, self.poses[i].key))
            self.poses[i - 1] = self.poses[i]
            if check_pose_directory(self.dataset_path + "/pose" + str(i+1)):
                os.rename(self.dataset_path + "/pose" + str(i+1), self.dataset_path + "/pose" + str(i))
        self.tree.delete(len(self.poses))
        self.poses.pop()
        with open(self.log_path, 'w') as f:
            f.write("--not trained--\n")
            for pose in self.poses:
                f.write(str(pose))
        self.edit_pose(self.poses[0])

    def edit_pose(self, pose):
        if self.options_frame is not None:
            self.options_frame['text'] = pose.name + " options"
            for slave in self.options_frame.pack_slaves():
                slave.pack_forget()
            #self.options_frame.grid_remove()
        #self.options_frame = ttk.LabelFrame(self.root, text=pose.name + " options")
        #self.options_frame.grid(row=1, column=1, rowspan=3, sticky='nsew', padx=10, pady=10)

        button_frame = ttk.Frame(self.options_frame)
        button_frame.pack(fill=tk.X, pady=5)
        if pose.id == 1 or pose.id == 2:
            key_msg = "Normal pose to not press keys" if pose.id == 1 else "Stop pose to stop playing"
            key_lbl = ttk.Label(button_frame, text=key_msg)
            key_lbl.pack(anchor='w', padx=10, pady=5)
        else:
            key_msg = "Choose key" if pose.key == "--" else "Change key"
            key_button = ttk.Button(button_frame, text=key_msg, bootstyle=(LIGHT, OUTLINE))
            key_button['command'] = lambda: self.choose_keys(pose)
            key_button.pack(fill=tk.X, padx=10, pady=5)

        pose_dir = self.dataset_path+"/pose"+str(pose.id)
        data_msg = "Change pose" if check_pose_directory(pose_dir) else "Take a pose"
        pose_button = ttk.Button(button_frame, text=data_msg, bootstyle=(SUCCESS, OUTLINE))
        pose_button['command'] = lambda: self.pose_popup(pose)
        pose_button.pack(fill=tk.X, padx=10, pady=5)
        if check_pose_directory(pose_dir):
            expand_button = ttk.Button(button_frame, text="Expand pose data", bootstyle=(SUCCESS, OUTLINE))
            expand_button['command'] = lambda: self.pose_popup(pose, count=len(os.listdir(pose_dir)))
            expand_button.pack(fill=tk.X, padx=10, pady=5)

            view_button = ttk.Button(button_frame, text="View pose data", bootstyle=(SUCCESS, OUTLINE))
            view_button['command'] = lambda: all_pose_images(pose_dir)
            view_button.pack(fill=tk.X, padx=10, pady=5)

            remove_button = ttk.Button(button_frame, text="Remove pose data", bootstyle=(SUCCESS, OUTLINE))
            remove_button['command'] = lambda: self.remove_pose_data(pose, pose_dir)
            remove_button.pack(fill=tk.X, padx=10, pady=5)

        if pose.id != 1 and pose.id != 2:
            delete_button = ttk.Button(button_frame, text="Delete pose", bootstyle=(DANGER, OUTLINE))
            delete_button['command'] = lambda: self.delete_pose(pose, pose_dir)
            delete_button.pack(fill=tk.X, padx=10, pady=5)

        name_frame = ttk.Frame(self.options_frame)
        name_frame.pack(fill=tk.X, pady=10)
        ttk.Label(name_frame, text="Rename the pose:").pack(anchor='w', padx=10)
        entry = ttk.Entry(name_frame, bootstyle=SECONDARY)
        entry.insert(0, pose.name)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        entry.focus_set()
        rename_button = ttk.Button(name_frame, text="rename", bootstyle=SECONDARY)
        rename_button['command'] = lambda: self.set_name(pose, entry)
        rename_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10)

        timer_frame = ttk.Frame(self.options_frame)
        timer_frame.pack(fill=tk.X, pady=10)
        ttk.Label(timer_frame, text="Set timer to take a pose:").pack(side=tk.LEFT, anchor='w', padx=10)
        vcmd = (timer_frame.register(lambda new_value: new_value.isdigit()), '%P')
        spin = ttk.Spinbox(timer_frame, from_=0, to=100, width=10, textvariable=self.timer,
                           validate="key", validatecommand=vcmd, bootstyle=SECONDARY)
        spin.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10)

        if check_pose_directory(pose_dir):
            image_frame = ttk.Frame(self.options_frame)
            image_frame.pack(fill=tk.BOTH, pady=5)
            ttk.Label(image_frame, text="Pose " + str(pose.id) + " image:").pack(anchor='w', padx=10)
            view_pose_image(pose_dir, image_frame)

    def choose_keys(self, pose):
        self.selected_keys = []
        win = ttk.Toplevel(title="keys options")
        win.grab_set()
        ttk.Label(win, text="Choose up to 2 keyboard or mouse keys", font=("Arial", 10)).pack(padx=10, pady=10)
        self.keyboard(win)
        self.mouse(win)
        exit_frame = ttk.Frame(win)
        exit_frame.pack(padx=10, pady=10)
        confirm_button = ttk.Button(exit_frame, text="Confirm", command=lambda: self.configure_keys(pose, win))
        confirm_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        reset_button = ttk.Button(exit_frame, text="Reset", command=lambda: self.reset_keys(pose, win))
        reset_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cancel_button = ttk.Button(exit_frame, text="Cancel", command=lambda: close_window(win))
        cancel_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        center(win)
        win.protocol("WM_DELETE_WINDOW", lambda: close_window(win))

    def keyboard(self, win):
        keyboard_frame = ttk.LabelFrame(win, text="Keyboard")
        keyboard_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        store_button = {}
        for key_section in KEYS:
            store_section = ttk.Frame(keyboard_frame)
            store_section.pack(side='left', expand='yes', fill='both', padx=10, pady=10, ipadx=10, ipady=10)
            for layer_name, layer_properties, layer_keys in key_section:
                store_layer = ttk.LabelFrame(store_section)
                store_layer.pack(layer_properties)
                for key_bunch in layer_keys:
                    store_key_frame = ttk.Frame(store_layer)
                    store_key_frame.pack(side='top',expand='yes',fill='both')
                    for k in key_bunch:
                        if len(k) <= 3:
                            store_button[k] = (ttk.Button(store_key_frame, text=k))
                        else:
                            store_button[k] = (ttk.Button(store_key_frame, text=k.center(5, ' ')))
                        if " " in k:
                            store_button[k]['state'] = 'disable'
                        store_button[k]['command'] = lambda x=k: self.update_keys(x, store_button[x])
                        store_button[k]['bootstyle'] = (LIGHT, OUTLINE)
                        store_button[k].pack(side='left', fill='both', expand='yes', ipadx=2, ipady=2)

    def mouse(self, win):
        self.images = []
        mouse_frame = ttk.LabelFrame(win, text="Mouse")
        mouse_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_click_image = tk.PhotoImage(file="images/mouse-left-click.png")
        left_button = ttk.Button(mouse_frame, text="left click", image=left_click_image, compound=tk.TOP)
        left_button['command'] = lambda x='mouse-left-click': self.update_keys(x, left_button)
        left_button['bootstyle'] = (LIGHT, OUTLINE)
        left_button.pack(side="left", expand=True, padx=10, pady=10)

        middle_click_image = tk.PhotoImage(file="images/mouse-middle-click.png")
        middle_button = ttk.Button(mouse_frame, text="middle click", image=middle_click_image, compound=tk.TOP)
        middle_button['command'] = lambda x='mouse-middle-click': self.update_keys(x, middle_button)
        middle_button['bootstyle'] = (LIGHT, OUTLINE)
        middle_button.pack(side="left", expand=True, padx=10, pady=10)

        right_click_image = tk.PhotoImage(file="images/mouse-right-click.png")
        right_button = ttk.Button(mouse_frame, text="right click", image=right_click_image, compound=tk.TOP)
        right_button['command'] = lambda x='mouse-right-click': self.update_keys(x, right_button)
        right_button['bootstyle'] = (LIGHT, OUTLINE)
        right_button.pack(side="left", expand=True, padx=10, pady=10)

        scroll_image = tk.PhotoImage(file="images/mouse-scroll.png")
        scroll_up = ttk.Button(mouse_frame, text="scroll up", image=scroll_image, compound=tk.TOP)
        scroll_up['command'] = lambda x='mouse-wheel-up': self.update_keys(x, scroll_up)
        scroll_up['bootstyle'] = (LIGHT, OUTLINE)
        scroll_up.pack(side="left", expand=True, padx=10, pady=10)
        scroll_down = ttk.Button(mouse_frame, text="scroll down", image=scroll_image, compound=tk.TOP)
        scroll_down['command'] = lambda x='mouse-wheel-down': self.update_keys(x, scroll_down)
        scroll_down['bootstyle'] = (LIGHT, OUTLINE)
        scroll_down.pack(side="left", expand=True, padx=10, pady=10)

        movement_image1 = tk.PhotoImage(file="images/mouse-move-up-down.png")
        move_up = ttk.Button(mouse_frame, text="move up", image=movement_image1, compound=tk.TOP)
        move_up['command'] = lambda x='mouse-move-up': self.update_keys(x, move_up)
        move_up['bootstyle'] = (LIGHT, OUTLINE)
        move_up.pack(side="left", expand=True, padx=10, pady=10)
        move_down = ttk.Button(mouse_frame, text="move down", image=movement_image1, compound=tk.TOP)
        move_down['command'] = lambda x='mouse-move-down': self.update_keys(x, move_down)
        move_down['bootstyle'] = (LIGHT, OUTLINE)
        move_down.pack(side="left", expand=True, padx=10, pady=10)

        movement_image2 = tk.PhotoImage(file="images/mouse-move-left-right.png")
        move_left = ttk.Button(mouse_frame, text="move left", image=movement_image2, compound=tk.TOP)
        move_left['command'] = lambda x='mouse-move-left': self.update_keys(x, move_left)
        move_left['bootstyle'] = (LIGHT, OUTLINE)
        move_left.pack(side="left", expand=True, padx=10, pady=10)
        move_right = ttk.Button(mouse_frame, text="move right", image=movement_image2, compound=tk.TOP)
        move_right['command'] = lambda x='mouse-move-right': self.update_keys(x, move_right)
        move_right['bootstyle'] = (LIGHT, OUTLINE)
        move_right.pack(side="left", expand=True, padx=10, pady=10)

        self.images.append([left_click_image, middle_click_image, right_click_image, scroll_image,
                            movement_image1, movement_image2])

    def update_keys(self, key, button):
        if key in switcher:
            key = switcher[key]
        else:
            key = re.sub(r'\s+', '', key)
        if key in self.selected_keys:
            self.selected_keys.remove(key)
            button['bootstyle'] = (LIGHT, OUTLINE)
        else:
            self.selected_keys.append(key)
            button['bootstyle'] = LIGHT

    def configure_keys(self, pose, win):
        if len(self.selected_keys) > 2:
            mb.showerror("Keys Error", "Can only choose up to 2 keys")
        elif len(self.selected_keys) == 0:
            mb.showwarning("Keys Warning", "No keys selected, keeping the current keys")
            close_window(win)
        else:
            keys = ""
            for key in self.selected_keys:
                keys += key + '_'
            keys = keys[:-1]
            pose.key = keys
            self.update_pose(pose)
            self.edit_pose(pose)
            close_window(win)

    def reset_keys(self, pose, win):
        if pose.key != "--":
            pose.key = "--"
        self.update_pose(pose)
        self.edit_pose(pose)
        close_window(win)

    def pose_popup(self, pose, count=0):
        popup_msg = "Collecting data for pose " + str(pose.id)
        if pose.key == '--':
            popup_msg += "\nYou haven't chosen a key for this pose yet!"
        popup_msg += "\nPress OK to take a pose or Cancel to quit."
        popup_msg += "\nOnce you press OK you will have " + str(self.timer.get()) + " seconds to take a pose."
        popup_msg += "\nYou can change the time using the timer button below."
        res = mb.askokcancel("Pose Popup", popup_msg)
        if res:
            camera_port = int(self.webcam_variable.get()[-1])
            try:
                collect_key_data("pose" + str(pose.id), camera_port, data_folder=self.dataset_path,
                                 timer=self.timer.get(), count=count)
            except cv2.error as e:
                mb.showerror("Webcam Error", "Webcam is not set properly")
            self.edit_pose(pose)

    def remove_pose_data(self, pose, pose_dir):
        shutil.rmtree(pose_dir)
        mb.showinfo("Removed Data", "Pose data has been removed from the computer")
        self.edit_pose(pose)

    def set_name(self, pose, entry):
        name = entry.get()
        if name == "":
            name = "Pose " + str(pose.id)
            mb.showwarning("Name warning", "No name entered, returning to default name")
        pose.name = name
        self.update_pose(pose)

    def train(self):
        train_msg = "Press Train to match poses to selected keys" \
                    "\nYou only need to train once when you change or remove poses."
        train_lbl = ttk.Label(self.train_frame, text=train_msg)
        train_lbl.pack(anchor='w', padx=10, pady=5)
        train_button = ttk.Button(self.train_frame, text="Train", command=self.train_popup)
        train_button.pack(pady=5)

    def train_popup(self):
        error_flag, trained_flag, past_train_flag = False, False, False
        error_msg = "Pose data is not ready or was changed manually.\nPlease take poses before trying to train."
        trained_msg = "model is already trained.\nAre you sure you want to train again?"
        past_train_msg = "Model was previously trained.\nDo you want to train new model?"
        msg = "Do you want to train the model?\nThis will take a few minutes..."
        with open(self.log_path, 'r') as f:
            data = f.readline()
        if data.strip('\n') == "--trained--" and check_pose_directory(self.model_path):
            trained_flag = True
        elif data.strip('\n') == "--not trained--" and check_pose_directory(self.model_path):
            past_train_flag = True
        for pose in self.poses:
            pose_dir = self.dataset_path + "/pose" + str(pose.id)
            if not check_pose_directory(pose_dir):
                error_flag = True
                break
        res = False
        if error_flag:
            mb.showerror("Train Error", error_msg)
        elif trained_flag:
            res = mb.askyesno("Train Question", trained_msg)
        elif past_train_flag:
            res = mb.askyesno("Train Question", past_train_msg)
        else:
            res = mb.askyesno("Train Question", msg)
        if res:
            mb.showinfo("Training", "Model is now training")
            controller_model(data_directory=self.dataset_path, model_directory=self.model_path)
            mb.showinfo("Finished", "Model is now ready!")
            with open(self.log_path, 'r') as f:
                data = f.readlines()
            data[0] = "--trained--\n"
            with open(self.log_path, 'w') as f:
                f.writelines(data)

    def play(self):
        play_msg = "Press Play to start playing games with your poses." \
                   "\nPlease have your game ready and have fun!"
        play_lbl = ttk.Label(self.play_frame, text=play_msg)
        play_lbl.pack(anchor='w', padx=10, pady=5)
        play_button = ttk.Button(self.play_frame, text="Play", command=self.play_popup)
        play_button.pack(pady=5)

    def play_popup(self):
        train_flag, keys_flag = False, False
        train_msg = "Model is not trained.\nPlease train the model before playing."
        keys_msg = "Some poses don't have keys yet.\nPlease choose a key for every pose so they can be pressed."
        if not check_pose_directory(self.model_path):
            train_flag = True
        for pose in self.poses:
            if pose.key == "--":
                keys_flag = True
        if train_flag:
            mb.showerror("Play Error", train_msg)
        elif keys_flag:
            mb.showerror("Play Error", keys_msg)
        else:
            mb.showinfo("Playing", "Keys will now be pressed according to your poses")
            camera_port = int(self.webcam_variable.get()[-1])
            try:
                print_poses(self.log_path, self.model_path, "movenet", camera_port)
            except cv2.error as e:
                mb.showerror("Webcam Error", "Webcam is not set properly")

    def update_index(self, tab_id):
        self.dataset_path = "datasets/dataset" + str(tab_id)
        self.log_path = "logs/log" + str(tab_id) + ".txt"
        self.model_path = "saved_models/model" + str(tab_id)


class Tab:
    def __init__(self, frame_id: int, name: String, frame: ttk.Frame):
        self.id = frame_id
        self.name = name
        self.frame = frame
        self.controller = None

    def __str__(self):
        return str(self.id) + ' ' + self.name + '\n'


class HumanControllerApp:
    def __init__(self, root):
        self.root = root
        self.heading()
        if not self.check_webcam():
            return
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.tabs = []
        self.add_lbl, self.add_button = None, None
        self.load_tabs()
        center(self.root)

    def heading(self):
        #ttk.Style("darkly")
        width, height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        if width == 3840 and height == 2160:  # 4k
            self.root.geometry('%dx%d+0+0' % (width*0.35, height*0.51))
        elif width == 1920 and height == 1080:  # 1080p
            self.root.geometry('%dx%d+0+0' % (width*0.59, height*0.86))
        self.root.title("Human Controller GUI")

        heading_frame = ttk.Frame(self.root)
        heading_frame.pack(fill=tk.BOTH, padx=50, pady=10)
        title = ttk.Label(heading_frame, text="Human Controller", font=("Arial", 30))
        title.pack(side=tk.LEFT)
        exit_button = ttk.Button(heading_frame, text="exit", command=self.root.destroy, bootstyle=DANGER)
        exit_button.pack(side=tk.RIGHT)
        separator = ttk.Separator(self.root, orient='horizontal')
        separator.pack(fill=tk.X, padx=20)

    def check_webcam(self):
        camera_ports = find_camera_ports()
        if len(camera_ports) == 0:
            no_webcam_found_msg = "No webcam found. Please connect a webcam to use human controller"
            webcam_lbl = ttk.Label(self.root, text=no_webcam_found_msg)
            webcam_lbl.pack(anchor='w', padx=10, pady=10)
            center(self.root)
            return False
        return True

    def load_tabs(self):
        create_data_folder("datasets")
        create_data_folder("logs")
        create_data_folder("saved_models")

        if not os.path.isfile('logs/general_log.txt'):
            with open('logs/general_log.txt', 'w') as f:
                f.write("1 Controller 1\n")
        with open('logs/general_log.txt', 'r') as f:
            data = f.readlines()
        for line in data:
            frame = ttk.Frame(self.notebook, width=1000, height=750)
            tab_info = line.split()
            self.tabs.append(Tab(int(tab_info[0]), ' '.join(tab_info[1:]), frame))
            self.notebook.add(frame, text=' '.join(tab_info[1:]))

        for tab in self.tabs:
            self.controller_tab(tab)
        self.add_tab()

    def controller_tab(self, tab):
        tab.controller = Controller(tab.frame, tab.id)
        options_frame = ttk.LabelFrame(tab.frame, text="Controller options")
        options_frame.grid(row=0, column=1, sticky='nsew', padx=10, pady=10, ipady=5)

        name_frame = ttk.Frame(options_frame)
        name_frame.pack(fill=tk.X, pady=5)
        delete_frame = ttk.Frame(options_frame)
        delete_frame.pack(fill=tk.X, pady=5)

        entry = ttk.Entry(name_frame, bootstyle=SECONDARY)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        entry.focus_set()
        rename_button = ttk.Button(name_frame, text="rename", bootstyle=SECONDARY)
        rename_button['command'] = lambda: self.set_name(tab, entry)
        rename_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10)

        delete_button = ttk.Button(delete_frame, text="Delete controller", bootstyle=(DANGER, OUTLINE))
        delete_button['command'] = lambda x=tab: self.delete_controller(x)
        delete_button.pack(side=tk.BOTTOM, fill=tk.X, padx=10)

    def add_tab(self):
        add_frame = ttk.Frame(self.notebook, width=1000, height=750)
        add_frame.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(add_frame, text='+')

        self.add_lbl = ttk.Label(add_frame, text="Press the button to add a new controller"
                                                 " - new poses for keys while keeping the old ones")
        self.add_lbl.pack(anchor='w')
        self.add_button = ttk.Button(add_frame, text="Add Controller")
        self.add_button['command'] = lambda x=add_frame: self.add_controller(x)
        self.add_button.pack(padx=10, pady=10)

    def add_controller(self, frame):
        self.add_lbl.pack_forget()
        self.add_button.pack_forget()
        frame_id = self.tabs[-1].id + 1 if self.tabs else 1
        tab = Tab(frame_id, "Controller " + str(frame_id), frame)
        self.notebook.tab(frame_id - 1, text=tab.name)
        self.tabs.append(tab)
        with open('logs/general_log.txt', 'a') as f:
            f.write(str(tab))
        self.controller_tab(tab)
        self.add_tab()

    def set_name(self, tab, entry):
        name = entry.get()
        if name == "":
            name = "Controller " + str(tab.id)
            mb.showwarning("Name warning", "No name entered, returning to default name")
        tab.name = name
        self.update_controller(tab)

    def update_controller(self, tab):
        self.notebook.tab(tab.id - 1, text=tab.name)
        for i in range(len(self.tabs)):
            if tab.id == self.tabs[i].id:
                self.tabs[i] = tab
                break
        with open('logs/general_log.txt', 'r') as f:
            data = f.readlines()
        data[tab.id - 1] = str(tab)
        with open('logs/general_log.txt', 'w') as f:
            f.writelines(data)

    def delete_controller(self, tab):
        self.notebook.forget(tab.frame)
        self.tabs.remove(tab)
        if os.path.isdir("datasets/dataset"+str(tab.id)):
            shutil.rmtree("datasets/dataset"+str(tab.id))
        if os.path.isfile("logs/log"+str(tab.id)+".txt"):
            os.remove("logs/log"+str(tab.id)+".txt")
        if os.path.isdir("saved_models/model"+str(tab.id)):
            shutil.rmtree("saved_models/model"+str(tab.id))

        datasets = os.listdir("datasets")
        logs = os.listdir("logs")
        models = os.listdir("saved_models")
        if tab.id - 1 < len(self.tabs):
            for i in range(tab.id - 1, len(self.tabs)):
                self.tabs[i].id -= 1
                if self.tabs[i].name == "Controller " + str(i+2):
                    self.tabs[i].name = "Controller " + str(i+1)
                self.notebook.tab(i, text=self.tabs[i].name)
                self.tabs[i].controller.update_index(i+1)

                if i < len(datasets) and os.path.isdir("datasets/"+datasets[i]):
                    os.rename("datasets/"+datasets[i], "datasets/dataset" + str(i+1))
                if i + 1 < len(logs) and os.path.isfile("logs/"+logs[i+1]):
                    os.rename("logs/"+logs[i+1], "logs/log" + str(i+1) + ".txt")
                if i < len(models) and os.path.isdir("saved_models/"+models[i]):
                    os.rename("saved_models/"+models[i], "saved_models/model" + str(i+1))

        if self.tabs:
            with open('logs/general_log.txt', 'w') as f:
                for tab in self.tabs:
                    f.write(str(tab))
        else:
            os.remove('logs/general_log.txt')


def center(win):
    """
    centers a tkinter window
    :param win: the main window or Toplevel window to center
    """
    win.update_idletasks()
    width = win.winfo_width()
    frm_width = win.winfo_rootx() - win.winfo_x()
    win_width = width + 2 * frm_width
    height = win.winfo_height()
    titlebar_height = win.winfo_rooty() - win.winfo_y()
    win_height = height + titlebar_height + frm_width
    x = win.winfo_screenwidth() // 2 - win_width // 2
    y = win.winfo_screenheight() // 2 - win_height // 2
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    win.deiconify()


def close_window(win):
    win.grab_release()
    win.destroy()


def find_camera_ports():
    port = 0
    port_list = []
    camera = cv2.VideoCapture(port)
    while camera.isOpened():
        port_list.append(port)
        port = port + 1
        camera = cv2.VideoCapture(port)
    return port_list


def check_pose_directory(pose_dir):
    return os.path.exists(pose_dir) and os.path.isdir(pose_dir) and len(os.listdir(pose_dir)) != 0


def view_pose_image(pose_dir, frame):
    pose_images = os.listdir(pose_dir)
    path = os.path.join(pose_dir, pose_images[len(pose_images)//2])
    image_ = Image.open(path)
    n_image = image_.resize((180, 180))
    photo = ImageTk.PhotoImage(n_image)
    img_label = tk.Label(frame, image=photo)
    img_label.photo = photo
    img_label.pack()


def all_pose_images(pose_dir):
    win = ttk.Toplevel(title="Pose Images")
    win.grab_set()
    win.resizable(width=False, height=False)
    canvas = tk.Canvas(win, width=1000, height=800)
    canvas.pack(side=tk.LEFT)
    scrollbar = tk.Scrollbar(win, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.LEFT, fill='y')
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor='nw')
    row_id, col_id = 0, 0
    for pose_image in os.listdir(pose_dir):
        path = os.path.join(pose_dir, pose_image)
        image_ = Image.open(path)
        n_image = image_.resize((200, 200))
        photo = ImageTk.PhotoImage(n_image)
        img_label = tk.Label(frame, image=photo)
        img_label.photo = photo
        img_label.grid(row=row_id, column=col_id)
        if (col_id+1) % 5 == 0:
            row_id += 1
            col_id = 0
        else:
            col_id += 1
    scrollbar.config()
    center(win)
    win.protocol("WM_DELETE_WINDOW", lambda: close_window(win))


if __name__ == '__main__':
    #root_gui = tk.Tk()
    root_gui = ttk.Window(themename="darkly")
    HumanControllerApp(root_gui)
    root_gui.mainloop()
