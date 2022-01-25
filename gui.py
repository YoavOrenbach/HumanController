import tkinter as tk
from tkinter import ttk
from tokenize import String

import PIL
from PIL import Image, ImageTk
from cv2 import cv2

from create_dataset import collectData, create_data_folder
import os
import re
from utils import get_number_from_user, get_string_from_user
from create_dataset import collectKeyData


class Pose:
    def __init__(self, pose_name: String, data_directory: String, buttons: list):
        self.pose_name = pose_name
        self.data_directory = data_directory
        self.buttons = buttons

    def __str__(self):
        print(self.pose_name, ":", self.data_directory)


keys = [
    [
        # =========================================
        # ===== Keyboard Configurations ===========
        # =========================================

        [
            # Layout Name
            ("Function_Keys"),

            # Layout Frame Pack arguments
            ({'side': 'top', 'expand': 'yes', 'fill': 'both'}),
            [
                # list of Keys
                ('esc', " ", 'F1', 'F2', 'F3', 'F4', "", 'F5', 'F6', 'F7', 'F8', "", 'F9', 'F10', 'F11', 'F12')
            ]
        ],

        [
            ("Character_Keys"),
            ({'side': 'top', 'expand': 'yes', 'fill': 'both'}),
            [
                ('`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 'backspace'),
                ('tab', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', 'back slash'),
                ('capslock', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', "'", "enter"),
                ("shift", 'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', 'slash', "shift"),
                ("ctrl", "[+]", 'alt', '\t\tspace\t\t', 'alt', '[+]', '[=]', 'ctrl')
            ]
        ]
    ],
    [
        [
            ("System_Keys"),
            ({'side': 'top', 'expand': 'yes', 'fill': 'both'}),
            [
                (
                    "print\nscreen\nsys",
                    "scroll\nlock",
                    "pause\nbreak"
                )
            ]
        ],
        [
            ("Editing_Keys"),
            ({'side': 'top', 'expand': 'yes', 'fill': 'both'}),
            [
                (
                    "insert",
                    "home",
                    "page\nup"
                ),
                ("delete",
                 "end",
                 "page\ndown"
                 ),
            ]
        ],

        [
            ("Navigation_Keys"),
            ({'side': 'top', 'expand': 'yes', 'fill': 'both'}),
            [
                (
                    "up",
                ),
                ("right",
                 "down",
                 "left"
                 ),
            ]
        ],

    ],
    [

        [
            ("Numeric_Keys"),
            ({'side': 'top', 'expand': 'yes', 'fill': 'both'}),
            [
                ("num\nlock", "/", "*", "-"),
                ("7", "8", "9", "+"),
                ("4", "5", "6", " "),
                ("0", "1", "2", "3"),
                ("0", ".", "enter")
            ]
        ],

    ]

]


def change_pose_key(pose_name: String, new_key):
    pose = get_pose(pose_name)
    source = pose.data_directory
    lst = source.split(" ")
    lst = lst[:-1] + [new_key]
    dest = ' '.join(lst)
    os.rename(source, dest)
    pose.data_directory = dest


# Frame Class
class Keyboard(tk.Frame):
    def __init__(self, parent, camera_variable, pose_name: String):  # , *args, **kwargs
        tk.Frame.__init__(self, parent)
        self.camera_port = camera_variable
        # Function For Creating Buttons
        self.create_frames_and_buttons(pose_name)

    # Function For Extracting Data From KeyBoard Table
    # and then provide us a well looking
    # keyboard gui
    def create_frames_and_buttons(self, pose_name):
        # take section one by one
        for key_section in keys:
            # create Sperate Frame For Every Section
            store_section = tk.Frame(self)
            store_section.pack(side='left', expand='yes', fill='both', padx=10, pady=10, ipadx=10, ipady=10)

            for layer_name, layer_properties, layer_keys in key_section:
                store_layer = tk.LabelFrame(store_section)
                store_layer.pack(layer_properties)
                for key_bunch in layer_keys:
                    store_key_frame = tk.Frame(store_layer)
                    store_key_frame.pack(side='top', expand='yes', fill='both')
                    for k in key_bunch:
                        k = k.capitalize()
                        if len(k) <= 3:
                            store_button = tk.Button(store_key_frame, text=k, width=2, height=2)
                        else:
                            store_button = tk.Button(store_key_frame, text=k.center(5, ' '), height=2)
                        if " " in k:
                            store_button['state'] = 'disable'
                        # flat, groove, raised, ridge, solid, or sunken
                        store_button['relief'] = "sunken"
                        store_button['command'] = lambda x=k: change_pose_key(pose_name, re.sub('\s+', '', x))
                        store_button.pack(side='left', fill='both', expand='yes')


def mouse(parent, pose_name: String):
    mouse_frame = tk.Frame(parent)
    mouse_frame.pack()

    left_click_image = tk.PhotoImage(file="images/left-click.png")
    left_button = tk.Button(mouse_frame, text="left click", image=left_click_image, compound=tk.TOP)
    left_button['command'] = lambda x='left_click': change_pose_key(pose_name, x)
    left_button.pack(side="left", expand=True)

    middle_click_image = tk.PhotoImage(file="images/mouse.png")
    middle_button = tk.Button(mouse_frame, text="middle click", image=middle_click_image, compound=tk.TOP)
    middle_button['command'] = lambda x='middle_click': change_pose_key(pose_name, x)
    middle_button.pack(side="left", expand=True)

    right_click_image = tk.PhotoImage(file="images/right-click.png")
    right_button = tk.Button(mouse_frame, text="right click", image=right_click_image, compound=tk.TOP)
    right_button['command'] = lambda x='right_click': change_pose_key(pose_name, x)
    right_button.pack(side="right", expand=True)

    return left_click_image, middle_click_image, right_click_image


def get_pose(pose_name):
    global pose_list
    for pose in pose_list:
        if pose.pose_name == pose_name:
            return pose
    print(pose_name)
    print(pose_list)
    raise Exception('there is no pose with this name')


def get_data_directory(pose_name):
    return get_pose(pose_name).data_directory


def select_button(button, pose_name: String):
    path = get_data_directory(pose_name)
    win = tk.Toplevel()
    win.wm_title("buttons")
    tk.Label(win, text="Press the key you want to take a pose for:").pack(anchor='w')
    unselect_button = tk.Button(win, text="unselect button", command=lambda x='select_button': change_pose_key(pose_name, x))
    unselect_button.pack()
    Keyboard(win, webcam_port, pose_name).pack()
    left_click, middle_click, right_click = mouse(win, pose_name)
    win.mainloop()
    button['text'] = get_data_directory(pose_name).split(" ")[-1]


def change_pose_name(win, button, pose_name: String):
    pose = get_pose(pose_name)
    source = pose.data_directory
    lst = source.split(" ")
    pose_name = get_string_from_user(win, "How do you wanna name this pose?")
    button['text'] = pose_name
    lst = lst[:2] + [pose_name] + [lst[-1]]
    dest = ' '.join(lst)
    os.rename(source, dest)
    pose.data_directory = dest


def delete_folder(pose_name: String):
    source = get_data_directory(pose_name)
    os.remove(source)


def pose_data_options(button, pose_name: String, data_folder="dataset", camera_port=0):
    win = tk.Toplevel()
    win.wm_title("pose data options")

    l = tk.Label(win, text="what do you wanna do?")
    l.grid(row=0, column=0)

    db = ttk.Button(win, text="rename pose data", command=lambda x=pose_name: change_pose_name(win, button, x))
    db.grid(row=1, column=0)

    rb = ttk.Button(win, text="rewrite pose data", command=lambda x=pose_name: collectData(win, x, True, camera_port, data_folder))
    rb.grid(row=2, column=0)

    eb = ttk.Button(win, text="expend pose data", command=lambda x=pose_name: collectData(win, x, False, camera_port, data_folder))
    eb.grid(row=3, column=0)

    vb = ttk.Button(win, text="view pose data", command=win.destroy)
    vb.grid(row=4, column=0)

    eb = ttk.Button(win, text="exit pose data", command=win.destroy)
    eb.grid(row=5, column=0)

    if pose_name != 'normal' and pose_name != 'ending pose':
        db = ttk.Button(win, text="delete pose data", command=lambda x=pose_name: delete_folder(x))
        db.grid(row=6, column=0)  # todo: dill with PermissionError: [WinError 5] Access is denied


def add_new_pose_buttons(data_folder="dataset"):
    global pose_list, pose_button_list
    pose_name = f"pose {len(pose_button_list)}"
    create_data_folder(data_folder + "/" + pose_name + "  choose_button")
    pose_list.append(Pose(pose_name, data_folder + "/" + pose_name + "  choose_button", ['choose_button']))
    pose_button_list.append(
        [tk.Button(root, text=pose_name, command=lambda x=pose_name: pose_data_options(pose_button_list[-1][0], x)),
         tk.Button(root, text="choose button", command=lambda x=pose_name: select_button(pose_button_list[-1][1], x))])
    pose_button_list[-1][0].grid(row=len(pose_button_list) + 2, column=0, sticky='nsew')
    pose_button_list[-1][1].grid(row=len(pose_button_list) + 2, column=1, sticky='nsew')


def find_camera_ports():
    port = 0
    port_list = []
    camera = cv2.VideoCapture(port)
    while camera.isOpened():
        port_list.append(port)
        port = port + 1
        camera = cv2.VideoCapture(port)
    return port_list


def choose_webcam(camera_ports: list):
    variable = tk.StringVar(root)
    if len(camera_ports) > 1:
        tk.Label(root, text="choose which webcam you want to use:").grid(row=1, column=0, sticky='nsew')
        camera_ports = ["webcam " + str(port) for port in camera_ports]
        dropdown = tk.OptionMenu(root, variable, *camera_ports)
        dropdown.grid(row=1, column=1, sticky='nsew')
    else:
        camera_ports = ["webcam 0"]
        tk.Label(root, text="default webcam is used").grid(row=1, column=0, sticky='nsew', columnspan=2)
    variable.set(camera_ports[0])
    return variable


def camera_preview(parent_frame, camera_variable):
    camera_button = tk.Button(parent_frame, text="click to see camera preview:")
    camera_button.pack()
    up_frame = tk.Frame(parent_frame)
    up_frame.pack(side=tk.TOP)
    lmain = tk.Label(up_frame)
    camera = camera_button.bind('<ButtonPress-1>', lambda x: start_camera(camera_variable, lmain))
    camera_button.bind('<ButtonRelease-1>', lambda x=camera: stop_camera(camera, lmain, up_frame))


def start_camera(camera_variable, lmain):
    width, height = 480, 360
    camera_port = int(camera_variable.get()[-1])
    camera = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    lmain.pack()
    show_frame(camera, lmain)


def show_frame(camera, lmain, cancel=False):
    global feed
    if not cancel:
        _, frame = camera.read()
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = PIL.Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        feed = lmain.after(10, show_frame, camera, lmain)
    else:
        lmain.after_cancel(feed)


def stop_camera(camera, lmain, up_frame):
    lmain.pack_forget()
    tk.Label(up_frame).pack()
    show_frame(camera, lmain, True)


def train_and_play(parent, camera_variable):
    down_frame = tk.Frame(parent)
    down_frame.pack(side=tk.BOTTOM)
    tk.Label(down_frame, text="press train so we can predict new poses:").pack(padx=10, pady=10)
    train_button = tk.Button(down_frame, text="train", command=train_movenet)
    train_button.pack()
    tk.Label(down_frame, text="press play to become the controller:").pack(padx=10, pady=10)
    play_button = tk.Button(down_frame, text="play")
    play_button['command'] = lambda: pose_and_play(int(camera_variable.get()[-1]))
    play_button.pack()


def load_data(data_folder="dataset"):
    global pose_list, pose_button_list
    for folder_name in os.listdir(data_folder):
        folder_path = data_folder + "/" + folder_name
        if not os.path.isdir(folder_path):
            continue
        lst = folder_name.split(' ')
        pose_name = ' '.join(lst[2:-1]) if len(lst) > 3 else ' '.join(lst[:2])
        pose_list.append(Pose(pose_name, folder_path, lst[-1].split('__')))
    for pose in pose_list:
        pose_button_list.append([tk.Button(root, text=pose.pose_name, command=lambda x=pose.pose_name: pose_data_options(pose_button_list[-1][0], x)),
                                 tk.Button(root, text="__".join(pose.buttons), command=lambda x=pose.pose_name: select_button(pose_button_list[-1][1], x))])
        pose_button_list[-1][0].grid(row=len(pose_button_list) + 2, column=0, sticky='nsew')
        pose_button_list[-1][1].grid(row=len(pose_button_list) + 2, column=1, sticky='nsew')


if __name__ == '__main__':
    root = tk.Tk(className="Human Controller GUI")
    frame = tk.Frame(root)
    frame.grid(row=1, column=0, sticky='nsew', columnspan=2)
    tk.Label(root, text="Human Controller App").grid(row=0, column=0, sticky='nsew', columnspan=2)

    camera_ports = find_camera_ports()
    if len(camera_ports) == 0:
        no_webcam_found_msg = "No webcam found. please connect a webcam to use human controller"
        tk.Label(root, text=no_webcam_found_msg).grid(row=1, column=0, sticky='nsew', columnspan=2)
        tk.Button(root, text="Quit!", command=root.quit).grid(row=2, column=0, sticky='nsew', columnspan=2)
    else:
        webcam_port = choose_webcam(camera_ports)
        # camera_preview(root, webcam_port)
        pose_list = []
        pose_button_list = []
        add_new_pose = tk.Button(root, text="add new pose", command=add_new_pose_buttons)
        add_new_pose.grid(row=2, column=0, sticky='nsew', columnspan=2)
        load_data()
        train_and_play(root, webcam_port)
    root.mainloop()
