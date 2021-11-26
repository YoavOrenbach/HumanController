import tkinter as tk
import PIL
from PIL import Image, ImageTk
from cv2 import cv2
from createDataset import collectKeyData
from trainModel import train_movenet
from predictAndPlay import pose_and_play

keys = [
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
                ('esc'," ", 'F1', 'F2','F3','F4',"",'F5','F6','F7','F8',"",'F9','F10','F11','F12')
            ]
        ],

        [
            ("Character_Keys"),
            ({'side':'top','expand':'yes','fill':'both'}),
            [
                ('~\n`','!\n1','@\n2','#\n3','$\n4','%\n5','^\n6','&\n7','*\n8','(\n9',')\n0','_\n-','+\n=','|\n\\','backspace'),
                ('tab','q','w','e','r','t','y','u','i','o','p','{\n[','}\n]','   '),
                ('capslock','a','s','d','f','g','h','j','k','l',':\n;',"\"\n'","enter"),
                ("shift",'z','x','c','v','b','n','m','<\n,','>\n.','?\n/',"shift"),
                ("ctrl", "[+]",'alt','\t\tspace\t\t','alt','[+]','[=]','ctrl')
            ]
        ]
    ],
    [
        [
            ("System_Keys"),
            ({'side':'top','expand':'yes','fill':'both'}),
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
            ({'side':'top','expand':'yes','fill':'both'}),
            [
                (
                    "insert",
                    "home",
                    "page\nup"
                ),
                ( "delete",
                  "end",
                  "page\ndown"
                  ),
            ]
        ],

        [
            ("Navigation_Keys"),
            ({'side':'top','expand':'yes','fill':'both'}),
            [
                (
                    "up",
                ),
                ( "right",
                  "down",
                  "left"
                  ),
            ]
        ],

    ],
    [

        [
            ("Numeric_Keys"),
            ({'side':'top','expand':'yes','fill':'both'}),
            [
                ("num\nlock","/","*","-"),
                ("7","8","9","+"),
                ("4","5","6"," "),
                ("0","1","2","3"),
                ("0",".","enter")
            ]
        ],

    ]

]


# Frame Class
class Keyboard(tk.Frame):
    def __init__(self, parent, camera_variable): #, *args, **kwargs
        tk.Frame.__init__(self, parent)
        self.camera_port = camera_variable
        # Function For Creating Buttons
        self.create_frames_and_buttons()

    # Function For Extracting Data From KeyBoard Table
    # and then provide us a well looking
    # keyboard gui
    def create_frames_and_buttons(self):
        # take section one by one
        for key_section in keys:
            # create Sperate Frame For Every Section
            store_section = tk.Frame(self)
            store_section.pack(side='left',expand='yes',fill='both',padx=10,pady=10,ipadx=10,ipady=10)

            for layer_name, layer_properties, layer_keys in key_section:
                store_layer = tk.LabelFrame(store_section)#, text=layer_name)
                #store_layer.pack(side='top',expand='yes',fill='both')
                store_layer.pack(layer_properties)
                for key_bunch in layer_keys:
                    store_key_frame = tk.Frame(store_layer)
                    store_key_frame.pack(side='top',expand='yes',fill='both')
                    for k in key_bunch:
                        k=k.capitalize()
                        if len(k)<=3:
                            store_button = tk.Button(store_key_frame, text=k, width=2, height=2)
                        else:
                            store_button = tk.Button(store_key_frame, text=k.center(5,' '), height=2)
                        if " " in k:
                            store_button['state']='disable'
                        #flat, groove, raised, ridge, solid, or sunken
                        store_button['relief']="sunken"
                        #store_button['bg']="powderblue"
                        store_button['command'] = lambda x=k: collectKeyData(x, int(self.camera_port.get()[-1]))
                        store_button.pack(side='left',fill='both',expand='yes')


def mouse(parent, camera_variable):
    mouse_frame = tk.Frame(parent)
    mouse_frame.pack()
    left_click_image = tk.PhotoImage(file="images/left-click.png")
    #left_click_image.subsample(3,3)
    left_button = tk.Button(mouse_frame, text="left click", image=left_click_image, compound=tk.RIGHT)
    left_button['command'] = lambda x = 'left_click': collectKeyData(x, int(camera_variable.get()[-1]))
    left_button.pack(side="left", anchor='e', expand=True)

    right_click_image = tk.PhotoImage(file="images/right-click.png")
    #right_click_image.subsample(3,3)
    right_button = tk.Button(mouse_frame, text="right click", image=right_click_image, compound=tk.LEFT)
    right_button['command'] = lambda x = 'right_click': collectKeyData(x, int(camera_variable.get()[-1]))
    right_button.pack(side="right", anchor='w', expand=True)
    return left_click_image, right_click_image


def find_camera_ports():
    port = 0
    port_list = []
    camera = cv2.VideoCapture(port)
    while camera.isOpened():
        port_list.append(port)
        port = port + 1
        camera = cv2.VideoCapture(port)
    return port_list


def choose_webcam(camera_ports):
    if len(camera_ports) > 1:
        tk.Label(root, text="choose which webcam you want to use").pack(anchor="w")
        variable = tk.StringVar(root)
        camera_ports = ["webcam "+str(port) for port in camera_ports]
        variable.set(camera_ports[0])
        dropdown = tk.OptionMenu(root, variable, *camera_ports)
        dropdown.pack()
        return variable
    return "webcam 0"


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


if __name__ == '__main__':
    root = tk.Tk(className="Human Controller GUI")
    tk.Label(root, text="Human Controller App").pack()

    camera_ports = find_camera_ports()
    if len(camera_ports) == 0:
        tk.Label(root, text="No webcam found. please connect a webcam to use human controller").pack()
    else:
        webcam_port = choose_webcam(camera_ports)
        camera_preview(root, webcam_port)
        tk.Label(root, text="Press the key you want to take a pose for:").pack(anchor='w')
        Keyboard(root, webcam_port).pack()
        left_click, right_click = mouse(root, webcam_port)
        train_and_play(root, webcam_port)
    root.mainloop()
