from doctest import master
import json

import tkinter as tk
from tkinter import Button, IntVar, ttk, messagebox
from tkinter import font

#for multilanguage support, https://pypi.org/project/translate/

#add confirmation message popup to "delete user" section

def clearFrame(frame):
    try:
        for i in frame.winfo_children():
            i.destroy()
    except:
        print('**CLEARFRAME ERROR')

def getParamVals(parameters):
    for i in parameters:
        p = parameters[i]
        if p != None:
            p["Values"] = p["Spinbox"].get()

def mainPage():
    clearFrame(masterFrame)

    with open("parameters.json", "r") as f:
        print('opened file')
        parameters = json.load(f)
    #print(parameters)
    row = 1
    for i in parameters:
        print(i)
        p = parameters[i]
        if p != None:
            print(p)
            p["Spinbox"] = ttk.Spinbox(masterFrame,
                from_=p["Range"][0],
                to=p["Range"][1],
                increment=p["Inc"]
                )
            p["Spinbox"].insert(0, p["Default"])
            p["Spinbox"].grid(row=row, column=1, padx=5, pady=5)
            ttk.Label(masterFrame, text=i).grid(row=row, column=0, padx=5, pady=5)
            row+=1
    applyButton = ttk.Button(masterFrame, 
        text="Apply",
        command=lambda: [getParamVals(parameters)] #add close window etc
        )
    applyButton.grid(row=row, column=1, padx=5, pady=5)


    
    

def mainPage2():
    clearFrame(masterFrame)

    LRLVal = 0 #loop this
    LRL = ttk.Spinbox(masterFrame, 
        from_ = 30, 
        to=175, 
        increment=5
        )
    LRL.insert(0, 120)
    LRL.grid(row=1, column=1, padx=5, pady=6)

    URLVal = 0
    URL = ttk.Spinbox(masterFrame, 
        from_=50,
        to=175,
        increment=5
        )
    URL.insert(0, 120)
    URL.grid(row=2, column=1, padx=5, pady=6)

    APAVal = 0
    APA = ttk.Spinbox(masterFrame,
        from_=0.5,
        to=7.0,
        increment=0.5
        )
    APA.insert(0, 3.5)
    APA.grid(row=3, column=1, padx=5, pady=6)

    APWVal = 0
    APW = ttk.Spinbox(masterFrame, 
        from_=0.1,
        to=1.9,
        increment=0.1
        )
    APW.insert(0, 0.4)
    APW.grid(row=4, column=1, padx=5, pady=6)

    VPAVal = 0
    VPA = ttk.Spinbox(masterFrame,
        from_=0.5,
        to=7.0,
        increment=0.5
        )
    VPA.insert(0, 3.5)
    VPA.grid(row=5, column=1, padx=5, pady=6)

    VPWVal = 0
    VPW = ttk.Spinbox(masterFrame, 
        from_=0.1,
        to=1.9,
        increment=0.1
        )
    VPW.insert(0, 0.4)
    VPW.grid(row=6, column=1, padx=5, pady=6)

    VRPVal = 0
    VRP = ttk.Spinbox(masterFrame,
        from_=150,
        to=500,
        increment=10
        )
    VRP.insert(0, 320)
    VRP.grid(row=7, column=1, padx=5, pady=6)

    ARPVal = 0
    ARP = ttk.Spinbox(masterFrame,
        from_=150,
        to=500,
        increment=10
        )
    ARP.insert(0, 250)
    ARP.grid(row=8, column=1, padx=5, pady=6)

    applyButton = ttk.Button(masterFrame, 
        text="Apply",
        command=lambda:[
            LRLVal := LRL.get(), #loop this
            URLVal := URL.get(),
            APAVal := APA.get(),
            APWVal := APW.get(),
            VPAVal := VPA.get(),
            VPWVal := VPW.get(),
            VRPVal := VRP.get(),
            ARPVal := ARP.get()
        ]
        )
    applyButton.grid(row=9, column=1, padx=5, pady=6)

def checkEmptyCredentials(username, password):
    if username == '' or password == '':
        return True


def deleteAccount(username):
    #enter password for 'username' first
    print('entered delete account') ##
    with open("userpass.json", "r") as f:
        print('opened file')
        data = json.load(f)
    for index, object in enumerate(data):
        if object['username'] == username:
            print('found') ##
            data.pop(index)

    with open("userpass.json", "w") as f:
        f.write(json.dumps(data, indent=2))
    
    if not username == '':
        messagebox.showinfo("Delete user", "User \"{}\" has been deleted.".format(username))
    else:
        messagebox.showwarning("Delete user", "No account was selected.")


def newAccount(username, password):
    print('entered create account') ##
    with open("userpass.json", "r") as f:
        print('opened file')
        data = json.load(f)
    data.append({'username':username, 'password':password})

    with open("userpass.json", "w") as f:
        f.write(json.dumps(data, indent=2))  
    print('new user created:', username)  

def maxUsersReached():
    popup = tk.Tk()
    popup.title("Max Users Reached!")
    frame = ttk.Frame(popup)
    frame.pack(padx=5, pady=5)
    popup.minsize(250, 130)

    header = ttk.Label(
        frame, 
        text="Maximum Users Reached!", 
        font="Calibri, 15"
        )
    header.grid(row = 0, column=0)
    subheader = ttk.Label(
        frame, 
        text="Select a user to delete",
        font="Calibri, 12"
    )
    subheader.grid(row=1, column=0)

    selectedUser = tk.StringVar(frame)
    with open("userpass.json", "r") as f:
        data = json.load(f)

    optionFrame = ttk.Frame(frame)
    optionFrame.grid(row=2, column=0)

    optionStyle = ttk.Style(optionFrame)
    optionStyle.configure("option.TRadiobutton", font="Calibri 12") ####
    row=0
    for loginInfo in data:
        user = loginInfo['username']
        print(user) ##
        ttk.Radiobutton(
            optionFrame, 
            style="option.TRadiobutton", ####
            text=user, 
            var=selectedUser, 
            value = user
            ).grid(row=row, column=0, sticky='w')
        row+=1

    buttonFrame = ttk.Frame(frame)
    
    deleteButton = ttk.Button(
        buttonFrame, 
        text="Delete", 
        command=lambda: [
            print("Now deleting:"+selectedUser.get()+"**"), ##
            deleteAccount(selectedUser.get()),
            popup.destroy()
            ]
        )
    cancelButton = ttk.Button(
        buttonFrame,
        text="Cancel",
        command=lambda: popup.destroy()
    )
    buttonFrame.grid(column=0, row=3, pady=(5, 0))
    deleteButton.grid(column=0, row=0)
    cancelButton.grid(column=1, row=0)


def login(userEnter, passEnter):
    if checkEmptyCredentials(userEnter, passEnter):
        messagebox.showwarning("Login", "Enter a Username and Password.")
        return 
    print('Success') ##
    userPassFound = False
    with open("userpass.json", "r") as f:
        data = json.load(f)
    for i in data:
        if i['username'] == userEnter and i['password'] == passEnter:
            userPassFound = True
            messagebox.showinfo("Login", "Login Successful!")
            mainPage()
    if not userPassFound: 
        messagebox.askretrycancel("Login", "Login Unsuccessful.\nUsername or password not found.\nPlease Try Again.")

def createNewUser(userEnter, passEnter):
    if checkEmptyCredentials(userEnter, passEnter):
        messagebox.showwarning("Login", "Enter a Username and Password.")
        return 
    UserAlreadyExists = False
    with open("userpass.json", "r") as f:
        data = json.load(f)
    for i in data:
        if i['username'] == userEnter:
            UserAlreadyExists = True
            messagebox.askretrycancel("New User", "Username already exists!")
            break
    if not UserAlreadyExists:
        if len(data) >= maxUsers:
            maxUsersReached()
        else:
            newAccount(userEnter, passEnter)
            messagebox.showinfo("New User", "New User Added!")

    

def loginPage():
    clearFrame(masterFrame)
    frame = ttk.Frame(masterFrame)
    frame.pack(padx=0, pady=0)

    titleLabel = ttk.Label(frame, text="Login", font="Calibri 20")

    userLabel = ttk.Label(frame, text="Username: ", font="Calibri 16")
    passLabel = ttk.Label(frame, text="Password: ", font = "Calibri 16")

    #boxStyle = ttk.Style()
    #boxStyle.configure("userbox.TEntry", width=500)
    userBox = ttk.Entry(frame, font="Calibri 12")
    passBox = ttk.Entry(frame, show="*", font="Calibri 12")

    buttonFrame = ttk.Frame(frame)

    style.configure("enter.TButton", font=('Calibri', 12))
    enter = ttk.Button(
        buttonFrame, 
        text="LOGIN", 
        style="enter.TButton", 
        command= lambda: login(userBox.get(), passBox.get())
    )
    create = ttk.Button(
        buttonFrame,
        text="CREATE NEW",
        style="enter.TButton",
        command= lambda: createNewUser(userBox.get(), passBox.get())
    )
    
    titleLabel.grid(row=0, column=0)#pack(pady=8)
    userLabel.grid(row=1, column=0, sticky='w')#pack()
    userBox.grid(row=2, column=0, sticky='w', ipadx=50)#pack(pady=3)
    passLabel.grid(row=3, column=0, sticky='w')#pack()
    passBox.grid(row=4, column=0, sticky='w', ipadx=50)#pack(pady=3)
    
    buttonFrame.grid(row=5, column=0)
    enter.grid(row=0, column=1, pady=(8, 0), padx=(3,0))#pack()
    create.grid(row=0, column=0, pady=(8, 0), padx=(0,3))

    root.minsize(300, 210)

def welcome():
    frame = ttk.Frame(masterFrame)
    frame.pack(padx=0, pady=0)

    titleLabel = ttk.Label(frame, text="Welcome", font="Calibri 25")
    style.configure("enter.TButton", font=('Calibri', 12))
    welcomeButton = ttk.Button(frame, style="welcome.TButton", text="Ok", command=lambda: loginPage())
    titleLabel.pack(pady=(0, 5))
    welcomeButton.pack()

    root.minsize(230, 70)

def main():
    #instantiate global variables
    global root
    global style
    global masterFrame
    global maxUsers
    maxUsers = 2
    #create tkinter window instance
    root = tk.Tk()
    root.title("3K04 app")
    style = ttk.Style()

    #create frame to size the window
    masterFrame = ttk.Frame(root)
    masterFrame.pack(padx=10, pady=10)
    welcome()
    #mainPage()

    root.mainloop()

main()