


##### Binding  #####

from tkinter import *

root = Tk()
contents = StringVar()
entryBox = Entry(root, textvariable=contents)
entryBox.pack()
myLabel = Label(root, text='Blank', textvariable=contents)
myLabel.pack()
root.mainloop()


#######################

from tkinter import *

def callback(sv):
    print (sv.get())

root = Tk()
sv = StringVar()
sv.trace("w", lambda name, index, mode, sv=sv: callback(sv))
e = Entry(root, textvariable=sv)
e.pack()
root.mainloop()










#############################


import tkinter as tk
# you must create a root window before you can create Tk variables
root = tk.Tk()

# create a Tk string variable
myVar = tk.StringVar()

# define a callback function that describes what it sees
# and also modifies the value
def callbackFunc(name, index, mode):
  print ("callback called with name=%r, index=%r, mode=%r" % (name, index, mode))
  varValue = root.globalgetvar(name)
  print ("    and variable value = {}".format(varValue))
  # modify the value, just to show it can be done
  root.globalsetvar(name, varValue + " modified by {} callback".format(mode))


# set up a trace for writing and for reading;
# save the returned names for future deletion of the trace
wCallbackName = myVar.trace_variable('w', callbackFunc)
rCallbackname = myVar.trace_variable('r', callbackFunc)

# set a value, triggering the write callback
myVar.set("first value")

# get the value, triggering a read callback and then print the value;
# do not perform the get in the print statement
# because the output from the print statement and from the callback
# will be blended together in a confusing fashion
varValue = myVar.get() # trigger read callback
print ("after first set, myVar =", varValue)

# set and get again to show that the trace callbacks persist
myVar.set("second value")
varValue = myVar.get() # trigger read callback
print ("after second set, myVar =", varValue)

# delete the write callback and do another set and get
myVar.trace_vdelete('w', wCallbackName)
myVar.set("third value")
varValue = myVar.get() # trigger read callback
print ("after third set, myVar =", varValue)
root.mainloop()


# Output is:
# callback called with name='PY_VAR0', index='', mode='w'
#     and variable value = 'first value'
# callback called with name='PY_VAR0', index='', mode='r'
#     and variable value = "first value modified by 'w' callback"
# after first set, myVar = first value modified by 'w' callback modified by 'r' callback
# callback called with name='PY_VAR0', index='', mode='w'
#     and variable value = 'second value'
# callback called with name='PY_VAR0', index='', mode='r'
#     and variable value = "second value modified by 'w' callback"
# after second set, myVar = second value modified by 'w' callback modified by 'r' callback
# callback called with name='PY_VAR0', index='', mode='r'
#     and variable value = 'third value'
# after third set, myVar = third value modified by 'r' callback







#####################################


# This was working but then the trace stopped updating unless you submit everything at once to interactive
# (ie the definition and call of ChartUpdater)
# - even this doesn't work now
# But does illustrate other ways of doing Update (with button and bind)

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

import time


class ChartUpdater_Old(tk.Frame):
    def __init__(self, master=None, plot_Fn=None):

        if (tk._default_root is None):
            raise IOError("tkinter not initialized")

        if master is None: master = tk.Tk()

        # Keep handle to destroy
        self.master = master
        self.plotFn = plot_Fn

        # Keep handle on figure for updating
        self.fig=plt.figure(figsize=(8,8))
        self.canvas=FigureCanvasTkAgg(self.fig,master=master)
        self.canvas.get_tk_widget().grid(row=1,column=1)
        self.canvas.draw()

        tk.Label(master, text="Enter initial letters: ").grid(row=0, column=0)



        # This works best
        #sv = tk.StringVar()
        #sv.trace_variable("w", callbackFunc)
        #sv.trace("w", lambda name, index, mode, sv=sv : self.update_callback(sv.get()))
        #sv.trace_add("write", lambda name, index, mode, sv=sv : callback(sv))

     #   sv = tk.StringVar()
      #  sv.trace("w", lambda name, index, mode, sv=sv: self.callback(sv))

        self.entry = tk.Entry(master)
        self.entry.grid(row=0, column=1)

     #   sv.set("5")
        #self.entry.insert(0, "...")



        # Works but lags behind:
        # - Problem with bind on <Key> is that it is called before the latest keypress is available
        # self.entry.bind('<Key>', self.update_event)

        # Works OK as manual update
        # - command takes no parameters and calls the update method
        # - passes a lambda which will call the given plotFn, passing the axis and text
        update_command = lambda: self.update(lambda: self.plotFn(self.fig.gca(), self.entry.get()))
        tk.Button(master=master, text="Update", command=update_command).grid(row=0, column=2)


       # tk.Button(master=master, text="Auto", command=self.auto_command).grid(row=2, column=1)

        tk.Button(master=master, text="Close", command=self.close_command).grid(row=2, column=2)

        master.mainloop()

    def callback(sv):
        print("Callback " + sv.get())


    def auto_command(self):
        def callback(sv):
            print("Callback " + sv.get())

        sv = tk.StringVar()
        sv.trace("w", lambda name, index, mode, sv=sv: self.callback(sv))
        self.entry.config(textvariable=sv)
        sv.set("5")

    def close_command(self):
        self.master.destroy()
        plt.close()

    def update_callback(self,text):
        print("Update_callback")
        self.plotFn(self.fig.gca(), text)
        self.canvas.draw()

    # Not used: Problem with bind on <Key> is that it is called before the latest keypress is available
    def update_event(self,event):
        print("Update event")
        self.plotFn(self.fig.gca(), self.entry.get())
        self.canvas.draw()

    # Not used: but could be as part of manual button
    def update(self, redrawFn):
        print("Update")
        redrawFn()
        self.canvas.draw()

def myPlot(ax, paras=None):
    n = int(paras)
    x = np.random.random(n)
    y = np.random.random(n)
    ax.clear()
    ax.plot(x, y)
    print("My plot: {}".format(paras))

# Demo:
tk.Tk()
app=ChartUpdater_Old(plot_Fn = myPlot)
app.mainloop()





#######################################

import tkinter as tk


class SimpleUpdater():
    def __init__(self):
        # Note: Need to have a close button to .quit and .destroy
        top = tk.Toplevel()

        def callback(sv):
            print("Try:" + sv.get())

        self.sv = tk.StringVar()
        tk.Entry(top, textvariable=self.sv).grid(row=0, column=1)

        self.sv.trace("w", lambda name, index, mode, sv=self.sv: callback(sv))

        top.mainloop()


app=SimpleUpdater()



#######################################

import tkinter as tk

# This now works - did have problem that trace was not triggering
class SimpleUpdater2():
    def __init__(self, master):
        # Note: Need to have a close button to .quit and .destroy

        def callback(sv):
            print("Try:" + sv.get())

        self.sv = tk.StringVar()
        tk.Entry(master, textvariable=self.sv).grid(row=0, column=1)

        self.sv.trace("w", lambda name, index, mode, sv=self.sv: callback(sv))


root = tk.Tk()
app=SimpleUpdater2(root)
root.mainloop()


###########################################################









########################### Other ways ##################################

from matplotlib.backends.backend_agg import FigureCanvasAgg


import matplotlib.backends.tkagg as tkagg

#import matplotlib
#matplotlib.use('TkAgg')
import tkinter.ttk as ttk
import sys
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def draw_figure(canvas, figure, loc=(0, 0)):
    """ Draw a matplotlib figure onto a Tk canvas

    loc: location of top-left corner of figure on canvas in pixels.
    Inspired by matplotlib source: lib/matplotlib/backends/backend_tkagg.py
    """
    figure_canvas_agg = FigureCanvasAgg(figure)
    figure_canvas_agg.draw()
    figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds
    figure_w, figure_h = int(figure_w), int(figure_h)
    photo = tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)

    # Position: convert from top-left anchor to center anchor
    canvas.create_image(loc[0] + figure_w/2, loc[1] + figure_h/2, image=photo)

    # Unfortunately, there's no accessor for the pointer to the native renderer
    tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)

    # Return a handle which contains a reference to the photo object
    # which must be kept live or else the picture disappears
    return photo

class MyFirstGUI:

    def __init__(self, master):
        self.master = master
        master.title("A simple GUI")

        self.label = tk.Label(master, text="This is our first GUI!")
        self.label.pack()

        self.canvas = tk.Canvas(master, width=300,height=200)
        self.canvas.pack
        self.fig_photo = None

        self.text = ""
        self.entry = tk.Entry(master)
        self.entry.insert(0, "...")
        self.entry.bind("<Return>", self.update)
        self.entry.pack()

        self.greet_button = tk.Button(master, text="Update", command=self.update)
        self.greet_button.pack()

        self.close_button = tk.Button(master, text="Close", command=master.quit)
        self.close_button.pack()

    def greet(self):
        print("Greetings!")

    def update(self):
        data = np.random.random(10)
        self.label.configure(text = "Hello:" + self.entry.get())
        fig = plt.plot(data)
        self.fig_photo = draw_figure(self.canvas, fig, loc=(100,200))

# root = tk.Tk()
# my_gui = MyFirstGUI(root)
# root.mainloop()

class mclass:
    def __init__(self,  window):
        self.window = window
        self.box = tk.Entry(window)
        self.button = tk.Button (window, text="check", command=self.plot)
        self.box.pack ()
        self.button.pack()

    def plot (self):

        x=np.array ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        v= np.array ([16,16.31925,17.6394,16.003,17.2861,17.3131,19.1259,18.9694,22.0003,22.81226])
        p= np.array ([16.23697,     17.31653,     17.22094,     17.68631,     17.73641 ,    18.6368,
            19.32125,     19.31756 ,    21.20247  ,   22.41444   ,  22.11718  ,   22.12453])

        fig = plt.Figure(figsize=(6,6))
        a = fig.add_subplot(111)
        a.scatter(v,x,color='red')
        a.plot(p, range(2 +max(x)),color='blue')

        canvas = FigureCanvasTkAgg(fig, master=self.window, )
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# window= tk.Tk()
# start= mclass (window)
# window.mainloop()
