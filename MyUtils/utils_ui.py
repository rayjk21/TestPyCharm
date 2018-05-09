
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt
# noinspection PyCompatibility
import tkinter as tk

import time


####################################

class ChartUpdater():
    def __init__(self, plot_Fn=None):
        # Create a TopLevel window to hold widgets in
        top = tk.Toplevel()

        # Keep handles for updating and closing etc
        self.top = top
        self.plotFn = plot_Fn

        # Use to fire event when text entry changes
        sv = tk.StringVar()
        sv.trace("w", lambda name, index, mode: self.update_callback(sv.get()))

        # Keep handle on figure for updating
        self.fig=plt.figure(figsize=(8,8))
        self.canvas=FigureCanvasTkAgg(self.fig,master=top)
        self.canvas.draw()

        tk.Label(top, text="Enter initial letters: ").grid(row=0, column=0)
        self.canvas.get_tk_widget().grid(row=1,column=1)

        tk.Entry(top, textvariable=sv).grid(row=0, column=1)
        tk.Button(master=top, text="Close", command=self.close_command).grid(row=0, column=2)

        # Start tk processing on this window
        top.mainloop()

    def test_callback(self, sv_):
        print("Try:" + sv_.get())

    def update_callback(self,text):
        self.plotFn(self.fig.gca(), text)
        self.canvas.draw()

    def close_command(self):
        # Stop tk general processing
        self.top.quit()
        # Kill this window
        self.top.destroy()
        # Close the plot
        plt.close()


def __ChartUpdaterDemo():

    def myPlot(ax, paras=None):
        n = int(paras)
        x = np.random.random(n)
        y = np.random.random(n)
        ax.clear()
        ax.plot(x, y)
        print("My plot: {}".format(paras))

    ChartUpdater(myPlot)








