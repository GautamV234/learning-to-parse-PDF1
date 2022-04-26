import matplotlib
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('TkAgg')
import pandas as pd

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

def graph(event=None,image_name='test'):
    tmptext = entry.get()
    tmptext = "$"+tmptext+"$"

    ax.clear()
    ax.text(0.2, 0.3, tmptext, fontsize=50)  
    # canvas.draw()
    canvas.print_jpg(f"images/{image_name}.jpg")


data = pd.read_csv('data.csv')
print(data.head)
formulae = data['latex']
names = data['image']
print(len(formulae))
for name,formula in zip(names,formulae):
    root = tk.Tk()
    name = name.split(".")[0]
    print(f"name is {name}")
    mainframe = tk.Frame(root)
    mainframe.pack()

    entry = tk.Entry(mainframe, width=70)
    entry.pack()

    label = tk.Label(mainframe)
    label.pack()

    fig = matplotlib.figure.Figure(figsize=(50, 40), dpi=20)
    ax = fig.add_subplot(111)

    canvas = FigureCanvasTkAgg(fig, master=label)
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
    canvas._tkcanvas.pack(side="top", fill="both", expand=True)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    term = formula
    entry.insert(0,term)
    print(term)
    # graph(image_name=name)

# root.bind("<Return>", graph)
# root.mainloop()