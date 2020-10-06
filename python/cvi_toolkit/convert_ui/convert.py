import os
import pygubu

import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from tkinter import messagebox

import threading
import subprocess

import queue
import time
import concurrent.futures

output_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.cvimodel")
std_output_flag = {'stdout': subprocess.DEVNULL, 'stderr': subprocess.STDOUT}
std_output_flag = {}

class ThreadedTask(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
    def run(self):
        time.sleep(1)  # Simulate long running process
        self.queue.put("Task finished")

def hide_me(event):
    event.widget.pack_forget()

# Function responsible for the updation
# of the progress bar value
def bar(progress, root):
    import time
    # infinity process
    t = threading.currentThread()
    progress.grid()
    while getattr(t, "do_run", True):
        time.sleep(0.5)
        print("do riun")
        #progress['value'] = 20
        #root.update_idletasks()
        #time.sleep(0.5)

        #progress['value'] = 40
        #root.update_idletasks()
        #time.sleep(0.5)

        #progress['value'] = 50
        #root.update_idletasks()
        #time.sleep(0.5)

        #progress['value'] = 60
        #root.update_idletasks()
        #time.sleep(0.5)

        #progress['value'] = 80
        #root.update_idletasks()
        #time.sleep(0.5)

        #progress['value'] = 100
        #root.update_idletasks()
        #time.sleep(0.5)

        #progress['value'] = 80
        #root.update_idletasks()
        #time.sleep(0.5)

        #progress['value'] = 60
        #root.update_idletasks()
        #time.sleep(0.5)

        #progress['value'] = 50
        #root.update_idletasks()
        #time.sleep(0.5)

        #progress['value'] = 40
        #root.update_idletasks()
        #time.sleep(0.5)

        #progress['value'] = 20
        #root.update_idletasks()
        #time.sleep(0.5)
        #progress['value'] = 0
        #print("proc")

class MyApp():
    def __init__(self):
        #self.parant = parent
        # diction for \regression/convert_model.sh
        self.params = {
            'i': '', #model_def="$OPTARG" ;;
            'd': '', #model_data="$OPTARG" ;;
            't': '', #model_type="$OPTARG" ;;
            'b': '', #bs="$OPTARG" ;;
            'q': '', #cali_table="$OPTARG" ;;
            'v': '', #chip_ver="$OPTARG" ;;
            'o': '', #output="$OPTARG" ;;
            'l': '', #do_layergroup="$OPTARG" ;;
            #'p': '', #do_fused_preprocess="1" ;;
            #'z': '', #net_input_dims="$OPTARG" ;;
            #'r': '', #raw_scale="$OPTARG" ;;
            #'m': '', #mean="$OPTARG" ;;
            #'s': '', #std="$OPTARG" ;;
            #'a': '', #input_scale="$OPTARG" ;;
            #'w': '', #channel_order="$OPTARG" ;;
            #'y': '', #image_resize_dims="$OPTARG" ;;
            #'f': '', #crop_offset="$OPTARG" ;;
            #'h': '', #usage
            }

        self.help = {
            'i': "model def",
            'd': "model data",
            't': "model type",
            'b': "batch",
            'q': "model calibration table",
            'v': "chip version",
            'o': "output",
            'l': "enable layergroup",
            }

        self.builder = builder = pygubu.Builder()
        builder.add_from_file('convert.ui')
        self.mainwindow = builder.get_object('mainwindow')
        self.tkcombo = builder.get_object('combobox1')
        self.model_type_msg = builder.get_object('model_type_msg')
        self.model_data_msg = builder.get_object('model_data_msg')
        self.model_cali_msg = builder.get_object('model_cali_msg')
        self.model_output_msg = builder.get_object('model_output_msg')
        self.model_convert_msg = builder.get_object('model_convert_msg')
        self.model_convert_msg.grid_forget()
        #self.model_batch_msg = builder.get_object('model_batch_msg')

        self.model_def = builder.get_object('model_def')
        self.model_data = builder.get_object('model_data')
        self.chip_ver = builder.get_object('chip_ver')
        self.model_cali = builder.get_object('model_cali')
        #self.model_batch = builder.get_object('model_batch')
        self.model_output = builder.get_object('model_output')
        self.model_do_layergroup_1 = builder.get_object('model_do_layergroup_1')
        self.model_do_layergroup_0 = builder.get_object('model_do_layergroup_0')
        self.model_convert = builder.get_object('model_convert')

        self.model_progressbar = builder.get_object('model_progressbar')
        # update ui
        self.model_progressbar.grid()
        self.model_progressbar.grid_forget()

        #batch_var = tk.IntVar()
        #batch_menu = tk.Menu(self.model_batch, tearoff=False)
        #self.model_batch.configure(menu=batch_menu)

        #self.model_def.place(x=999, y=999) # visible it
        #self.model_def.pack_forget()
        #self.model_def.pack()

        # append event
        def on_button_click(event):
            filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
            self.params['i'] = filename
            self.model_type_msg.configure(text=filename)

        self.model_def.bind('<Button-1>', on_button_click)


        def model_data_click(event):
            filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
            self.params['d'] = filename
            self.model_data_msg.configure(text=filename)

        self.model_data.bind('<Button-1>', model_data_click)


        def model_cali_click(event):
            filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
            self.params['q'] = filename
            self.model_cali_msg.configure(text=filename)

        self.model_cali.bind('<Button-1>', model_cali_click)

        def model_output_click(event):
            dirname = askdirectory()
            if dirname:
                self.params['o'] = os.path.join(dirname, "output.cvimodel")
                self.model_output_msg.configure(text=self.params['o'])

        self.model_output.bind('<Button-1>', model_output_click)


        def model_do_layergroup_1_click(event):
            self.params['l'] = "1"

        self.model_do_layergroup_1.bind('<Button-1>', model_do_layergroup_1_click)

        def model_do_layergroup_0_click(event):
            self.params['l'] = "0"

        self.model_do_layergroup_0.bind('<Button-1>', model_do_layergroup_0_click)

        #def model_batch_click(event):
        #    w = event.widget
        #    #print(batch_var.get())
        #    self.params['b'] = batch_var.get()
        #    self.model_batch_msg.configure(text=batch_var.get())

        #batch_menu.bind('<<MenuSelect>>', model_batch_click)

        def model_convert_click(event):
            self.model_convert_msg.configure(text="")
            self.params['v'] = self.chip_ver.get()
            errs = []
            convert_cmd = ['convert_model.sh']
            is_caffe = self.params['t'] == 'caffe'
            for k in self.params:
                if not is_caffe and k == 'd':
                    # skip check data in onnx/tensor
                    pass
                elif not self.params[k]:
                    errs.append(self.help[k])
                convert_cmd.extend(["-" + k, self.params[k]])

            #builder.get_object('Frame_2').attribute("-alpha", 0.0)
            #print(dir(self.mainwindow))

            if errs:
                messagebox.showinfo("invalid params",
                        "please select property attribute\n{}".format("\n".join(errs)))
                return

            # Progress bar widget
            #bar(self.model_progressbar, self.mainwindow)
            #t = threading.Thread(target = bar,
            #        args = (self.model_progressbar, self.mainwindow,))
            #t.start()
            event.widget.grid_forget()
            self.model_convert_msg.grid_forget()
            self.model_progressbar.grid()
            self.model_progressbar.start()
            #self.queue = queue.Queue()
            #ThreadedTask(self.queue).start()
            #self.mainwindow.after(1, self.process_queue)
            #print(" ".join(convert_cmd))
            ret = None
            #def r(convert_cmd, std_output_flag):
            #    p = subprocess.Popen(convert_cmd, stdout=subprocess.PIPE)
            #    print(p.pid, "pid")
            #    #try:
            #    #    ret = subprocess.run(convert_cmd, **std_output_flag)
            #    #except exception as error:
            #    #    messagebox.showinfo("invalid setting",
            #    #            "maybe not source ./envsetup.sh, err is {}".format(error))
            #    #    raise
            #    #ret.wait()

            #t = threading.Thread(target = r,
            #        args = (convert_cmd, std_output_flag, ))
            #t.start()
            try:
                ret = subprocess.run(convert_cmd, **std_output_flag)
            except exception as error:
                messagebox.showinfo("invalid setting",
                        "maybe not source ./envsetup.sh, err is {}".format(error))
                raise

            #t.do_run = False
            #t.join()

            #ret = None
            #with concurrent.futures.ThreadPoolExecutor() as executor:
            #    future = executor.submit(r, convert_cmd, std_output_flag)
            #    ret  = future.result()
            #    print(ret)

            text = "convert success, export to {}".format(self.params['o'])
            fg="green"
            #return

            if ret.returncode == 0:
                pass
            else:
                fg="red"
                text = "convert fail, err is {}".format(ret.returncode)

            event.widget.grid(row=10, sticky=tk.E)
            self.model_progressbar.grid_forget() # hide progress for show message
            self.model_convert_msg.grid()
            self.model_convert_msg.configure(text=text, fg=fg)
            print("status:", text)

        self.model_convert.bind('<Button-1>', model_convert_click)

        self.tkcombo.bind('<<ComboboxSelected>>', self.on_tkcombo_select)

        # append default value
        options = ['caffe', 'onnx', 'tensorflow']
        self.tkcombo.config(values=options)
        self.tkcombo.set('caffe')
        self.params['t'] = self.tkcombo.get()


        #batch_menu.add_radiobutton(label="1", variable=batch_var, value=1)
        #batch_menu.add_radiobutton(label="2", variable=batch_var, value=2)
        #batch_menu.add_radiobutton(label="3", variable=batch_var, value=3)
        #batch_menu.add_radiobutton(label="4", variable=batch_var, value=4)
        #batch_var.set(1)
        #self.model_batch_msg.configure(text=batch_var.get())
        #self.params['b'] = batch_var.get()
        self.params['o'] = output_name
        self.model_output_msg.configure(text=self.params['o'])

        self.pycombo = builder.get_object('combobox2')
        options2 = [
            ('1', 1),
            ('2', 2),
            ('3', 3),
            ('4', 4),
            ]
        self.pycombo.configure(values=options2)
        self.pycombo.bind('<<ComboboxSelected>>', self.on_pycombo_select)
        self.pycombo.set('1')
        self.params['b'] = self.pycombo.get()

        self.model_do_layergroup_1.select()
        self.params['l'] = "1"

    def on_tkcombo_select(self, event):
        #self.model_def.place(x=0, y=30)
        #print(self.model_def.winfo_rootx(), self.model_def.winfo_rooty())
        #print(self.tkcombo.get())
        self.params['t'] = self.tkcombo.get()
        if self.params['t'] == "caffe":
            self.model_data.grid(row=4, column=0, sticky=tk.W)
            self.model_data_msg.grid(row=4, column=1)
        else:
            self.model_data.grid_forget()
            self.model_data_msg.grid_forget()

    def on_pycombo_select(self, event):
        self.params['b'] = self.pycombo.get()

    def process_queue(self):
        try:
            msg = self.queue.get(0)
            # Show result of the task if needed
            self.model_progressbar.stop()
        except queue.Empty:
            self.mainwindow.after(100, self.process_queue)

    def run(self):
        self.mainwindow.mainloop()


if __name__ == '__main__':
    app = MyApp()
    app.run()
