#!/usr/bin/env python3
import argparse
import sys
import re
import csv
import os
import pandas as pd
sys.path.append('../')

from logparser import Face
from excel import Excel

#
# default value
#
input_acc_file= 'accuracy.txt'
output_dir    = 'result/' # The output directory of parsing results
input_dir     = '.' # The input directory of log file
shuffix       = "_accuracy"
output_accuracy='accuracy.xls'

# support log format
faceLogs = ['retinaface_mnet25', 'retinaface_res50']
lfwLogs = ["arcface_res50"]
detectionLogs = ["ssd300", "yolo_v3_416"]
imagenetLogs = ["resnet50", "mobilenet_v2", "vgg16", "googlenet", "inception_v4", "shufflenet_v2", "squeezenet"]
livenessLogs = ["liveness"]

# global excel format
title = ["Acc@1", "Acc@5", "AP", "AR", "AUC", "TPR", "AP-easy"]
# plz ref https://docs.google.com/spreadsheets/d/1ihNaZcUh7961yU7db1-Db0lbws4NT24B7koY8v8GHNQ/pubhtml?gid=1072579560&single=true
#int8_bg_style = "pattern: pattern solid, fore_color pale_blue; "
int8_bg_style = 'font: bold off, color black;\
                     borders: top_color black, bottom_color black, right_color black, left_color black,\
                              left thin, right thin, top thin, bottom thin;\
                     pattern: pattern solid, fore_color pale_blue;'
sheet_all = "all"
excel = None
curr_row = 0

def gen_report_title(excel):
    title_cells = len(title)
    col_start = 0

    # head - title 
    excel = excel.write(row=0, col=col_start, content="Quantization", sheet=sheet_all)
    col_start = col_start + 1

    # head - caffe
    excel = excel.write(row=0, col=col_start, top_row = 0, end_row = 0,
        left_column=col_start, right_column=col_start + title_cells - 1, content="caffe", sheet=sheet_all, style="align: horiz center")

    for t in title:
        excel = excel.title(row=1, col=col_start, content = t, sheet=sheet_all)
        col_start = col_start + 1
     
    # head - int8_multiplier
    excel = excel.write(row=0, col=col_start, top_row = 0, end_row = 0,
        left_column=col_start, right_column=col_start + title_cells - 1, content="int8_multiplier", sheet=sheet_all, style="align: horiz center")

    for t in title:
        excel = excel.title(row=1, col=col_start, content = t, sheet=sheet_all)
        col_start = col_start + 1

    global curr_row
    curr_row = 2

def get_export_csv_name(log_file):
    log_shuffix_name = os.path.splitext(log_file)[0]
    net = log_shuffix_name.split(shuffix)[0]
    out_structured = os.path.join(output_dir, "{}.log_structured.csv". format(log_shuffix_name))
    return net, out_structured

#
# parser by dataset
#
def parseFace(log_file):
    log_format    = '<Component> AP of <Level> is <Accuracy>' # HDFS log format
    minEventCount = 0 # The minimum number of events in a bin
    merge_percent = 0.5 # The percentage of different tokens 
    #regex         = [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'] # Regular expression list for optional preprocessing (default: [])
    regex         = [] # Regular expression list for optional preprocessing (default: [])

    # parse log
    parser = Face.LogParser(input_dir, output_dir, log_format, rex=regex,
               minEventCount=minEventCount, merge_percent=merge_percent)
    parser.parse(log_file)

    # get base name / export file name
    net, out_structured = get_export_csv_name(log_file)

    # read csv
    my_csv = pd.read_csv(out_structured)

    # hard code
    caffe_acc = "{:.2f}".format(100 * float(my_csv["Accuracy"][2]))
    int8_acc  = "{:.2f}".format(100 * float(my_csv["Accuracy"][5]))

    global curr_row
    # write easy_val only
    excel.write(row = curr_row, col=0, content=net, sheet=sheet_all)
    excel.write(row = curr_row, col=len(title), content=caffe_acc, sheet=sheet_all)
    excel.write(row = curr_row, col=2 * len(title), content=int8_acc, sheet=sheet_all,
        style=int8_bg_style)

    # export detail to sheet
    exclude_idx = [my_csv.columns.get_loc(i) for i in ["LineId", "EventId", "EventTemplate", "ParameterList"]]
    for index, row in my_csv.iterrows():
        for col in range(row.shape[0]):
            if col not in exclude_idx:
                excel.write(row = index, col=col, content=row[col], sheet=net)

    curr_row = curr_row + 1


def parseLFW(log_file):
    log_format    = '<Component>: <Accuracy>' # HDFS log format
    minEventCount = 2 # The minimum number of events in a bin
    merge_percent = 0.5 # The percentage of different tokens 
    #regex         = [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'] # Regular expression list for optional preprocessing (default: [])
    regex         = [r'.+Diff.+'] # Regular expression list for optional preprocessing (default: [])
    exceptRex     = r'Diff' # Regular expression list for optional preprocessing (default: [])

    parser = Face.LogParser(input_dir, output_dir, log_format, rex=regex, 
               minEventCount=minEventCount, merge_percent=merge_percent,
               keep_para=False, exceptRex=exceptRex)
    parser.parse(log_file)

    # get base name / export file name
    net, out_structured = get_export_csv_name(log_file)

    # read csv
    my_csv = pd.read_csv(out_structured)

    # fill net name
    global curr_row
    excel.write(row = curr_row, col=0, content=net, sheet=sheet_all)

    # hard code
    # 0 is caffe 1 is int8
    acc_order = 0
    for index, row in my_csv.iterrows():
        if row.Component.strip() == "AUC":
            style = int8_bg_style if acc_order == 1 else ""
            acc = "{:.2f}".format(float(row.Accuracy) * 100)
            excel.write(row = curr_row, col=(acc_order) * len(title) +  5, content=acc, sheet=sheet_all, style = style)
            acc_order = acc_order + 1

    curr_row = curr_row + 1



def parseDetection(log_file):
    log_format    = '<Component> = <Accuracy>' # HDFS log format
    minEventCount = 0 # The minimum number of events in a bin
    merge_percent = 0.5 # The percentage of different tokens 
    #regex         = [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'] # Regular expression list for optional preprocessing (default: [])
    regex         = [r'(Average.+)'] # Regular expression list for optional preprocessing (default: [])

    parser = Face.LogParser(input_dir, output_dir, log_format, rex=regex, 
               minEventCount=minEventCount, merge_percent=merge_percent,
               keep_para=False)
    parser.parse(log_file)

    # get base name / export file name
    net, out_structured = get_export_csv_name(log_file)

    # read csv
    my_csv = pd.read_csv(out_structured)

    # fill net name
    global curr_row
    excel.write(row = curr_row, col=0, content=net, sheet=sheet_all)

    # hard code
    # export detail to sheet
    idx = 0
    # 0 is caffe 1 is int8
    acc_order = 0
    # hard code, 0 is caffe 1 is fp32 2 is int8 3 is bf16
    if net.strip().startswith("yolo"):
        is_yolo = True
        acc_col_idx = [0, -1, 1, -1]
    else:
        is_yolo = False
        acc_col_idx = [0, 1]

    exclude_idx = [my_csv.columns.get_loc(i) for i in ["LineId", "EventId", "EventTemplate"]]
    for index, row in my_csv.iterrows():
        if row.Component.find('Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100') != -1:
            acc = "{:.2f}".format(100 * float(row.Accuracy))
            if not is_yolo or (is_yolo and acc_order == 0 or acc_order == 2):
                style = int8_bg_style if acc_col_idx[acc_order] == 1 else ""
                excel.write(row = curr_row, col=acc_col_idx[acc_order] * len(title) + 3,
                    content=row.Component + (acc), sheet=sheet_all, style=style)

        if row.Component.find('Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1') != -1:
            acc = "{:.2f}".format(100 * float(row.Accuracy))
            if not is_yolo or (is_yolo and acc_order == 0 or acc_order == 2):
                excel.write(row = curr_row, col=acc_col_idx[acc_order] * len(title) + 4,
                    content=row.Component + (acc), sheet=sheet_all, style=style)
            acc_order = acc_order + 1

        if row.Component.strip().startswith("Average"):
            for col in range(row.shape[0]):
                if col not in exclude_idx:
                    excel.write(row = idx, col=col, content=row[col], sheet=net)

            idx = idx + 1

    curr_row = curr_row + 1



def parseImagenet(log_file):
    log_format    = '<Component> Acc@1 <Accuracy1> Acc@5 <Accuracy5>' # HDFS log format
    minEventCount = 0 # The minimum number of events in a bin
    merge_percent = 0.2 # The percentage of different tokens 
    #regex         = [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'] # Regular expression list for optional preprocessing (default: [])
    regex         = [r'((?!\().)*', r'((?!Loss).)*'] # Regular expression list for optional preprocessing (default: [])
    exceptRex     = r'Time' # Regular expression list for optional preprocessing (default: [])

    parser = Face.LogParser(input_dir, output_dir, log_format, rex=regex,
               minEventCount=minEventCount, merge_percent=merge_percent,
               keep_para=False, exceptRex=exceptRex)
    parser.parse(log_file)

    # get base name / export file name
    net, out_structured = get_export_csv_name(log_file)

    # read csv
    my_csv = pd.read_csv(out_structured)

    # fill net name
    global curr_row
    excel.write(row = curr_row, col=0, content=net, sheet=sheet_all)

    # hard code
    # export detail to sheet
    # 0 is caffe 1 is fp32 2 is int8
    caffe_1_acc = my_csv["Accuracy1"][0]
    caffe_5_acc = my_csv["Accuracy5"][0]
    int8_1_acc = my_csv["Accuracy1"][2]
    int8_5_acc = my_csv["Accuracy5"][2]
    excel.write(row = curr_row, col=1 + 0 * len(title), content=caffe_1_acc, sheet=sheet_all)
    excel.write(row = curr_row, col=2 + 0 * len(title), content=caffe_5_acc, sheet=sheet_all)
    excel.write(row = curr_row, col=1 + 1 * len(title), content=int8_1_acc, sheet=sheet_all, style=int8_bg_style)
    excel.write(row = curr_row, col=2 + 1 * len(title), content=int8_5_acc, sheet=sheet_all, style=int8_bg_style)

    curr_row = curr_row + 1


def parseLiveness(log_file):
    log_format    = 'FPR@<Component> <Content> <FPR> - <TPR>' # HDFS log format
    minEventCount = 0 # The minimum number of events in a bin
    merge_percent = 0.2 # The percentage of different tokens 
    #regex         = [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'] # Regular expression list for optional preprocessing (default: [])
    regex         = [] # Regular expression list for optional preprocessing (default: [])

    parser = Face.LogParser(input_dir, output_dir, log_format, rex=regex, 
               minEventCount=minEventCount, merge_percent=merge_percent)
    parser.parse(log_file)

    # get base name / export file name
    net, out_structured = get_export_csv_name(log_file)

    # read csv
    my_csv = pd.read_csv(out_structured)

    # fill net name
    global curr_row
    excel.write(row = curr_row, col=0, content=net, sheet=sheet_all)

    # hard code
    # export detail to sheet
    # 0 is caffe 1 is int8
    caffe_acc = "{:.2f}".format(100 * float(my_csv["TPR"][0]))
    int8_acc = "{:.2f}".format(100 * float(my_csv["TPR"][1]))
    excel.write(row = curr_row, col=6 + 0 * len(title), content=caffe_acc, sheet=sheet_all)
    excel.write(row = curr_row, col=6 + 1 * len(title), content=int8_acc, sheet=sheet_all, style=int8_bg_style)

    # export detail to sheet
    # caffe 
    excel.write(row = 0, col=0, 
        content="FPR:" + str(my_csv["FPR"][0].replace("TPR : ", "")), sheet=net)
    excel.write(row = 0, col=1, 
        content="TPR:" + str(my_csv["TPR"][0]), sheet=net)
    # int8
    excel.write(row = 1, col=0, 
        content="FPR:" + str(my_csv["FPR"][0].replace("TPR : ", "")), sheet=net)
    excel.write(row = 1, col=1, 
        content="TPR:" + str(my_csv["TPR"][0]), sheet=net)
    
    curr_row = curr_row + 1

def rawcount(filename):
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)

    return lines

#
# main function
#

if __name__ == "__main__":
    parser = argparse.ArgumentParser("auto gen accuracy report")
    parser.add_argument(
        "--input_acc_file",
        help="parallel accuracy txt, the possible content as {}".format(
          "$REGRESSION_PATH/parallel/run_accuracy.sh resnet50 50000"
          ),
        default=input_acc_file
        )
    parser.add_argument(
        "--acc_log_path",
        help="accuracy log path, the passible file"
        " name in path is resnet50{}.log".format(shuffix),
        default=input_dir
    )
    parser.add_argument(
        "--output_dir",
        help="output dictory",
        default=output_dir
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    input_dir = args.acc_log_path
    pattern = re.compile("^(?P<Script>.*?)\\s+(?P<NET>.*?)\\s+(?P<Count>.*?)$")

    excel = Excel(file=output_accuracy, sheet="all")
    print("output to {}".format(output_accuracy))

    # fill background color
    rows = rawcount(args.input_acc_file)
    for i in range(rows):
        for j in range(len(title)):
            excel = excel.write(row=2 + i, col=len(title) + 1 + j, sheet=sheet_all, style=int8_bg_style)

    gen_report_title(excel)

    with open(args.input_acc_file, 'r') as fin:
        for line in fin.readlines():
          try:
              match = pattern.search(line.strip())
              net = match.group("NET")
              net_name = "{}{}.log".format(net, shuffix)
              if net in faceLogs:
                  parseFace(net_name)
              elif net in lfwLogs:
                  parseLFW(net_name)
              elif net in detectionLogs:
                  parseDetection(net_name)
              elif net in imagenetLogs:
                  parseImagenet(net_name)
              elif net in livenessLogs:
                  parseLiveness(net_name)
              else:
                  print("Not support", net_name)

          except Exception as e:
              pass

    excel.save()
    print("export to ", excel.file)

