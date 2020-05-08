# -*- coding=utf-8 -*-
#try:
import xlrd
from xlwt import Workbook, Worksheet, easyxf
import time
from os import path, mkdir
import traceback
from xlutils.copy import copy
#except Exception as e:
#    print('\033[1;31;0m' + "[Import Error]:" + '\033[0m' + " The imported modules are not exist: " + str(e))


# 读写Excel封装
class Excel(object):

    # 当前操作的文件完整路径
    file = ''
    _file_path = []

    # 错误码 0表示没有错误
    code = 0

    # 错误信息
    msg = None

    # 实例化的Excel文件对象
    _workbook = None
    _read_workbook = None

    # 实例化的sheet列表
    _worksheet = {}

    # 当前操作的sheet名称
    _sheet_name = None

    # 待写入sheet的数据
    _data = {}

    # 当前操作的最大行号
    _last_row = -1

    # 错误码
    # 写入Excel失败
    _WRITE_EXCEL_ERROR = 1
    # 创建文件失败
    _MAKE_DIR_ERROR = 2
    # 打开的文件不存在
    _NO_EXIST_FILE = 3

    def __init__(self, file=None, sheet=None, rebuild=True):
        self._file_path = []
        self.code = 0
        self.msg = None
        self._workbook = None
        self._read_workbook = None
        self._worksheet = {}
        self._data = {}
        self._last_row = -1
        file = path.realpath(file or "Excel\\"+time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))+".xlsx")
        self._file_path.append(file)
        # 创建路径中不存在的文件夹
        self.file = self._set_file(file)
        self._sheet_name = sheet or "sheet1"

        # 表存在时，是否重建新表
        self._rebuild = rebuild

        # 当前操作的Excel是否存在
        if path.exists(file):
            self._file_exist = True
        else:
            self._file_exist = False

    def file_exist(self):
        return self._file_exist
            
    def _get_read_workbook(self):
        """
            单例模式获取读取的excel workbook对象
            :return: workbook对象
            """
        if not self._read_workbook:
            self._read_workbook = xlrd.open_workbook(self.file, encoding_override='utf-8')
        return self._read_workbook

    def _read(self,sheet=None):
        self._clear_error()
        # 如果使用的是默认sheet名称
        # 默认获取第一个sheet对象
        self._sheet_name = sheet or self._sheet_name
        sheet = None
        try:
            excel = self._get_read_workbook()
            sheet = excel.sheet_by_name(unicode(self._sheet_name))
        except IOError as ie:
            self._print_error(self._NO_EXIST_FILE, str(ie))
        except Exception as e:
            if not sheet and self._sheet_name == "sheet1":
                sheet = excel.sheet_by_index(0)
            else:
                self._print_error(self.NO_EXIST_SHEET, e.message)
        return sheet

    def read(self, sheet=None):
        """
        读取文件内容
        :param file: 文件名称，默认使用实例化时指定的文件名
        :return: 文件内容字典
        """
        sheet = self._read(sheet)
        return Sheet(sheet)

    def _get_write_workbook(self):
        """
        单例模式获取待写入的excel workbook对象
        :return: workbook对象
        """
        if not self._workbook:
            self._workbook = Workbook(encoding='utf-8')
        return  self._workbook

    def _get_sheet(self, sheet=None):
        """"
        获取一个sheet对象,所有实例化的sheet保存在self._worksheet对象中
        """
        # 追加内容
        if not self._rebuild and self._file_exist:
            if sheet not in self._worksheet:
                w = copy(self._get_read_workbook())
                # 设置当前操作的excel对象
                self._workbook = w
                if sheet in w._Workbook__worksheet_idx_from_name:
                    self._worksheet[sheet] = w.get_sheet(w._Workbook__worksheet_idx_from_name[sheet])
                    # 更新待插入的行号
                    self._last_row = self._worksheet[sheet].last_used_row
        # 创建新的excel文本
        else:
            if sheet not in self._worksheet:
                w = self._get_write_workbook()
                self._worksheet[sheet] = w.add_sheet(sheet, cell_overwrite_ok=True)
        return self._worksheet[sheet]

    def title(self, row=0, col=0, content=None, sheet=None, absTitle = False):
        """
        当文件需要重建或者文件第一次创建时，写入标题
        :param row: 行号 ，默认为0
        如果该参数是一个list，这忽略其他属性，输入一行
        :param col: 列号，默认为0
        :param content: 内容，默认为空
        :param file: 文件名，默认使用实例化时指定的文件名
        :param absTitle: 无论如何都写入标题，该标题追加到文件最后一行
        :return: 写入是否成功的标志标志
        """
        if self._rebuild or not self._file_exist or absTitle:
            self.write(row, col, content, sheet)
        return self

    def write(self, row=0, col=0, content=None, sheet=None,
        left_column=0, right_column=0, top_row=0, end_row=0, style=""):
        """
        写入文件内容
        :param row: 行号 ，默认为0
        如果该参数是一个list，这忽略其他属性，输入一行
        :param col: 列号，默认为0
        :param content: 内容，默认为空
        :param file: 文件名，默认使用实例化时指定的文件名
        :return: 写入是否成功的标志标志
        """
        self._clear_error()
        self._sheet_name = sheet or self._sheet_name
        self._write(row, col, content, self._sheet_name, left_column, right_column,
        top_row, end_row, style)
        return self

    def _write(self, row, col, content, sheet, left_column, right_column, top_row, end_row, style):
        """
        把待写入的数据保存到self._data对象中
        :param row: 行号
        :param col: 列号
        :param content: 内容
        :param sheet: sheet名称
        :return: None
        """
        # 如果是追加数据，先获取实例化待写入的sheet对象
        if not self._rebuild and self._file_exist:
            self._get_sheet(sheet)

        if isinstance(row, int):
            # self._last_row = max(row, self._last_row)+1
            if row > self._last_row:
                self._last_row += 1
        else:
            self._last_row += 1
        if isinstance(row, list):

            for key, val in enumerate(row):
                self._data[str(sheet) + ":" + str(self._last_row) + ":" + str(key)] = [sheet, self._last_row, key, val, left_column, right_column, top_row, end_row, style]
        else:
            self._data[str(sheet) + ":" + str(row) + ":" + str(col)] = [sheet, row, col, content, left_column, right_column, top_row, end_row, style]

    def save(self):
        for item in self._data:
            item = self._data[item]
            try:
                sheet, row, col, content, left_column, right_column, top_row, end_row, style = item
                sheet = self._get_sheet(sheet)
                if len(style):
                    #sheet.write(row, col, content, easyxf("pattern: pattern solid, fore_color yellow; font: color white; align: horiz right"))
                    sheet.write(row, col, content, easyxf(style))
                else:
                    sheet.write(row, col, content)

                if left_column == 0 and right_column == 0:
                    pass
                else:
                    sheet.merge(top_row, end_row, left_column, right_column)

            except Exception as e:
                self._print_error(self._WRITE_EXCEL_ERROR, "Excel write error : " + str(e))
        self._workbook.save(self.file)
        self._data = {}

    def _set_file(self, file):
        if not path.exists(file):
            loc_path = [item for item in path.split(file)]
            self._file_path.insert(1, loc_path[0])
            self._file_path.insert(2, loc_path[1])
            self._file_path.pop(0)
            self._set_file(loc_path[0])
        else:
            for index, dir in enumerate(self._file_path):
                try:
                    if index == 0 or index+1 == len(self._file_path):continue

                    mkdir('\\'.join(self._file_path[:index+1]))
                except OSError as e:
                    self._print_error(self._MAKE_DIR_ERROR, "make dir error: " + e)

        return file

    def error(self):
        """
        如果操作失败，可以通过该函数查看错误原因
        格式为：错误码 和 错误信息
        {
            'code' : 1,
            'msg' : '文件不存在'
        }
        :return: 错误数据的字典
        """
        return {
            'code': self.code,
            'msg': self.msg
        }

    def _clear_error(self):
        """
        清除错误信息
        :return: None
        """
        self.code = 0
        self.msg = None
        self._e = None

    def _print_error(self, code=0, msg=""):
        """"
        打印错误信息
        """
        self.code = code
        self.msg = msg
        # print('\033[1;31;0m' + "[xml Error]:" + '\033[0m' + str(self._e))
        print ('\033[1;31;0m[' + msg+"]\033[0m")
        print (traceback.format_exc())


class Sheet(object):
    _sheet = None

    def __init__(self, sheet):
        # self.__dict__ = sheet.__dict__
        self._sheet = sheet

    def __call__(self, row=0, col=0):
        return self._sheet.cell(row, col).value

    def __getattr__(self, item):
        if item in self._sheet.__dict__:
            return self._sheet.__dict__[item]
        else:
            raise AttributeError('this attribute [ '+item+' ] is not exist')
