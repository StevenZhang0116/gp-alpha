"""
从log文件中的阿尔法公式生成用于提交的.py文件. 可以在遗传规划结束后调用, 也可以单独python3调用.
"""

# 加载自定义包
from base_classes_funcs import *


def copy_code_from_file(file_name):
    """读取file_name中COPY到ENDCOPY间的部分"""
    with open(file_name, 'r') as f:
        text = f.read()

    i_start = text.find('# COPY #')
    i_end = text.find('# ENDCOPY #')
    return text[i_start: i_end]


def gen_code_from_log(log_lines, valid_size, start_date_train):
    """从log文件中生成代码"""

    # 决定生成的文件的文件名
    save_folder = '/home/zhyu/asim_4.0.0/asim_client/pyx/gp_alpha'
    max_index = 0
    for file_name in os.listdir(save_folder):
        if file_name[:2] == 'gp' and file_name.split('.')[0].split('_')[-1].isnumeric():
            max_index = max(max_index, int(file_name.split('.')[0].split('_')[-1]))
    file_postfix = str(max_index + 1).zfill(3)

    # 读取最后一行, 它是dict_data_eval_info.keys(). 然后剔除最后一行.
    dict_data_eval_info_keys = log_lines[-1].strip().split(',')
    log_lines = log_lines[:-1]

    for str_expr in log_lines[::-1]:
        if str_expr.strip()[-1] != ')':  # 到"以下为最终达标的个体:"时停止
            break
        text = f"""
# 遗传规划 gpAlpha{file_postfix}
import alphaequdailybase
"""

        text += copy_code_from_file('base_classes_funcs.py')
        text += copy_code_from_file('operators.py')
        text += copy_code_from_file('data.py')
        text += f"""
class DailyEquAlpha(alphaequdailybase.AlphaEquDailyBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        # 定义新的函数: f_load_data_submit, 使用self.get_data获取数据
        def f_load_data_submit(field):
            slc1 = slice(self.nStartDi - {valid_size}, self.nEndDi + 1)  # 注意这里不delay, data.py中已delay.
            slc2 = slice(None)
            slc3 = slice(None)
            data_field = self.get_data(field)
            data_field = np.array(data_field[slc1, slc3])
            return data_field
        
        data = load_and_preprocess_data(self.delay, f_load_data_submit)
        """  # 这种写法使得下一行开始就有缩进

        # text += copy_code_from_file('run_daily.py').replace('\n', '\n    ')  # 需要多一个tab缩进
        text += f"""
        data_train = {{k: v[{valid_size}:] for k, v in data.items()}}  # 剔除前backdays天, 而非backdays - 1
        """
        for field in dict_data_eval_info_keys:
            # 例如OPEN = data_train['OPEN']
            text += f"""{field} = data_train['{field}']
        """
        text += f"""alpha = {str_expr}
        self._alpha = alpha

    def work(self, di):
        self.alpha[:] = self._alpha[di - self.nStartDi]

        """
        py_name = 'gp_alpha_{}.py'.format(file_postfix)

        with open(os.path.join(save_folder, py_name), 'w') as f:
            f.write(text)
        print("已生成{}".format(os.path.join(save_folder, py_name)))

        xml_name = 'gp_alpha_{}.xml'.format(file_postfix)
        text = f"""
<?xml version="1.0" encoding="utf8"?>
<baseinfo asset="equity" region="cn" areas="cn" />
<const commonconfig="/work/asim/release/${{ver}}/config/config_common_cn_equ.xml" />
<include target="${{commonconfig}}" only="brickPile,const,const_windows,const_linux" />
<portfolio brick="BrickControllerCommon" >
        <include target="${{commonconfig}}" only="tradedate,universe" data="${{cacheroot}}/data" backdays="{valid_size}" startdate="{start_date_train}" enddate="20201231" readonly="true" />
        <datapool brick="DataPool" skipUpdate="true" >
                <include target="${{commonconfig}}" only="data" />
                <data brick="LibLoadPythonEnv" pypath="${{asimpath}}" global_symbol="true" />
        </datapool>
        <strategy brick="StrategyEquityDaily" delay="1" printdate="false" >
                <stats brick="StatsEquityDailySimple" maxCap="2e7" feeRate="0.00"/>
                <alpha alphaid="gp_alpha_{file_postfix}" brick="AlphaPythonWrapperEquDaily" subUniv="UnivTOP4000" pymodule="gp_alpha_{file_postfix}" pypath={save_folder}>
                </alpha>
        </strategy>
        <summary brick="SummaryEquity" showSummary="true" showOS="false" exportPNLcsv="false" />
        <pnl brick="BrickGenPNL" pnlFolder="pnl/mypnl" quiet="false" />
</portfolio>
        """
        with open(os.path.join(save_folder, xml_name), 'w') as f:
            f.write(text)
        print("已生成{}".format(os.path.join(save_folder, xml_name)))
        file_postfix = str(int(file_postfix) + 1).zfill(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, required=True, default='', help='log名称, 如20210806_235326')
    parser.add_argument('--valid_size', type=int, required=False, default=244, help='验证集天数')
    parser.add_argument('--start_date_train', type=int, required=False, default=20160104, help='开始日期')
    # 注意这里start_date_train与run_daily.py中start_date不同, 这里的date往前取valid_size天后为那个start_date
    args = parser.parse_args()

    with open(os.path.join('./logs', args.log + '.txt'), 'r') as f:
        log_lines = f.readlines()

    gen_code_from_log(log_lines, args.valid_size, args.start_date_train)
