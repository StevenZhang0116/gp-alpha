"""
将pyx/gp_alpha/文件夹下未提交的py文件进行提交
"""

# 加载自定义包
from base_classes_funcs import *


if __name__ == "__main__":
    # 首先对比/home/zhyu/asim_4.0.0/asim_client/pyx/gp_alpha/下的.py文件的文件名与/home/zhyu/alpha_submit下的文件夹名

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    path_gp_alpha = config['SUBMIT']['PATH_GP_ALPHA']
    path_submit = config['SUBMIT']['PATH_SUBMIT']
    submit_prefix = config['SUBMIT']['SUBMIT_PREFIX']

    idx_gp_alpha = []
    for file_name in os.listdir(path_gp_alpha):
        if file_name[:9] == 'gp_alpha_' and file_name.split('.')[-1] == 'py' and file_name.split('.')[0][9:].isnumeric():
            idx_gp_alpha.append(file_name.split('.')[0][9:])

    idx_submitted = []
    for file_name in os.listdir(path_submit):
        if file_name[:len(submit_prefix)] == submit_prefix:  # 提交名字里不能有下划线, 所以不一样
            idx_submitted.append(file_name[len(submit_prefix):])

    idx_to_submit = sorted([idx for idx in idx_gp_alpha if idx not in idx_submitted])
    print('本次提交以下标号的{}: {}'.format(submit_prefix, idx_to_submit))

    for idx in idx_to_submit:
        text_alpha_alpha = """<alpha istest="true">
</alpha>
"""
        text_alpha_readme = """因子的配置文件。
"""
        text_alpha_submit = f"""<asim submit="alpha" timetype="daily" asset="equity" region="cn"
        startdate="20160104" backdays="244" delay="1"
        weight="" xmintime="" advisor="jyxie"
        name="{submit_prefix}{idx}" subUniv="UnivTOP4000" op="new" brickid="alpha_4.0.0_{submit_prefix}{idx}_1"
        category="random" sys_name="asim_py" sys_ver="4.0.0" importbricks=""
        notification="true"
        memo=""
/>
"""
        with open(os.path.join(path_gp_alpha, 'gp_alpha_{}.py'.format(idx)), 'r') as f:
            text_code_py = f.read()

        text_code_readme = """遗传规划阿尔法
"""
        text_code_submit = f"""<asim submit="code" bricktype="alpha" timetype="daily" asset="equity"
        sys_name="asim_py" sys_ver="4.0.0"
        notification="true"
        name="{submit_prefix}{idx}" op="new" >
</asim>
"""
        path_to_submit = '{}/{}{}'.format(path_submit, submit_prefix, idx)

        os.mkdir(path_to_submit)
        os.mkdir(os.path.join(path_to_submit, 'code'))
        os.mkdir(os.path.join(path_to_submit, 'alpha'))

        with open(os.path.join(path_to_submit, 'code/{}{}.pyx'.format(submit_prefix, idx)), 'w') as f:
            f.write(text_code_py)
        with open(os.path.join(path_to_submit, 'code/readme.txt'), 'w') as f:
            f.write(text_code_readme)
        with open(os.path.join(path_to_submit, 'code/submit.xml'), 'w') as f:
            f.write(text_code_submit)
        with open(os.path.join(path_to_submit, 'alpha/alpha.xml'), 'w') as f:
            f.write(text_alpha_alpha)
        with open(os.path.join(path_to_submit, 'alpha/readme.txt'), 'w') as f:
            f.write(text_alpha_readme)
        with open(os.path.join(path_to_submit, 'alpha/submit.xml'), 'w') as f:
            f.write(text_alpha_submit)

        # 运行提交的代码
        subprocess.call('/work/asim/aa_helper -s /home/zhyu/alpha_submit/{}{}/code/'.format(submit_prefix, idx), shell=True)
        time.sleep(10)
        subprocess.call('/work/asim/aa_helper -s /home/zhyu/alpha_submit/{}{}/alpha/'.format(submit_prefix, idx), shell=True)
        time.sleep(60)
