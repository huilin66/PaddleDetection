import os

def cmd_run(cmd_str):
    os.system(cmd_str)

if __name__ == '__main__':
    pass

    # cmd_str_ppyoloe_train = r'python E:\repository\PaddleDetection27\tools\train.py ' \
    #                        r'-c E:\repository\PaddleDetection27\configs\ppyoloe\ppyoloe_l_5w.yml ' \
    #                        r'--eval'
    cmd_str_ppyoloe_train = r'python E:\repository\PaddleDetection27\tools\train.py ' \
                           r'-c E:\repository\PaddleDetection27\configs\ppyoloe\ppyoloe_l_5wt.yml ' \
                           r'--eval'

    cmd_run(cmd_str_ppyoloe_train)


    # cmd_str_ppyoloe_eval = r'python E:\repository\PaddleDetection27\tools\eval.py ' \
    #                        r'-c E:\repository\PaddleDetection27\configs\ppyoloe\ppyoloe_l_5wt.yml ' \
    #                        r'-o weights=E:\best_model.pdparams'
    #
    #
    # cmd_run(cmd_str_ppyoloe_eval)


