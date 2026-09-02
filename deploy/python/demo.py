import os
# cmd_str = r'python E:\repository\PaddleDetection\deploy\python\infer.py ' \
#            r'--model_dir=E:\data\DefectDet\ppyoloe_plus_l_defect_contrast ' \
#            r'--image_file=E:\repository\defectdet\BDDetection\dataddd\input\1701661721.7274697.jpg --device=GPU ' \
#            r'--output_dir=E:\repository\defectdet\BDDetection\dataddd\vis '\
#            r'--slice_infer --slice_size 640 640 --overlap_ratio 0.1 0.1'

# cmd_str = r'python E:\repository\PaddleDetection\deploy\ppyolo\infer.py ' \
#            r'--model_dir=E:\data\DefectDet\ppyoloe_plus_l_defect_contrast ' \
#            r'--image_file=E:\repository\defectdet\BDDetection\dataddd\input\1701661721.7274697.jpg ' \
#            r'--device=GPU ' \
#            r'--output_dir=E:\repository\defectdet\BDDetection\dataddd\vis '\
#            r'--slice_infer --slice_size 640 640 --overlap_ratio 0.1 0.1'
#


cmd_str = r'python E:\repository\defectdet\BDDetection\Bin\Win64\plugin\pythonv3plugin\libs\ppyolo\infer.py ' \
           r'--model_dir=E:\repository\defectdet\BDDetection\Bin\Win64\plugin\pythonv3plugin\libs\ppyolo\models ' \
           r'--image_file=E:\repository\defectdet\BDDetection\dataddd\input\1701661721.7274697.jpg --device=GPU ' \
           r'--output_dir=E:\repository\defectdet\BDDetection\dataddd\vis '\
           r'--slice_infer --slice_size 640 640 --overlap_ratio 0.1 0.1'



def cmd_run(cmd_str):
    print('[runing]：\n', cmd_str)
    os.system(cmd_str)

if __name__ == '__main__':
    pass
    cmd_run(cmd_str)