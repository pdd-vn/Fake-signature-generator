import argparse
import os 

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_sample", 
                        default=10, 
                        type=int,
                        required=False, 
                        help="Number of generated sample")

    parser.add_argument("--type_template", 
                        type=str, 
                        required=True,
                        choices=["cc", "cm", "pp", "dl", "tc", "cmca", "cmqd"],
                        help="Generate in new form of type")

    TYPE = {
        'cc': "Can cuoc cong dan",
        'cm': "Chung minh nhan dan",
        'pp': "Passport",
        'pp2': "Passport two page",
        'dl': "Driver license",
        'tc': "the cao",
        'cmqd': "Chung minh quan doi",
        'cmca': "Chung minh cong an"
    }

    parser.add_argument("--template_path",
                        default="template",
                        type=str, 
                        required=False, 
                        help="Template folder")
            
    parser.add_argument("--font_path",
                        default="font",
                        type=str, 
                        required=False, 
                        help="font folder")

    parser.add_argument("--output_ocr",
                        default=None,
                        type=str, 
                        required=False, 
                        help="Generate data ocr in random background or template")

    parser.add_argument("--output_template", 
                        default=None, 
                        type=str,
                        required=False, 
                        help="Folder path to output template dataset")

    parser.add_argument("--output_augment",
                        default=None,
                        type=str, 
                        required=False, 
                        help="Generate augmented template in random background")

    parser.add_argument("--num_workers",
                        default=8, 
                        type=int,
                        required=False,
                        help="Number of core use for multiprocessing.")
    
    opt = parser.parse_args()
    return opt
    