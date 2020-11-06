from opt import parse_args
import os
from tqdm import tqdm
import multiprocessing
from template_format import TheCao
import json
from template_format import init_template


def gen(ind):
    template_img, template_anno, ocr_imgs, ocr_annos = template[ind]
    if ocr_imgs is None:
        pass 
    # Save template images
    if opt.output_template is not None:
        template_img.save(os.path.join(opt.output_template,
                        "{}_{}.png".format(opt.type_template, ind)))
        with open(os.path.join(opt.output_template, 
                    "{}_{}.json".format(opt.type_template, ind)), "w") as f:
            json.dump(template_anno,  f, ensure_ascii=False, indent=4)
    
    if opt.output_ocr is not None:
        for k, ocr_img in enumerate(ocr_imgs):
            ocr_img.save(os.path.join(opt.output_ocr,
                            "{}_{}.png".format(opt.type_template, ind)))
            # with open(os.path.join(opt.output_ocr, 
            #             "{}_{}_{}.txt".format(opt.type_template, ind, k)), "w") as f:
            #     f.write(ocr_annos[k])


def main(opt):
    if opt.output_template is not None:
        os.makedirs(opt.output_template, exist_ok=True)
    
    if opt.output_ocr is not None:
        os.makedirs(opt.output_ocr, exist_ok=True)
        ocr_output=True
    else:
        ocr_output=False
    
    if opt.output_augment is not None:
        os.makedirs(opt.output_augment, exist_ok=True)

    global template
    # template = TheCao(template_dir=opt.template_path, 
    #                 font_dir=opt.font_path, random_mode=True)
    template = init_template(opt.type_template, template_dir=opt.template_path, 
                           font_dir=opt.font_path, random_mode=False, ocr_output=ocr_output)

    pool = multiprocessing.Pool(opt.num_workers)
    output = list(tqdm(pool.imap(gen, range(opt.num_sample)), 
                      total=opt.num_sample, desc="Generating "))
    pool.terminate()
    # for i in tqdm(range(opt.num_sample)):
    #     gen(i)



if __name__ == "__main__":
    opt = parse_args()

    main(opt)

