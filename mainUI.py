import os
import cv2
import argparse
import glob
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.registry import ARCH_REGISTRY
from reflib.face.restoration import CFaceRestoration
from reflib.face.utils import CFaceUtils
import ttkbootstrap as ttk
from tkinter import filedialog
from tkinter.messagebox import showerror, askyesno
from tkinter import colorchooser
from PIL import Image, ImageOps, ImageTk, ImageFilter, ImageGrab
import threading
from time import sleep

# defining global variables
WIDTH = 1100
HEIGHT = 600
in_file_path = ""
pen_size = 3
pen_color = "black"
g_Option = 'None'
g_deviceInfo = 'cpu'
g_bInit = False

def inpainting(in_image_path, out_image_path):
    global g_bInit
    global g_deviceInfo

    # if(g_bInit == False):
        #. registry, model init
    network = ARCH_REGISTRY.get('FaceSuper')(dim=512, cbSize=512, conlist=['32', '64', '128']).to(g_deviceInfo)
    model_path = 'reflib/model/inpainting.pth'
    model_H = torch.load(model_path)['params_ema']
    network.load_state_dict(model_H)
    network.eval()
        # g_bInit = True

    fileName = os.path.basename(in_image_path)
    print(f'\tInpainting start ... {fileName}')

    # image size(512 * 512)
    infaceImg = cv2.imread(in_image_path)
    assert infaceImg.shape[:2] == (512, 512), 'The resolution of image must be 512x512.'
    infaceImg = img2tensor(infaceImg / 255., bgr2rgb=True, float32=True)
    normalize(infaceImg, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    infaceImg = infaceImg.unsqueeze(0).to(g_deviceInfo)

    try:
        with torch.no_grad():
            mask = torch.zeros(512, 512)
            mask_id = torch.sum(infaceImg[0], dim=0)
            mask[mask_id == 3] = 1.0
            mask = mask.view(1, 1, 512, 512).to(g_deviceInfo)

            # makse_img = tensor2img(mask, rgb2bgr=True, min_max=(-1, 1))
            outfaceImg = network(infaceImg, 1, False)[0]
            outfaceImg = (1 - mask) * infaceImg + mask * outfaceImg
            savefaceImg = tensor2img(outfaceImg, rgb2bgr=True, min_max=(-1, 1))

        del outfaceImg
        torch.cuda.empty_cache()
        # Save
        savefaceImg = savefaceImg.astype('uint8')
        imwrite(savefaceImg, out_image_path)
        print(f'\tInpainting OK ... {out_image_path}')

    except Exception as error:
        print(f'\tFailed inpainting: {error}')




def colorization(in_image_path, out_image_path):
    global g_bInit
    global g_deviceInfo

    # if(g_bInit == False):
    #. registry, model init
    network = ARCH_REGISTRY.get('FaceSuper')(dim=512, cbSize=1024, conlist=['32', '64', '128']).to(g_deviceInfo)
    model_path = 'reflib/model/colorization.pth'
    model_H = torch.load(model_path)['params_ema']
    network.load_state_dict(model_H)
    network.eval()
        # g_bInit = True

    fileName = os.path.basename(in_image_path)
    print(f'\tColorization start ... {fileName}')

    # image size(512 * 512)
    infaceImg = cv2.imread(in_image_path)
    assert infaceImg.shape[:2] == (512, 512), 'The resolution of image must be 512x512.'
    infaceImg = img2tensor(infaceImg / 255., bgr2rgb=True, float32=True)
    normalize(infaceImg, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    infaceImg = infaceImg.unsqueeze(0).to(g_deviceInfo)

    try:
        with torch.no_grad():

            outfaceImg = network(infaceImg, 0, False)[0]
            savefaceImg = tensor2img(outfaceImg, rgb2bgr=True, min_max=(-1, 1))

        del outfaceImg
        torch.cuda.empty_cache()
        # Save
        savefaceImg = savefaceImg.astype('uint8')
        imwrite(savefaceImg, out_image_path)
        print(f'\tColorization OK ... {out_image_path}')

    except Exception as error:
        print(f'\tFailed colorization: {error}')

# The unoptimized RealESRGAN is slow on CPU.
# def set_realESRGan():
#     from basicsr.archs.rrdbnet_arch import RRDBNet
#     from basicsr.utils.realesrgan_utils import RealESRGANer
#
#     use_half = False
#     if torch.cuda.is_available(): # set False in CPU/MPS mode
#         no_half_gpu_list = ['1650', '1660'] # set False for GPUs that don't support f16
#         if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
#             use_half = True
#
#     model = RRDBNet(
#                 num_in_ch=3,
#                 num_out_ch=3,
#                 num_feat=64,
#                 num_block=23,
#                 num_grow_ch=32,
#                 scale=2,
#     )
#     upsampler = RealESRGANer(
#                     scale=2,
#                     model_path="model/RealESRGAN_x2plus.pth",
#                     model=model,
#                     tile=args.bg_tile,
#                     tile_pad=40,
#                     pre_pad=0,
#                     half=use_half
#     )
#     return upsampler


def restoration(in_image_path, out_image_path, face_temp_path):
    global g_bInit
    global g_deviceInfo

    # if (g_bInit == False):
    # . registry, model init
    network = ARCH_REGISTRY.get('FaceSuper')(dim=512, cbSize=1024, conlist=['32', '64', '128', '256']).to(g_deviceInfo)
    model_path = 'reflib/model/restoration.pth'
    model_H = torch.load(model_path)['params_ema']
    network.load_state_dict(model_H)
    network.eval()
        # g_bInit = True

    fileName = os.path.basename(in_image_path)
    print(f'\tRestoration start ... {fileName}')

    faceRestor = CFaceRestoration(  upscale_factor= 2,  # Default : 2
                                    face_size=512,
                                    crop_ratio=(1, 1),
                                    det_model = 'retinaface_resnet50',
                                    save_ext='png',
                                    use_parse=True,
                                    device=g_deviceInfo)

    # isinstance(in_image_path, str)
    img = cv2.imread(in_image_path, cv2.IMREAD_COLOR)
    # faceRestor.init()
    faceRestor.read_image(img)
    # get landmark of face
    detFaces = faceRestor.get_face_landmarks_5(False, False, 640, eye_dist_threshold=5)
    print(f'\tNumber of face detection - ({detFaces} faces)')
    faceRestor.align_warp_face()

    for _, idx_face in enumerate(faceRestor.cropped_faces):

        idx_face_tensor = img2tensor(idx_face / 255., True, True)
        normalize(idx_face_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), True)
        idx_face_tensor = idx_face_tensor.unsqueeze(0).to(g_deviceInfo)

        try:
            with torch.no_grad():
                res_tensor = network(idx_face_tensor, 0.5, True)[0]
                restored_face = tensor2img(res_tensor, True, min_max=(-1, 1))
            del res_tensor
            torch.cuda.empty_cache()

            restored_face = restored_face.astype('uint8')
            faceRestor.add_restored_face(restored_face, idx_face)

        except Exception as error:
            print(f'\tFailed restoration: {error}')

    faceRestor.get_inverse_affine(None)
    savefaceImg = faceRestor.paste_orgimage(upsample_img=None, draw_box=False)
    for idx, (_, restored_face) in enumerate(zip(faceRestor.cropped_faces, faceRestor.restored_faces)):
        fileName = os.path.basename(in_image_path)
        face_image_path = f'{face_temp_path}{fileName}_{idx}.png'
        imwrite(restored_face, face_image_path)
    # Save
    imwrite(savefaceImg, out_image_path)
    print(f'\tRestoration OK ... {out_image_path}')

# function to open the image file
def open_image():
    global in_file_path
    global progressbar_button

    in_file_path = filedialog.askopenfilename(title="Open Image File",  filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.gif;*.bmp")])
    if in_file_path:
        global  image
        image = Image.open(in_file_path)
        image = image.resize((512, 512), Image.LANCZOS)
        image = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor="nw", image=image)

    progressbar_button['value'] = 0;
def result_image(result_file_path):
    global image, image_in
    image_in = Image.open(in_file_path)
    image_in = image_in.resize((512, 512), Image.LANCZOS)
    image_in = ImageTk.PhotoImage(image_in)
    canvas.create_image(0, 0, anchor="nw", image=image_in)

    image = Image.open(result_file_path)
    image = image.resize((512, 512), Image.LANCZOS)
    image = ImageTk.PhotoImage(image)
    canvas.create_image(550, 0, anchor="nw", image=image)

def apply_method(method):
    global g_Option
    if method == 'Inpainting' :
        g_Option = 'inpaint'
    elif method == 'Clorization' :
        g_Option = 'color'
    elif method == 'Restoration' :
        g_Option = 'restor'
    else:
        g_Option = 'None'

# def progress_func():
#     global progressbar_button
#     print('\tprogress_func -----------')
#     progressbar_button['value'] = 20
#     print('\tprogress_func -------End ')

def start_proc():
    g_deviceInfo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fileName = os.path.basename(in_file_path)
    global g_Option
    global progressbar_button

    progressbar_button['value'] = 10;
    sleep(1)
    if g_Option == 'inpaint':
        out_imgPath = f'./result/inpainting/{fileName}'
        inpainting(in_file_path, out_imgPath)
        # th = threading.Thread(target=inpainting, args=(in_file_path, out_imgPath))
        # th.start()
        # th.join()
        result_image(out_imgPath)
    elif g_Option == 'color':
        out_imgPath = f'./result/colorization/{fileName}'
        colorization(in_file_path, out_imgPath)
        result_image(out_imgPath)
    elif g_Option == 'restor':
        out_imgPath = f'./result/restoration/{fileName}'
        out_FaceTempPath = f'./result/restoration/face/'
        restoration(in_file_path, out_imgPath, out_FaceTempPath)
        result_image(out_imgPath)

    progressbar_button['value'] = 100;


root = ttk.Window(themename="cosmo")
root.title("Face Super Restoration")
root.geometry("1300x600")
root.resizable(0, 0)
icon = ttk.PhotoImage(file='./icon/icon.png')
root.iconphoto(False, icon)

# the left frame to contain the 4 buttons
left_frame = ttk.Frame(root, width=200, height=600)
left_frame.pack(side="left", fill="y")

# the right canvas for displaying the image
canvas = ttk.Canvas(root, width=WIDTH, height=HEIGHT)
canvas.pack()
# label
filter_label = ttk.Label(left_frame, text="Method : ", background="white")
filter_label.pack(padx=0, pady=2)

# a list of Method
image_methodss = ["Inpainting", "Clorization", "Restoration"]

# combobox for the filters
method_combobox = ttk.Combobox(left_frame, values=image_methodss, width=15)
method_combobox.pack(padx=10, pady=5)

# binding the apply_filter function to the combobox
method_combobox.bind("<<ComboboxSelected>>", lambda event: apply_method(method_combobox.get()))

# loading the icons for the 4 buttons
open_icon = ttk.PhotoImage(file='./icon/add.png').subsample(8, 8)
start_icon = ttk.PhotoImage(file='./icon/start.png').subsample(8, 8)

# button for adding/opening the image file
image_button = ttk.Button(left_frame, image=open_icon, bootstyle="light", command=open_image)
image_button.pack(pady=5)
# button for processing the image file
start_button = ttk.Button(left_frame, image=start_icon, bootstyle="light", command=start_proc)
start_button.pack(pady=5)

# progressbar
progressbar_button = ttk.Progressbar(left_frame, orient='horizontal', mode='determinate', length=200)
progressbar_button.pack(padx=15, pady=100)
# progressbar_button.start()

root.mainloop()